import argparse
import signal
import sys
import time
import uuid
from collections.abc import Callable

from sqlalchemy import select
from sqlalchemy.orm import Session

from forge import gpu
from forge.config import get_settings
from forge.db import get_engine
from forge.errors import ApiError
from forge.models import DatasetBuild, Job, Model, Run, Tokenizer
from forge.schemas import EstimateRequest, RunCreate, SpeedTestCreate
from forge.train.presets import PRESETS
from forge.train.settings import TrainSettings

SUMMARY = "Train from a terminal, for a GPU with no browser (a lab machine, a job queue)."
POLL_S = 2.0
ACTIVE = ("pending", "running")


class Problem(Exception):
    """Something the person running this needs to read. One plain sentence."""


def _session() -> Session:
    return Session(get_engine())


def _api(call: Callable, *args):
    try:
        return call(*args)
    except ApiError as problem:
        raise Problem(problem.message) from problem


def _find(session: Session, table, text: str, what: str):
    """A row by its name, its id, or the start of its id."""
    rows = list(session.scalars(select(table)))
    exact = [row for row in rows if row.name == text or str(row.id) == text]
    if not exact:
        exact = [row for row in rows if str(row.id).startswith(text.lower())]
    if len(exact) == 1:
        return exact[0]
    if not exact:
        raise Problem(f"No {what} called {text!r}. `forge list` shows them.")
    raise Problem(f"More than one {what} matches {text!r}. Use its id.")


def _hours(seconds: float) -> str:
    if seconds < 3600:
        return f"{seconds / 60:.0f} min"
    return f"{seconds / 3600:.1f} h"


def cmd_info(args: argparse.Namespace) -> None:
    reading = gpu.read()
    print(f"Data folder: {get_settings().data_dir}")
    if not reading.cards:
        print("Graphics card: none found" + (f" ({reading.said})" if reading.said else ""))
    for card in reading.cards:
        print(
            f"Graphics card {card.index}: {card.name}, "
            f"{card.free_bytes / 2**30:.1f} GB free of {card.memory_bytes / 2**30:.1f} GB"
        )
    try:
        import torch
    except ImportError:
        print("PyTorch: not installed. Install the train extra from the README.")
        return
    cuda = "can use the graphics card" if torch.cuda.is_available() else "CPU only"
    print(f"PyTorch: {torch.__version__}, {cuda}")


def cmd_list(args: argparse.Namespace) -> None:
    with _session() as session:
        if args.what in ("models", "all"):
            print("Models:")
            for model in session.scalars(select(Model).order_by(Model.created_at)):
                vocabulary = (
                    session.get(Tokenizer, model.tokenizer_id) if model.tokenizer_id else None
                )
                print(
                    f"  {model.name}  ({model.param_count / 1e6:.1f}M, "
                    f"vocabulary {vocabulary.name if vocabulary else model.vocab_size})"
                    f"  {model.id}"
                )
        if args.what in ("sets", "all"):
            print("Training sets:")
            for build in session.scalars(select(DatasetBuild).order_by(DatasetBuild.created_at)):
                vocabulary = session.get(Tokenizer, build.tokenizer_id)
                print(
                    f"  {build.name}  ({build.status}, vocabulary "
                    f"{vocabulary.name if vocabulary else '?'})  {build.id}"
                )
        if args.what in ("runs", "all"):
            print("Runs:")
            for run in session.scalars(select(Run).order_by(Run.created_at)):
                print(
                    f"  {run.name}  ({run.status}, step {run.current_step:,} of "
                    f"{run.max_steps:,})  {run.id}"
                )


def _wait_job(job_id: uuid.UUID) -> Job:
    last = 0.0
    while True:
        with _session() as session:
            job = session.get(Job, job_id)
            assert job is not None
            if job.status not in ACTIVE:
                return job
            if time.monotonic() - last > EVERY_S:
                print(f"  {job.progress.get('message') or job.status}", flush=True)
                last = time.monotonic()
        time.sleep(POLL_S)


def cmd_speed_test(args: argparse.Namespace) -> None:
    from forge.measure_api import start_speed_test

    payload = SpeedTestCreate(
        preset=args.size, sequences_per_step=args.sequences, device=args.device
    )
    with _session() as session:
        job = _api(start_speed_test, payload, session)
    print(f"Timing {args.size} with {args.sequences} sequences per step...", flush=True)
    done = _wait_job(job.id)
    if done.status != "completed":
        raise Problem(done.error or f"The speed test {done.status}.")
    result = done.result
    peak = result.get("peak_memory_bytes")
    print(f"{result['device_name']}: {result['tokens_per_sec']:,.0f} word pieces per second")
    if peak:
        print(f"Peak card memory: {peak / 2**30:.2f} GB")


def cmd_budget(args: argparse.Namespace) -> None:
    from forge.measure_api import BUDGET_SHARE, estimate

    request = EstimateRequest(
        preset=args.size,
        train_tokens=1,
        sequences_per_step=args.sequences,
        budget_hours=args.hours,
    )
    with _session() as session:
        out = _api(estimate, request, session)
    if out.tokens_per_sec is None or out.budget_tokens is None:
        raise Problem(f"Run `forge speed-test --size {args.size}` on this computer first.")
    where = {
        "measured": f"measured on {out.device_name}",
        "scaled": f"scaled from a speed test of another size on {out.device_name}",
    }[out.speed_source]
    print(f"Speed: {out.tokens_per_sec:,.0f} word pieces per second ({where})")
    print(
        f"Budget: {out.tokens_per_sec:,.0f} x {args.hours:g} h x 3,600 s x "
        f"{BUDGET_SHARE:g} (time kept for checks and saves) = {out.budget_tokens:,} word pieces"
    )
    per_param = out.budget_tokens / out.parameters
    print(
        f"That is {per_param:.1f} word pieces per parameter for {out.parameters / 1e6:.0f}M "
        "parameters (about 20 per parameter is the usual target for a model to be well trained)."
    )
    if out.fits is False and out.suggestion:
        print(f"Memory: it does not fit. {out.suggestion}")
    print(f"To train it: forge train ... --tokens {out.budget_tokens}")


def _settings(args: argparse.Namespace) -> TrainSettings:
    chosen = {
        name: value
        for name, value in (
            ("batch_size", args.batch),
            ("micro_batch_size", args.micro),
            ("learning_rate", args.lr),
            ("max_steps", args.steps),
            ("eval_every", args.eval_every),
            ("save_every", args.save_every),
        )
        if value is not None
    }
    try:
        return TrainSettings(**chosen)
    except ValueError as problem:
        raise Problem(str(problem).splitlines()[-1]) from problem


def _create(session: Session, args: argparse.Namespace, name: str) -> Run:
    from forge.runs_api import create_run

    if not args.model or not args.set:
        raise Problem(f"There is no run called {name!r} yet. Give --model and --set to make it.")
    model = _find(session, Model, args.model, "model")
    build = _find(session, DatasetBuild, args.set, "training set")
    payload = RunCreate(
        name=name,
        model_id=model.id,
        build_id=build.id,
        device=args.device,
        seed=args.seed,
        train_tokens=args.tokens,
        settings=_settings(args),
    )
    out = _api(create_run, payload, session)
    print(f"Made the run {name!r}: {out.max_steps:,} steps.", flush=True)
    return _find(session, Run, str(out.id), "run")


def _begin(session: Session, run: Run) -> None:
    from forge.models import Checkpoint
    from forge.runs_api import resume_run, start_run

    saved = session.scalar(select(Checkpoint.step).where(Checkpoint.run_id == run.id).limit(1))
    if _training(session, run):
        print(f"{run.name!r} is already queued or training. Following it.", flush=True)
        return
    if run.status == "completed":
        raise Problem(f"{run.name!r} already finished.")
    if saved is not None:
        _api(resume_run, run.id, session)
        print(f"Resuming {run.name!r} from its last save point.", flush=True)
    else:
        _api(start_run, run.id, session)
        print(f"Starting {run.name!r}.", flush=True)


def _training(session: Session, run: Run) -> bool:
    from forge.runs_api import _active

    return run.status == "running" or _active(session, run) is not None


def _stop(run_id: uuid.UUID) -> None:
    from forge.runs_api import stop_run

    with _session() as session:
        try:
            stop_run(run_id, session)
        except ApiError:
            pass


def _follow(run_id: uuid.UUID, stop_after_s: float | None) -> Run:
    """Print progress until the run ends. Ctrl+C or the time limit stops it with a save."""
    began, last, stopping = time.monotonic(), 0.0, False
    while True:
        try:
            with _session() as session:
                run = session.get(Run, run_id)
                assert run is not None
                going = _training(session, run)
                session.expunge(run)
            if not going:
                return run
            now = time.monotonic()
            if now - last > EVERY_S:
                loss = f", loss {run.loss:.3f}" if run.loss is not None else ""
                print(
                    f"  step {run.current_step:,} of {run.max_steps:,}{loss}, "
                    f"{run.tokens_seen:,} word pieces, {_hours(now - began)} so far",
                    flush=True,
                )
                last = now
            if stop_after_s is not None and not stopping and now - began > stop_after_s:
                print("Time limit reached: saving and stopping.", flush=True)
                _stop(run_id)
                stopping = True
            time.sleep(POLL_S)
        except KeyboardInterrupt:
            if stopping:
                raise
            print("\nStopping with a save point. Press Ctrl+C again to leave now.", flush=True)
            _stop(run_id)
            stopping = True


def _report(run: Run) -> None:
    if run.status == "completed":
        print(f"Finished {run.name!r} at step {run.current_step:,}.")
    elif run.status == "stopped":
        print(f"Stopped {run.name!r} at step {run.current_step:,}. Run the same command to go on.")
    else:
        raise Problem(run.error or f"{run.name!r} ended as {run.status}.")


def cmd_train(args: argparse.Namespace) -> None:
    with _session() as session:
        existing = session.scalars(select(Run).where(Run.name == args.name)).all()
        if len(existing) > 1:
            raise Problem(f"More than one run is called {args.name!r}. Rename one in the app.")
        run = existing[0] if existing else _create(session, args, args.name)
        _begin(session, run)
        run_id = run.id
    stop_after = args.stop_after * 3600 if args.stop_after else None
    _report(_follow(run_id, stop_after))


def cmd_stop(args: argparse.Namespace) -> None:
    with _session() as session:
        run = _find(session, Run, args.name, "run")
    _stop(run.id)
    print(f"Asked {run.name!r} to save and stop.")


def cmd_sweep(args: argparse.Namespace) -> None:
    """Short runs that differ only in learning rate and batch size, then a table."""
    rates = [float(value) for value in args.lr.split(",")]
    batches = [int(value) for value in args.batch.split(",")]
    made = []
    with _session() as session:
        for batch in batches:
            for rate in rates:
                name = f"{args.prefix} lr {rate:g} batch {batch}"
                one = argparse.Namespace(
                    **{**vars(args), "lr": rate, "batch": batch, "tokens": None}
                )
                run = _create(session, one, name)
                _begin(session, run)
                made.append(run.id)
    print(f"{len(made)} runs queued. They train one after another.", flush=True)
    results = []
    for run_id in made:
        results.append(_follow(run_id, None))
    with _session() as session:
        rows = []
        for run in results:
            fresh = session.get(Run, run.id)
            assert fresh is not None
            rows.append(fresh)
    rows.sort(key=lambda run: (run.eval_loss is None, run.eval_loss or 0.0))
    print(f"\n{'Run':<40} {'Status':<10} {'Training loss':>14} {'Validation loss':>16}")
    for run in rows:
        loss = f"{run.loss:.4f}" if run.loss is not None else "-"
        val = f"{run.eval_loss:.4f}" if run.eval_loss is not None else "-"
        print(f"{run.name:<40} {run.status:<10} {loss:>14} {val:>16}")
    if rows and rows[0].eval_loss is not None:
        print(f"\nLowest validation loss: {rows[0].name}")


def _train_flags(page: argparse.ArgumentParser, sweep: bool = False) -> None:
    page.add_argument("--model", help="the model, by name or id")
    page.add_argument("--set", help="the training set, by name or id")
    if not sweep:
        length = page.add_mutually_exclusive_group()
        length.add_argument("--tokens", type=int, help="how many word pieces to train on")
        length.add_argument("--steps", type=int, help="or how many steps")
        page.add_argument("--batch", type=int, help="sequences per step (default 16)")
        page.add_argument("--lr", type=float, help="peak learning rate (default 6e-4)")
    else:
        page.add_argument("--steps", type=int, default=300, help="steps per run (default 300)")
        page.add_argument("--lr", default="3e-4,6e-4,1e-3", help="learning rates, commas")
        page.add_argument("--batch", default="16", help="sequences per step, commas")
        page.add_argument("--prefix", default="sweep", help="start of each run's name")
    page.add_argument("--micro", type=int, help="sequences per pass (default: the most that fit)")
    page.add_argument("--eval-every", type=int, help="steps between validation checks")
    page.add_argument("--save-every", type=int, help="steps between save points")
    page.add_argument("--seed", type=int, default=0)
    page.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="forge", description=SUMMARY)
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("info", help="graphics card, PyTorch and the data folder")

    page = commands.add_parser("list", help="models, training sets and runs")
    page.add_argument("what", nargs="?", choices=("models", "sets", "runs", "all"), default="all")

    sizes = sorted(PRESETS)
    page = commands.add_parser("speed-test", help="time a size on this computer")
    page.add_argument("--size", choices=sizes, required=True)
    page.add_argument("--sequences", type=int, default=4, help="sequences per step (default 4)")
    page.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")

    page = commands.add_parser("budget", help="word pieces that fit in your hours")
    page.add_argument("--size", choices=sizes, required=True)
    page.add_argument("--hours", type=float, required=True)
    page.add_argument("--sequences", type=int, default=4, help="sequences per step (default 4)")

    page = commands.add_parser("train", help="start a run, or resume it if it exists")
    page.add_argument("--name", required=True, help="the run's name")
    page.add_argument(
        "--stop-after", type=float, help="save and stop after this many hours (a job limit)"
    )
    _train_flags(page)

    page = commands.add_parser("stop", help="save and stop a run")
    page.add_argument("--name", required=True)

    page = commands.add_parser("sweep", help="short runs to pick learning rate and batch size")
    _train_flags(page, sweep=True)
    return parser


HANDLERS = {
    "info": cmd_info,
    "list": cmd_list,
    "speed-test": cmd_speed_test,
    "budget": cmd_budget,
    "train": cmd_train,
    "stop": cmd_stop,
    "sweep": cmd_sweep,
}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "sweep" and (not args.model or not args.set):
        print("A sweep needs --model and --set.", file=sys.stderr)
        return 2
    before = signal.signal(signal.SIGTERM, signal.default_int_handler)
    try:
        HANDLERS[args.command](args)
    except Problem as problem:
        print(problem, file=sys.stderr)
        return 1
    finally:
        signal.signal(signal.SIGTERM, before)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
