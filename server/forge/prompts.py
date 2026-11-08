import uuid

from sqlalchemy.orm import Session

from forge.db import get_engine
from forge.jobs import report
from forge.models import EvalResult, Job
from forge.version import current_commit


def answer(loaded, prompt: str, settings: dict, seed: int) -> str:
    import torch

    tokenizer, model = loaded.tokenizer, loaded.model
    encoded = tokenizer(prompt, return_tensors="pt").to(loaded.device)
    # keep room in the window for the answer
    window = getattr(model.config, "n_positions", None) or getattr(
        model.config, "max_position_embeddings", 2048
    )
    keep = max(1, window - settings["max_new_tokens"])
    input_ids = encoded["input_ids"][:, -keep:]
    torch.manual_seed(seed)
    sampled = settings["temperature"] > 0
    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            max_new_tokens=settings["max_new_tokens"],
            do_sample=sampled,
            temperature=settings["temperature"] if sampled else None,
            top_k=settings["top_k"] if sampled else None,
            top_p=None,
            pad_token_id=tokenizer.pad_token_id,
        )
    return tokenizer.decode(output[0, input_ids.shape[1] :], skip_special_tokens=True)


def run(job_id: uuid.UUID) -> dict:
    from forge.measuring import load

    with Session(get_engine()) as session:
        job = session.get(Job, job_id)
        assert job is not None
        payload = dict(job.payload)
    settings = payload["settings"]
    prompts: list[str] = payload["prompts"]
    seed = int(settings["seed"])
    models, answers = [], []
    total = len(payload["model_ids"]) * len(prompts)
    done = 0
    for model_id in payload["model_ids"]:
        report(job_id, done, total, "Loading a model")
        loaded = load(uuid.UUID(model_id), None, payload.get("device", "auto"))
        models.append({"id": model_id, "name": loaded.name, "parameters": loaded.parameters})
        mine = []
        for index, prompt in enumerate(prompts):
            report(job_id, done, total, f"{loaded.name}: prompt {index + 1} of {len(prompts)}")
            mine.append(answer(loaded, prompt, settings, seed + index))
            done += 1
        answers.append(mine)
        del loaded
    return {
        "settings": settings,
        "models": models,
        "rows": [
            {"prompt": prompt, "answers": [mine[index] for mine in answers]}
            for index, prompt in enumerate(prompts)
        ],
    }


def finish(job_id: uuid.UUID, status: str, message: str | None) -> None:
    if status != "completed":
        return
    with Session(get_engine()) as session:
        job = session.get(Job, job_id)
        if not job or not job.result:
            return
        res = job.result
        settings = dict(res.get("settings") or {})
        models_list = res.get("models") or []
        rows = res.get("rows") or []
        code_ver = current_commit()
        for row in rows:
            prompt_text = row.get("prompt")
            row_answers = row.get("answers") or []
            for side_idx, model_info in enumerate(models_list):
                if side_idx >= len(row_answers):
                    continue
                m_id_str = model_info.get("id")
                session.add(
                    EvalResult(
                        id=uuid.uuid4(),
                        result_type="prompt",
                        model_id=uuid.UUID(m_id_str) if m_id_str else None,
                        model_name=model_info.get("name") or "unknown",
                        checkpoint_id=None,
                        evaluation_id=job_id,
                        settings=settings,
                        code_version=code_ver,
                        prompt=prompt_text,
                        answer=row_answers[side_idx],
                    )
                )
        session.commit()
