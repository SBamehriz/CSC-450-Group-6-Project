import argparse
from pathlib import Path


def split(source: Path, folder: Path, size_bytes: int) -> int:
    folder.mkdir(parents=True, exist_ok=True)
    count, buffer, held = 0, [], 0
    with source.open(encoding="utf-8") as text:
        for line in text:
            buffer.append(line)
            held += len(line.encode("utf-8"))
            if held >= size_bytes:
                count = _write(folder, source.stem, count, buffer)
                buffer, held = [], 0
    if buffer:
        count = _write(folder, source.stem, count, buffer)
    return count


def _write(folder: Path, stem: str, count: int, lines: list[str]) -> int:
    (folder / f"{stem}-{count:05}.txt").write_text("".join(lines), encoding="utf-8")
    return count + 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("source", type=Path, help="the big text file")
    parser.add_argument("folder", type=Path, help="where the pieces go")
    parser.add_argument("--size-mb", type=float, default=1.0, help="size of each piece (default 1)")
    args = parser.parse_args()
    count = split(args.source, args.folder, int(args.size_mb * 1_000_000))
    print(f"Wrote {count} files to {args.folder}. Upload them up to 200 at a time.")


if __name__ == "__main__":
    main()
