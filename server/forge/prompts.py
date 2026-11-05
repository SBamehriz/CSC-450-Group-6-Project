import uuid

from sqlalchemy.orm import Session

from forge.db import get_engine
from forge.jobs import report
from forge.models import Job


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
