"""Local Qwen chat model, loaded lazily and shared across requests."""

from __future__ import annotations

import logging
import threading
from collections.abc import Iterator

from app.config import settings
from app.embeddings import resolve_device

logger = logging.getLogger(__name__)

_tokenizer = None
_model = None
_device = None
_load_lock = threading.Lock()
# One generation at a time: a single local model is not re-entrant on GPU.
_generate_lock = threading.Lock()


def _resolve_dtype():
    import torch

    if settings.dtype != "auto":
        return getattr(torch, settings.dtype)
    device = resolve_device()
    if device == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if device == "mps":
        return torch.float16
    return torch.float32


def load_model() -> tuple:
    """Load tokenizer + model once. Safe to call from anywhere."""
    global _tokenizer, _model, _device
    if _model is None:
        with _load_lock:
            if _model is None:
                from transformers import AutoModelForCausalLM, AutoTokenizer

                _device = resolve_device()
                dtype = _resolve_dtype()
                logger.info(
                    "Loading %s on %s (%s). First run downloads the weights.",
                    settings.llm_model,
                    _device,
                    dtype,
                )
                _tokenizer = AutoTokenizer.from_pretrained(settings.llm_model)
                _model = AutoModelForCausalLM.from_pretrained(
                    settings.llm_model,
                    torch_dtype=dtype,
                    device_map=_device if _device == "cuda" else None,
                    low_cpu_mem_usage=True,
                )
                if _device != "cuda":
                    _model = _model.to(_device)
                _model.eval()
                logger.info("Model ready")
    return _tokenizer, _model


def is_loaded() -> bool:
    return _model is not None


def _prepare_inputs(messages: list[dict[str, str]]):
    tokenizer, model = load_model()
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return tokenizer, model, tokenizer([prompt], return_tensors="pt").to(model.device)


def _generation_kwargs(max_new_tokens: int | None, temperature: float | None) -> dict:
    temp = settings.temperature if temperature is None else temperature
    kwargs = {
        "max_new_tokens": max_new_tokens or settings.max_new_tokens,
        "do_sample": temp > 0,
        "repetition_penalty": 1.05,
    }
    if temp > 0:
        kwargs["temperature"] = temp
        kwargs["top_p"] = settings.top_p
    return kwargs


def generate(
    messages: list[dict[str, str]],
    max_new_tokens: int | None = None,
    temperature: float | None = None,
) -> str:
    """Blocking generation; returns the full answer."""
    import torch

    tokenizer, model, inputs = _prepare_inputs(messages)
    with _generate_lock, torch.inference_mode():
        output = model.generate(
            **inputs,
            pad_token_id=tokenizer.eos_token_id,
            **_generation_kwargs(max_new_tokens, temperature),
        )
    new_tokens = output[0][inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def stream(
    messages: list[dict[str, str]],
    max_new_tokens: int | None = None,
    temperature: float | None = None,
) -> Iterator[str]:
    """Yield the answer token by token so the UI can type it out."""
    import torch
    from transformers import TextIteratorStreamer

    tokenizer, model, inputs = _prepare_inputs(messages)
    streamer = TextIteratorStreamer(
        tokenizer, skip_prompt=True, skip_special_tokens=True
    )
    kwargs = {
        **inputs,
        **_generation_kwargs(max_new_tokens, temperature),
        "streamer": streamer,
        "pad_token_id": tokenizer.eos_token_id,
    }

    def _run() -> None:
        with _generate_lock, torch.inference_mode():
            model.generate(**kwargs)

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    for piece in streamer:
        if piece:
            yield piece
    thread.join()
