from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal


PROVIDER = Literal["openai", "huggingface"]


@dataclass(frozen=True)
class ModelConfig:
    provider: PROVIDER
    model: str
    params: dict[str, Any]


def _coerce_torch_dtype(value: Any) -> Any:
    """Allow YAML strings like 'float16'/'bfloat16'/'float32'."""
    if not isinstance(value, str):
        return value
    try:
        import torch
    except Exception:
        return value
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    return mapping.get(value.lower(), value)


def _materialize_hf_quantization_config(model_kwargs: dict[str, Any]) -> dict[str, Any]:
    """If quantization_config is a dict, try to materialize BitsAndBytesConfig."""
    if not isinstance(model_kwargs, dict):
        return model_kwargs
    qc = model_kwargs.get("quantization_config")
    if not isinstance(qc, dict):
        return model_kwargs

    try:
        from transformers import BitsAndBytesConfig  # type: ignore
    except Exception:
        # Leave as dict; user may be using a different quantization backend.
        return model_kwargs

    # Coerce common dtypes inside config as well.
    qc = {k: _coerce_torch_dtype(v) for k, v in qc.items()}
    try:
        model_kwargs = dict(model_kwargs)
        model_kwargs["quantization_config"] = BitsAndBytesConfig(**qc)
        return model_kwargs
    except Exception:
        return model_kwargs


def normalize_model_params(provider: PROVIDER, params: dict[str, Any]) -> dict[str, Any]:
    """Normalize YAML params into kwargs for `init_chat_model(..., model_provider=...)`."""
    params = dict(params or {})

    if provider == "huggingface":
        # Transformers requires temperature > 0. For deterministic extraction we
        # treat non-positive temperatures as greedy decoding.
        temp = params.get("temperature")
        if isinstance(temp, (int, float)) and temp <= 0:
            params.pop("temperature", None)
            params.setdefault("do_sample", False)
        if "do_sample" not in params:
            params["do_sample"] = False
        if params.get("do_sample") is False:
            # Neutral values suppress warnings while preserving greedy decoding.
            params.setdefault("temperature", 1.0)
            params.setdefault("top_p", 1.0)
            params.setdefault("top_k", 0)

        # Prefer explicit low-memory defaults for local inference when absent.
        params.setdefault("device_map", "cuda:0")

        # Common place to put HF model loading args is `model_kwargs`.
        model_kwargs = params.get("model_kwargs")
        if not isinstance(model_kwargs, dict):
            model_kwargs = {}
        if isinstance(model_kwargs, dict):
            # Coerce torch_dtype if provided as string.
            if "torch_dtype" in model_kwargs:
                model_kwargs = dict(model_kwargs)
                model_kwargs["torch_dtype"] = _coerce_torch_dtype(model_kwargs["torch_dtype"])
            else:
                model_kwargs = dict(model_kwargs)
                # bf16 where supported provides the best memory/perf tradeoff.
                model_kwargs["torch_dtype"] = _coerce_torch_dtype("bfloat16")
            model_kwargs = _materialize_hf_quantization_config(model_kwargs)
            params["model_kwargs"] = model_kwargs

        # Also coerce torch_dtype at top-level if someone puts it there.
        if "torch_dtype" in params:
            params["torch_dtype"] = _coerce_torch_dtype(params["torch_dtype"])

    return params


def load_model_config(path: str | Path) -> ModelConfig:
    path = Path(path)
    try:
        import yaml  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Missing dependency: pyyaml. Install with `pip install pyyaml`."
        ) from e

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Invalid config in {path}: expected a YAML mapping/object at top-level.")

    provider = data.get("provider")
    model = data.get("model")
    params = data.get("params", {})

    if provider not in ("openai", "huggingface"):
        raise ValueError(f"Invalid config in {path}: provider must be 'openai' or 'huggingface'.")
    if not isinstance(model, str) or not model.strip():
        raise ValueError(f"Invalid config in {path}: model must be a non-empty string.")
    if params is None:
        params = {}
    if not isinstance(params, dict):
        raise ValueError(f"Invalid config in {path}: params must be a mapping/object.")

    params = normalize_model_params(provider, params)
    return ModelConfig(provider=provider, model=model, params=params)

