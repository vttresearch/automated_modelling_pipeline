"""Bridge LoRA-fine-tuned TimesFM (Transformers) weights into the native checkpoint.

TimesFM 2.5 ships in two weight-identical formats:

* ``google/timesfm-2.5-200m-transformers`` — HuggingFace ``TimesFm2_5ModelForPrediction``.
  Supports PEFT/LoRA fine-tuning (separate ``q_proj``/``k_proj``/``v_proj``/``o_proj``).
* ``google/timesfm-2.5-200m-pytorch`` — native ``timesfm`` package. Supports
  ``forecast_with_covariates`` (XReg) but uses a *fused* ``qkv_proj`` projection
  and has no PEFT support.

The two checkpoints are byte-identical apart from this attention layout:

    native  stacked_xf.{i}.attn.qkv_proj.weight  (3*d, d)  == cat([q, k, v], dim=0)
    native  stacked_xf.{i}.attn.out.weight       (d, d)    == o_proj.weight

This module merges a LoRA adapter into the Transformers model and converts the
result into a native-format state dict, so a fine-tuned model can run inference
through the native covariate (XReg) path — giving covariate support and an
apples-to-apples comparison with base TimesFM.
"""

from __future__ import annotations

import os
import re

# Required before importing torch on macOS to avoid an OpenMP double-load crash.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")

TRANSFORMERS_MODEL = "google/timesfm-2.5-200m-transformers"
NATIVE_MODEL = "google/timesfm-2.5-200m-pytorch"

# File name used to cache the converted native weights alongside the adapter.
NATIVE_WEIGHTS_FILENAME = "native_finetuned.safetensors"


def _layer_indices(state_dict) -> list[int]:
    """Return sorted transformer layer indices found in a Transformers state dict."""
    ids = set()
    for key in state_dict:
        m = re.match(r"model\.layers\.(\d+)\.", key)
        if m:
            ids.add(int(m.group(1)))
    return sorted(ids)


def build_native_finetuned_state_dict(adapter_dir: str, device: str = "cpu"):
    """Merge a PEFT LoRA adapter into TimesFM and return a *native* state dict.

    Parameters
    ----------
    adapter_dir : str
        Directory containing the saved PEFT LoRA adapter
        (``adapter_config.json`` + ``adapter_model.safetensors``).
    device : str
        Device used while loading/merging the Transformers model.

    Returns
    -------
    dict[str, torch.Tensor]
        A complete native-format state dict (all 232 keys) with the
        LoRA-fine-tuned attention projections fused into ``qkv_proj``.
    """
    import torch
    import timesfm
    from transformers import TimesFm2_5ModelForPrediction
    from peft import PeftModel

    if not os.path.isdir(adapter_dir):
        raise FileNotFoundError(f"LoRA adapter directory not found: {adapter_dir}")

    # 1. Load the Transformers base model and merge the LoRA adapter into it.
    base_hf = TimesFm2_5ModelForPrediction.from_pretrained(TRANSFORMERS_MODEL)
    merged = PeftModel.from_pretrained(base_hf, adapter_dir).merge_and_unload()
    hsd = merged.state_dict()

    # 2. Start from the native base weights (everything except attention q/k/v is
    #    unchanged by LoRA, so we only overwrite the fused qkv projection).
    native = timesfm.TimesFM_2p5_200M_torch.from_pretrained(NATIVE_MODEL)
    nsd = {k: v.clone() for k, v in native.model.state_dict().items()}

    # 3. Re-fuse merged q/k/v -> native qkv_proj for every transformer layer.
    for i in _layer_indices(hsd):
        q = hsd[f"model.layers.{i}.self_attn.q_proj.weight"]
        k = hsd[f"model.layers.{i}.self_attn.k_proj.weight"]
        v = hsd[f"model.layers.{i}.self_attn.v_proj.weight"]
        o = hsd[f"model.layers.{i}.self_attn.o_proj.weight"]

        qkv_key = f"stacked_xf.{i}.attn.qkv_proj.weight"
        out_key = f"stacked_xf.{i}.attn.out.weight"
        if qkv_key not in nsd:
            raise KeyError(f"Expected native key missing: {qkv_key}")

        fused = torch.cat([q, k, v], dim=0).to(nsd[qkv_key].dtype)
        if fused.shape != nsd[qkv_key].shape:
            raise ValueError(
                f"Shape mismatch for {qkv_key}: built {tuple(fused.shape)} "
                f"vs native {tuple(nsd[qkv_key].shape)}"
            )
        nsd[qkv_key] = fused.cpu()
        nsd[out_key] = o.to(nsd[out_key].dtype).cpu()

    # Free the Transformers/native loader objects' references early.
    del base_hf, merged, native
    return nsd


def export_native_finetuned(adapter_dir: str, out_path: str | None = None) -> str:
    """Convert a LoRA adapter to native weights and save them as safetensors.

    Parameters
    ----------
    adapter_dir : str
        Directory containing the saved PEFT LoRA adapter.
    out_path : str, optional
        Destination ``.safetensors`` file. Defaults to
        ``{adapter_dir}/native_finetuned.safetensors``.

    Returns
    -------
    str
        The path the native weights were written to.
    """
    from safetensors.torch import save_file

    if out_path is None:
        out_path = os.path.join(adapter_dir, NATIVE_WEIGHTS_FILENAME)

    nsd = build_native_finetuned_state_dict(adapter_dir)
    # safetensors requires contiguous tensors.
    nsd = {k: v.contiguous() for k, v in nsd.items()}
    save_file(nsd, out_path)
    return out_path


def load_native_finetuned_state_dict(adapter_dir: str):
    """Return a native fine-tuned state dict, using the cached export if present.

    Prefers ``{adapter_dir}/native_finetuned.safetensors`` (lightweight — only
    needs the ``timesfm`` + ``safetensors`` packages). Falls back to building the
    weights on the fly from the PEFT adapter (needs ``transformers`` + ``peft``).
    """
    native_path = os.path.join(adapter_dir, NATIVE_WEIGHTS_FILENAME)
    if os.path.exists(native_path):
        from safetensors.torch import load_file
        return load_file(native_path)
    return build_native_finetuned_state_dict(adapter_dir)
