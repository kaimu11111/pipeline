# prompts/error.py
"""
Prompt template for automatic kernel repair.
Uses `string.Template` to avoid `{}` brace conflicts with C/CUDA code.
Adds GPU hardware context and architecture source for better fixes.
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional
from string import Template

# Project roots (adjust if your tree differs)
ROOT = Path(__file__).resolve().parents[1]  # project root
HW_FILE = ROOT / "prompts/hardware/gpu_specs.py"

# Reuse your existing GPU spec loader
from prompts.generate_custom_cuda import _load_gpu_spec  # noqa: E402


COMPILE_ERROR = Template(
    """You are a senior CUDA-extension developer.
Your job is to **FIX** the compilation or runtime errors in the Python script
shown below.

OUTPUT RULES (STRICT) ────────────────────────────────────────────────
1. Inside the block, follow **exactly** this order:
   1. Imports – `torch`, `torch.nn`, `load_inline`.
   2. `source` – triple‑quoted CUDA string(s) (kernel + host wrapper).
   3. `cpp_src` – prototypes for *all* kernels you expose.
   4. **One** `load_inline` call per kernel group.
   5. `class ModelNew(nn.Module)` – mirrors original inputs/outputs but calls
      your CUDA kernels.
2. **Do NOT include** testing code, `if __name__ == "__main__"`, or extra prose.

────────────────────────────────────────
ERROR LOG
────────────────────────────────────────
$ERROR_LOG

────────────────────────────────────────
OLD CODE (read-only)
────────────────────────────────────────
$OLD_CODE

```python
# <your corrected code>
```
# ==========================================================
"""
)


def build_error_prompt(
    *,
    old_code: str,
    error_log: str,
    gpu_name: Optional[str] = None,
) -> str:
    """
    Build the error-repair prompt with GPU context + architecture source.

    Parameters
    ----------
    old_code : str
        The broken Python script content to show under OLD CODE.
    error_log : str
        The compiler/runtime error text to show under ERROR LOG.
    arch_path : Path
        Path to the reference architecture Python file to display.
    gpu_name : Optional[str]
        Human-readable GPU name key to lookup in gpu_specs.
        If None, attempts torch.cuda.get_device_name(0).

    Returns
    -------
    str
        The final prompt string to send to the LLM.
    """
    # Load the GPU spec dictionary
    gpu_info = _load_gpu_spec()

    # Resolve GPU name
    if gpu_name is None:
        try:
            import torch  # local import to avoid hard dependency if CPU-only
            gpu_name = torch.cuda.get_device_name(0)
        except Exception as exc:
            raise RuntimeError("CUDA device not found – pass --gpu <name>.") from exc

    if gpu_name not in gpu_info:
        raise KeyError(f"{gpu_name} not present in GPU_SPEC_INFO (file: {HW_FILE})")

    info = gpu_info[gpu_name]
    gpu_arch = info.get("GPU Architecture", "Unknown")

    # Bullet list of key specs except the arch line (already printed separately)
    gpu_items = "\n".join(
        f"• {k}: {v}" for k, v in info.items() if k != "GPU Architecture"
    )

    # Substitute all fields
    return COMPILE_ERROR.substitute(
        ERROR_LOG=error_log.strip(),
        OLD_CODE=old_code.strip(),
    )
