from __future__ import annotations
from pathlib import Path
from typing import Optional
from string import Template

ROOT = Path(__file__).resolve().parents[1]  # project root
HW_FILE = ROOT / "prompts/hardware/gpu_specs.py"

from prompts.generate_custom_cuda import _load_gpu_spec  # 按需调整导入路径

_OPTIMIZATION_PROMPT_TEMPLATE = Template("""\
# ----------  Previously generated kernels ----------
$history_block

You are a CUDA-kernel optimization specialist.

Target GPU: **NVIDIA $gpu_name ($gpu_arch)**
$gpu_items

Analyze the provided architecture and kernel history to produce an improved CUDA kernel.

[ARCHITECTURE FILE]
```python
$arch_src
```

GOAL
────
- Improve latency and throughput on the target GPU.
- Maintain correctness within atol=1e-4 or rtol=1e-4.
- Preserve the public Python API (same inputs/outputs, shapes, dtypes).

CONSTRAINTS
───────────
- No test code, prints, timing, CLI.
- No extra prose or markdown outside the code block.
- Return exactly ONE fenced code block labeled `python`.

OUTPUT RULES (STRICT) ────────────────────────────────────────────────
1. Reply with **one—and only one—fenced Python block**.  No prose.
2. The block must be directly runnable:
       python model_new.py
3. Inside the block, follow **exactly** this order:
   1. Imports – `torch`, `torch.nn`, `load_inline`.
   2. `source` – triple‑quoted CUDA string(s) (kernel + host wrapper).
   3. `cpp_src` – prototypes for *all* kernels you expose.
   4. **One** `load_inline` call per kernel group.
   5. `class ModelNew(nn.Module)` – mirrors original inputs/outputs but calls
      your CUDA kernels.
4. **Do NOT include** testing code, `if __name__ == "__main__"`, or extra prose.

# ==========================================================
# ❶ OUTPUT FORMAT – Copy exactly
Return the fixed script wrapped like this – no extra text:

```python
# <your corrected code>
```
# ==========================================================
""")

def build_optimization_prompt(
    arch_path: Path,
    gpu_name: Optional[str] = None,
    *,
    history_block: str = "",
) -> str:
    """Build LLM prompt for CUDA‑kernel optimisation (optimization phase)."""
    gpu_info = _load_gpu_spec()

    if gpu_name is None:
        try:
            import torch
            gpu_name = torch.cuda.get_device_name(0)
        except Exception as exc:
            raise RuntimeError("CUDA device not found – pass --gpu <name>.") from exc

    if gpu_name not in gpu_info:
        raise KeyError(f"{gpu_name} not present in GPU_SPEC_INFO")

    info = gpu_info[gpu_name]
    gpu_arch = info.get("GPU Architecture", "Unknown")
    gpu_items = "\n".join(f"• {k}: {v}" for k, v in info.items() if k != "GPU Architecture")

    arch_src = Path(arch_path).read_text().strip()
    hist = history_block or "(None)\n"

    return _OPTIMIZATION_PROMPT_TEMPLATE.substitute(
        gpu_name=gpu_name,
        gpu_arch=gpu_arch,
        gpu_items=gpu_items,
        arch_src=arch_src,
        history_block=hist,
    )
