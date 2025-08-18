# prompts/error.py

"""Prompt template for automatic kernel repair.
Uses `string.Template` to avoid `{}` brace conflicts with C/CUDA code.
"""
from string import Template

COMPILE_ERROR = Template(
    """You are a senior CUDA‑extension developer.
Your job is to **FIX** the compilation or runtime errors in the Python script
shown below.

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

────────────────────────────────────────
OLD CODE (read‑only)
────────────────────────────────────────
$OLD_CODE

────────────────────────────────────────
ERROR LOG
────────────────────────────────────────
$ERROR_LOG

# ==========================================================
# ❶ OUTPUT FORMAT – Copy exactly
Return the fixed script wrapped like this – no extra text:

```python
# <your corrected code>
```
# ==========================================================
"""
)