# utils/kernel_io.py
"""Utility helpers for Mind‑Evolution CUDA‑kernel workflow.

This tiny module centralises two common I/O helpers that were previously
inlined in the end‑to‑end test script:

1. ``extract_code_block`` – extract first ```python ... ``` (or generic) code
   block from LLM output. Raises if none found.
2. ``save_kernel_code`` – writes extracted code to *kernels/* with a unique
   timestamped filename and returns the *Path* object.

Keeping them here avoids duplication across evolution loops / diagnostics.
"""
from __future__ import annotations

import datetime as dt
import re
from pathlib import Path
from typing import Final
import json
from typing import Any, Dict
__all__: Final = [
    "extract_code_block",
    "save_kernel_code",
]

# ---------------------------------------------------------------------------
# 1. Code‑block extraction
# ---------------------------------------------------------------------------
_CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.S)


def extract_code_block(text: str) -> str:
    """Return the **first** triple‑back‑ticked block in *text*.

    Parameters
    ----------
    text : str
        Raw LLM output that should include a python code block.

    Returns
    -------
    str
        The code inside the back‑ticks (stripped) with a trailing newline.

    Raises
    ------
    RuntimeError
        If no code block is found.
    """
    match = _CODE_BLOCK_RE.search(text)
    if not match:
        # 保存 LLM 原始输出
        from datetime import datetime
        dump_path = f"llm_output_error_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(dump_path, "w") as f:
            f.write(text)
        raise RuntimeError(f"No ``` code block found in LLM output – raw output saved to {dump_path}")
    
    return match.group(1).strip() + "\n"



# ---------------------------------------------------------------------------
# 2. Persist kernel to file
# ---------------------------------------------------------------------------

def save_kernel_code(code: str, out_dir: Path | str = "kernels") -> Path:
    """Save *code* to *out_dir/kernel_YYYYmmdd_HHMMSS.py* and return the path."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"kernel_{stamp}.py"
    path.write_text(code, encoding="utf-8")

    return path


# utils/kernel_io.py




def extract_json(raw: str) -> Any:
    """
    从字符串中提取第一个 JSON 并解析成 Python 对象。
    支持 fenced code block ```json ...``` 或直接 JSON。
    
    Args:
        raw: LLM 原始输出文本
    Returns:
        Python 对象 (dict / list)
    Raises:
        ValueError: 如果没有找到有效 JSON
    """
    if not isinstance(raw, str):
        raw = str(raw)

    # 尝试 ```json ...``` 格式
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw, re.IGNORECASE)
    if match:
        candidate = match.group(1).strip()
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    # 尝试匹配第一个 { ... } 或 [ ... ]
    match = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", raw)
    if match:
        candidate = match.group(1).strip()
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    # 全文直接解析
    try:
        return json.loads(raw.strip())
    except json.JSONDecodeError:
        raise ValueError(f"Failed to extract valid JSON from reply:\n{raw}")

def save_prompt_text(text: str, out_dir: Path, *, tag: str = "repair") -> Path:
    """
    Save *text* to out_dir/{tag}_YYYYMMDD-HHMMSS.txt and return the Path.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    ts   = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    path = out_dir / f"{tag}_{ts}.txt"
    path.write_text(text, encoding="utf-8")
    return path