# main.py
from __future__ import annotations
import argparse
import re
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

from agents.query_server import query_server
from prompts.generate_custom_cuda import build_seed_prompt, system_prompt
from utils.compile_and_run import compare_and_bench
from utils.kernel_io import extract_code_block, save_kernel_code
from scripts.individual import KernelIndividual  # 按你的实际路径调整
from prompts.error import COMPILE_ERROR
from prompts.optimization import build_optimization_prompt


# ------------------------- CLI -------------------------
def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Single-LLM self-iterative kernel generation/optimization")
    p.add_argument("arch_py", type=Path, help="Reference model (baseline) containing class Model")
    p.add_argument("--gpu", default="Quadro RTX 6000", help="GPU name in prompt spec")
    p.add_argument("--server_type", default="local", help="LLM provider (local, openai, deepseek, etc.)")
    p.add_argument("--server_address", default="localhost", help="LLM server address (for vllm/sglang)")
    p.add_argument("--server_port", type=int, default=8000, help="LLM server port (for vllm/sglang)")
    p.add_argument("--model_name", default="deepseek-ai/deepseek-coder-6.7b-instruct", help="LLM model")
    p.add_argument("--round", "-G", type=int, default=10, help="Number of generations")
    p.add_argument("--work_dir", type=Path, default=Path("runs"), help="Output root directory")
    p.add_argument("--device", type=int, default=0, help="CUDA device index")
    p.add_argument("--warmup", type=int, default=5, help="Warm-up iterations")
    p.add_argument("--repeat", type=int, default=20, help="Timed iterations per benchmark")
    p.add_argument("--tol", type=float, default=1e-3, help="Max |err| tolerated")
    p.add_argument("--max_tokens", type=int, default=16384, help="LLM max new tokens")
    p.add_argument("--temperature", type=float, default=0.9, help="LLM temperature")
    p.add_argument("--top_p", type=float, default=1.0, help="LLM top_p")
    return p


# ---------------------- small utils --------------------
def _last_n_lines(text: str, n: int = 150) -> str:
    lines = text.splitlines()
    return "\n".join(lines[-n:]) if len(lines) > n else text


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _extract_full_cuda_source(text: str) -> str:
    """Extract CUDA source from a Python or markdown-like file.

    Order:
      1) ```cuda ... ``` fenced code
      2) source = \"\"\" ... \"\"\"
      3) fallback: raw text
    """
    m = re.search(r"```cuda\n(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    m = re.search(r"source\s*=\s*([\"']{3})(.*?)(?:\1)", text, flags=re.DOTALL)
    if m:
        return m.group(2).strip()
    return text.strip()


def _build_history_block(kernel_dir: Path, keep_last: int = 10) -> str:
    """Collect the CUDA `source` of the most recent *keep_last* kernel files."""
    if not kernel_dir.exists():
        return "## Existing kernels\n(None yet)\n"

    files: List[Path] = sorted(
        list(kernel_dir.glob("*.py")) + list(kernel_dir.glob("*.cu")),
        key=lambda p: p.stat().st_mtime,
    )[-keep_last:]

    if not files:
        return "## Existing kernels\n(None yet)\n"

    snippets: List[str] = []
    for idx, p in enumerate(files, 1):
        try:
            cuda_src = _extract_full_cuda_source(_read_text(p))
        except Exception:
            cuda_src = "(failed to read/extract)"
        snippets.append(f"### Kernel {idx} · {p.name}\n```cuda\n{cuda_src}\n```")

    return "## Existing kernels\n" + "\n\n".join(snippets) + "\n"


# ------------------- deduped core steps ----------------
def _make_llm_caller(args):
    def call_llm(prompt: str, sys_prompt: str = system_prompt) -> str:
        return query_server(
            prompt=prompt,
            system_prompt=sys_prompt,
            server_type=args.server_type,
            model_name=args.model_name,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            server_address=args.server_address,
            server_port=args.server_port,
        )
    return call_llm


def _llm_to_kernel(prompt: str, kernel_dir: Path, call_llm) -> KernelIndividual:
    """LLM -> code -> save -> KernelIndividual（不做评测）"""
    raw = call_llm(prompt)
    code = extract_code_block(raw)
    path = save_kernel_code(code, kernel_dir)
    ind = KernelIndividual(code)
    ind.code_path = path  # type: ignore[attr-defined]
    return ind


def _bench_and_score(
    ind: KernelIndividual,
    *,
    ref_py: Path,
    device_idx: int,
    warmup: int,
    repeat: int,
    tol: float,
    phase: str = "seed",  # 保持与原流程一致（统一写 seed）
    metrics_dir : Path,
) -> None:
    """评测并更新个体的 metrics/score；异常时填充失败信息"""
    try:
        metrics = compare_and_bench(
            ref_py=ref_py,
            test_py=ind.code_path,  # type: ignore[attr-defined]
            device_idx=device_idx,
            warmup=warmup,
            repeat=repeat,
            tol=tol,
        )
        metrics["runnable"] = True
        metrics["phase"] = phase
        speedup = metrics["ref_latency_ms"]["avg"] / max(1e-9, metrics["test_latency_ms"]["avg"])
        metrics["score"] = speedup

        ind.metrics = metrics
        ind.score = speedup
        print(f"[{phase}] score={speedup:.4f}")

    except Exception as exc:
        ind.metrics = {
            "runnable": False,
            "phase": phase,
            "error_type": exc.__class__.__name__,
            "message": _last_n_lines(str(exc)),
        }
        ind.score = float("-inf")
        print(f"[{phase}] failed. See metrics.message for details.")
    saved = ind.save_metrics(metrics_dir)





# --------------------------- main ----------------------
def main():
    args = _build_arg_parser().parse_args()

    # --- 目录准备 ---
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    work_dir = (args.work_dir / run_stamp).resolve()
    kernel_dir = work_dir / "kernels"
    eval_dir = work_dir / "evaluation"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    call_llm = _make_llm_caller(args)

    # --- 状态变量 ---
    current_kernel: Optional[KernelIndividual] = None
    current_score: Optional[float] = None
    best_kernel: Optional[KernelIndividual] = None
    best_score: float = float("-inf")

    for round_idx in range(args.round):
        print(f"Round {round_idx}")

        if round_idx == 0:
            # -------- Seed 生成（内联 _attempt）--------
            print("[Seed] Generating the initial kernel ...")
            seed_prompt = build_seed_prompt(arch_path=args.arch_py, gpu_name=args.gpu)

            # LLM -> 保存 -> 个体
            ind = _llm_to_kernel(seed_prompt, kernel_dir, call_llm)

            # 评测 & 打分
            _bench_and_score(
                ind,
                ref_py=args.arch_py,
                device_idx=args.device,
                warmup=args.warmup,
                repeat=args.repeat,
                tol=args.tol,
                phase="seed",
                metrics_dir = eval_dir
            )

            # 更新状态
            current_kernel = ind
            current_score = None if not getattr(ind, "metrics", {}).get("runnable", False) else ind.score
            if ind.score is not None and ind.score > best_score:
                best_score = ind.score
                best_kernel = ind

            continue

        # -------- 后续轮次：修复 or 优化 --------
        is_runnable = bool(getattr(current_kernel, "metrics", {}).get("runnable", False))

        if not is_runnable:
            # -------- 修复（内联 _attempt）--------
            print("[Repair] start repairing")
            error_log = _last_n_lines(getattr(current_kernel, "metrics", {}).get("message", ""))
            repair_prompt = COMPILE_ERROR.substitute(
                OLD_CODE=current_kernel.code,  # type: ignore[union-attr]
                ERROR_LOG=error_log,
            )

            ind = _llm_to_kernel(repair_prompt, kernel_dir, call_llm)

            _bench_and_score(
                ind,
                ref_py=args.arch_py,
                device_idx=args.device,
                warmup=args.warmup,
                repeat=args.repeat,
                tol=args.tol,
                phase="seed",  # 保持与原流程一致
                metrics_dir = eval_dir
            )

        else:
            # -------- 优化（内联 _attempt）--------
            print("Optimizing start")
            history_block = _build_history_block(kernel_dir, keep_last=10)
            opt_prompt = build_optimization_prompt(
                arch_path=current_kernel.code_path,  # type: ignore[union-attr]
                gpu_name=args.gpu,
                history_block=history_block,
            )

            ind = _llm_to_kernel(opt_prompt, kernel_dir, call_llm)

            _bench_and_score(
                ind,
                ref_py=args.arch_py,
                device_idx=args.device,
                warmup=args.warmup,
                repeat=args.repeat,
                tol=args.tol,
                phase="seed",  # 保持与原流程一致
                metrics_dir = eval_dir
            )

        # -------- 统一更新状态 --------
        current_kernel = ind
        current_score = None if not getattr(ind, "metrics", {}).get("runnable", False) else ind.score
        if ind.score is not None and ind.score > best_score:
            best_score = ind.score
            best_kernel = ind


if __name__ == "__main__":
    main()
