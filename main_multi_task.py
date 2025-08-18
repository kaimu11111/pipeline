# main.py
from __future__ import annotations
import argparse
import re
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

import matplotlib
matplotlib.use("Agg")  # headless save
import matplotlib.pyplot as plt

from agents.query_server import query_server
from prompts.generate_custom_cuda import build_seed_prompt, system_prompt
from utils.compile_and_run import compare_and_bench
from utils.kernel_io import extract_code_block, save_kernel_code
from scripts.individual import KernelIndividual  # adjust path if needed
from prompts.error import COMPILE_ERROR
from prompts.optimization import build_optimization_prompt


# ------------------------- CLI -------------------------
def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Single-LLM self-iterative kernel generation/optimization")
    p.add_argument(
        "arch_py",
        type=Path,
        help="Path to a single task .py file OR a directory containing many tasks (.py)",
    )
    p.add_argument("--gpu", default="Quadro RTX 6000", help="GPU name in prompt spec")
    p.add_argument("--server_type", default="local", help="LLM provider (local, openai, deepseek, vllm, etc.)")
    p.add_argument("--server_address", default="localhost", help="LLM server address (for vllm/sglang)")
    p.add_argument("--server_port", type=int, default=8000, help="LLM server port (for vllm/sglang)")
    p.add_argument("--model_name", default="deepseek-ai/deepseek-coder-6.7b-instruct", help="LLM model")
    p.add_argument("--round", "-G", type=int, default=10, help="Number of generations per task")
    p.add_argument("--work_dir", type=Path, default=Path("run"), help="Output root directory")
    p.add_argument("--device", type=int, default=0, help="CUDA device index for benchmarking")
    p.add_argument("--warmup", type=int, default=5, help="Warm-up iterations")
    p.add_argument("--repeat", type=int, default=20, help="Timed iterations per benchmark")
    p.add_argument("--tol", type=float, default=1e-3, help="Max |err| tolerated")
    p.add_argument("--max_tokens", type=int, default=16384, help="LLM max new tokens")
    p.add_argument("--temperature", type=float, default=0.9, help="LLM temperature")
    p.add_argument("--top_p", type=float, default=1.0, help="LLM top_p")
    # multi-task controls
    p.add_argument("--first_n", type=int, default=0, help="When arch_py is a directory, take the first N tasks (sorted)")
    p.add_argument("--num_tasks", type=int, default=1, help="When sampling, how many tasks to pick (if >0 and first_n=0)")
    p.add_argument("--shuffle_seed", type=int, default=0, help="Random seed for sampling (0 = time)")
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


def _build_history_block(code_dir: Path, keep_last: int = 10) -> str:
    """Collect the CUDA `source` of the most recent *keep_last* kernel files from code_dir."""
    if not code_dir.exists():
        return "## Existing kernels\n(None yet)\n"

    files: List[Path] = sorted(
        list(code_dir.glob("*.py")) + list(code_dir.glob("*.cu")),
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


# ------------------- LLM & eval steps ------------------
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


def _llm_to_kernel(prompt: str, code_dir: Path, call_llm) -> KernelIndividual:
    """LLM -> code -> save -> KernelIndividual（不做评测）"""
    raw = call_llm(prompt)
    code = extract_code_block(raw) or raw  # fallback
    path = save_kernel_code(code, code_dir)
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
    phase: str = "seed",
    metrics_dir: Path | None = None,
) -> None:
    """评测并更新个体的 metrics/score；异常时填充失败信息，并在指定目录保存 metrics。"""
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

    # —— 无论成功/失败，都尝试保存 metrics —— 
    if metrics_dir is not None:
        try:
            saved = ind.save_metrics(metrics_dir)
            print(f"[{phase}] metrics saved to: {saved}")
        except Exception as save_exc:
            print(f"[{phase}] WARNING: failed to save metrics: {save_exc}")


# ---------------------- task helpers -------------------
def _collect_tasks(maybe_dir: Path) -> List[Path]:
    """若是目录返回其中所有 .py 文件（排序），若是文件返回 [该文件]。"""
    if maybe_dir.is_file():
        return [maybe_dir]
    if maybe_dir.is_dir():
        return sorted([p for p in maybe_dir.rglob("*.py") if p.is_file()])
    raise FileNotFoundError(f"{maybe_dir} not found")


def _pick_first_n(tasks: List[Path], n: int) -> List[Path]:
    n = max(1, min(max(n, 0), len(tasks)))
    return tasks[:n]


def _sample_tasks(all_tasks: List[Path], k: int, seed: int | None) -> List[Path]:
    if not all_tasks:
        raise RuntimeError("No .py tasks found.")
    k = max(1, min(k, len(all_tasks)))
    if seed is None or seed == 0:
        seed = int(time.time())
    rng = random.Random(seed)
    return rng.sample(all_tasks, k)


def _plot_scores(save_path: Path, scores: List[float], err_flags: List[bool], title: str):
    """画每轮分数曲线；错误轮用 x 标记。"""
    xs = list(range(len(scores)))
    plt.figure()
    plt.plot(xs, scores, marker="o")
    for x, y, bad in zip(xs, scores, err_flags):
        if bad:
            plt.scatter([x], [y], marker="x")
    plt.xlabel("Round")
    plt.ylabel("Speedup (ref/test)")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.5)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


# --------------------- single-task run -----------------
def _run_single_task(task_path: Path, args) -> Dict[str, Any]:
    # --- per-task directories: run/<stamp>_<task>/{code,evaluation,figures}
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    work_dir = (args.work_dir / f"{run_stamp}_{task_path.stem}").resolve()
    code_dir = work_dir / "code"
    eval_dir = work_dir / "evaluation"
    fig_dir = work_dir / "figures"
    code_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    call_llm = _make_llm_caller(args)

    current_kernel: Optional[KernelIndividual] = None
    best_kernel: Optional[KernelIndividual] = None
    best_score: float = float("-inf")

    scores: List[float] = []
    err_flags: List[bool] = []
    last_score_for_curve = 1.0  # default baseline for plotting on early failures

    for round_idx in range(args.round):
        print(f"[{task_path.name}] Round {round_idx}")

        if round_idx == 0:
            print("[Seed] Generating the initial kernel ...")
            seed_prompt = build_seed_prompt(arch_path=task_path, gpu_name=args.gpu)

            ind = _llm_to_kernel(seed_prompt, code_dir, call_llm)
            _bench_and_score(
                ind,
                ref_py=task_path,
                device_idx=args.device,
                warmup=args.warmup,
                repeat=args.repeat,
                tol=args.tol,
                phase="seed",
                metrics_dir=eval_dir,
            )

        else:
            is_runnable = bool(getattr(current_kernel, "metrics", {}).get("runnable", False)) if current_kernel else False

            if not is_runnable:
                print("[Repair] start repairing")
                error_log = _last_n_lines(getattr(current_kernel, "metrics", {}).get("message", "")) if current_kernel else ""
                repair_prompt = COMPILE_ERROR.substitute(
                    OLD_CODE=(current_kernel.code if current_kernel else ""),  # type: ignore[union-attr]
                    ERROR_LOG=error_log,
                )
                ind = _llm_to_kernel(repair_prompt, code_dir, call_llm)
                _bench_and_score(
                    ind,
                    ref_py=task_path,
                    device_idx=args.device,
                    warmup=args.warmup,
                    repeat=args.repeat,
                    tol=args.tol,
                    phase="seed",
                    metrics_dir=eval_dir,
                )
            else:
                print("Optimizing start")
                history_block = _build_history_block(code_dir, keep_last=10)
                opt_prompt = build_optimization_prompt(
                    arch_path=current_kernel.code_path,  # type: ignore[union-attr]
                    gpu_name=args.gpu,
                    history_block=history_block,
                )
                ind = _llm_to_kernel(opt_prompt, code_dir, call_llm)
                _bench_and_score(
                    ind,
                    ref_py=task_path,
                    device_idx=args.device,
                    warmup=args.warmup,
                    repeat=args.repeat,
                    tol=args.tol,
                    phase="seed",
                    metrics_dir=eval_dir,
                )

        # -------- update state + record curve --------
        current_kernel = ind
        runnable = bool(getattr(ind, "metrics", {}).get("runnable", False))
        this_score = ind.score if (ind.score is not None and runnable) else None

        if this_score is not None:
            last_score_for_curve = this_score
            scores.append(this_score)
            err_flags.append(False)
            if this_score > best_score:
                best_score = this_score
                best_kernel = ind
        else:
            # on failure: keep last score and mark error
            scores.append(last_score_for_curve)
            err_flags.append(True)

    # plot per-task curve
    fig_path = fig_dir / f"{task_path.stem}_score.png"
    _plot_scores(fig_path, scores, err_flags, title=f"{task_path.stem} (best={best_score:.4f})")
    print(f"[{task_path.name}] Figure saved to: {fig_path}")

    return {
        "task": str(task_path),
        "best_score": float(best_score) if best_score != float("-inf") else 0.0,
        "best_runnable": bool(getattr(best_kernel, "metrics", {}).get("runnable", False)) if best_kernel else False,
        "work_dir": str(work_dir),
        "figure": str(fig_path),
    }


# --------------------------- main ----------------------
def main():
    args = _build_arg_parser().parse_args()

    all_tasks = _collect_tasks(args.arch_py)

    # single file → run once
    if args.arch_py.is_file():
        res = _run_single_task(all_tasks[0], args)
        avg_speedup = res["best_score"]
        accuracy = 1.0 if res["best_runnable"] else 0.0
        print(f"[SUMMARY] {res}")
        print(f"[GLOBAL] Avg speedup={avg_speedup:.4f}, Accuracy={accuracy:.4f}")
        return

    # directory: first_n takes precedence; else optionally sample
    if args.first_n and args.first_n > 0:
        picked = _pick_first_n(all_tasks, args.first_n)
        print(f"[Task Picker] Found {len(all_tasks)} tasks, taking first {len(picked)} (sorted).")
    else:
        picked = _sample_tasks(all_tasks, args.num_tasks, args.shuffle_seed)
        print(f"[Task Picker] Found {len(all_tasks)} tasks, sampled {len(picked)} with seed={args.shuffle_seed}.")

    summary: List[Dict[str, Any]] = []
    for i, task in enumerate(picked, 1):
        print(f"\n===== [{i}/{len(picked)}] Running task: {task} =====")
        res = _run_single_task(task, args)
        summary.append(res)

    # global summary using each task's best kernel
    if summary:
        avg_speedup = sum(s["best_score"] for s in summary) / len(summary)
        accuracy = sum(1 for s in summary if s["best_runnable"]) / len(summary)
        print("\n===== SUMMARY =====")
        for s in summary:
            print(f"{s['task']}: best_score={s['best_score']:.4f}  runnable={s['best_runnable']}  fig={s['figure']}")
        print(f"\n[GLOBAL] Avg speedup={avg_speedup:.4f}, Accuracy={accuracy:.4f}")
    else:
        print("No tasks were run.")


if __name__ == "__main__":
    main()
