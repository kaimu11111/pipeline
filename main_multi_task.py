# main.py
from __future__ import annotations
import argparse
import re
import random
import time
import json
import csv
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

import matplotlib
matplotlib.use("Agg")  # headless save
import matplotlib.pyplot as plt

from agents.query_server import query_server
from prompts.generate_custom_cuda import build_seed_prompt, default_system_prompt
from utils.compile_and_run import compare_and_bench
from utils.kernel_io import extract_code_block, save_kernel_code, extract_json
from scripts.individual import KernelIndividual  # adjust path if needed
from prompts.error import build_error_prompt
from prompts.optimization import build_optimization_prompt
from prompts.judger_repair import build_correctness_prompts
_INVOCATION_SPLITTER = "Invoked with:"

def _sanitize_error_message(exc: Exception) -> str:
    """去掉 pybind 把大张量打印出来的部分，只保留报错关键信息。"""
    msg = str(exc)
    if _INVOCATION_SPLITTER in msg:
        msg = msg.split(_INVOCATION_SPLITTER, 1)[0].rstrip()
    return msg

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
    p.add_argument("--temperature", type=float, default=0.2, help="LLM temperature")
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

    def call_llm(prompt: str, sys_prompt: Optional[str] = None) -> str:
        sp = default_system_prompt if sys_prompt is None else sys_prompt
        return query_server(
            prompt=prompt,
            system_prompt=sp,
            server_type=args.server_type,
            model_name=args.model_name,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            server_address=args.server_address,
            server_port=args.server_port,
        )
    return call_llm


def _llm_to_kernel(
    prompt: str,
    code_dir: Path,
    call_llm,
    io_dir: Path,
    round_idx,
    sys_prompt: Optional[str] = None,   # 新增：可选系统提示
) -> KernelIndividual:
    """LLM -> code -> save -> KernelIndividual（不做评测）"""
    raw = call_llm(prompt, sys_prompt=sys_prompt)  # 传给 call_llm
    reply_file = io_dir / f"{round_idx}_raw_reply.txt"
    reply_file.write_text(raw, encoding="utf-8")
    code = extract_code_block(raw) or raw  # fallback
    path = save_kernel_code(code, code_dir)
    ind = KernelIndividual(code)
    ind.code_path = path  # type: ignore[attr-defined]
    return ind

# ================== 顶层 worker：必须放在模块顶层，不能写在函数里 ==================
def _bench_worker_entry(test_py: str,
                        ref_py: str,
                        device_idx: int,
                        warmup: int,
                        repeat: int,
                        tol: float,
                        conn) -> None:
    """
    子进程入口：固定 GPU，调用 compare_and_bench，把结果或错误通过 Pipe 传回父进程。
    注意：这里传的是字符串路径，避免传不可 pickl e 的对象。
    """
    import torch
    from pathlib import Path

    try:
        if torch.cuda.is_available():
            torch.cuda.set_device(device_idx)

        res = compare_and_bench(
            ref_py=Path(ref_py),
            test_py=Path(test_py),
            device_idx=device_idx,
            warmup=warmup,
            repeat=repeat,
            tol=tol,
        )
        conn.send(("ok", res))
    except Exception as e:
        # 清洗错误信息（如果你的工具函数可用就用它们；否则退化为 str(e)）
        try:
            cleaned = _sanitize_error_message(e)
            msg = _last_n_lines(cleaned)
        except Exception:
            msg = str(e)
        conn.send(("err", msg))
    finally:
        # 子进程末尾尽量同步，让错误在本轮内暴露
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize(device_idx)
            except Exception:
                pass
        try:
            conn.close()
        except Exception:
            pass


# ================== 保持原功能的 _bench_and_score（使用 spawn + 顶层 worker） ==================
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
    """
    评测并更新个体的 metrics/score；异常时填充失败信息，并在指定目录保存 metrics。
    与原版功能一致，但把 compare_and_bench 放到 **spawn 子进程**中执行，隔离 CUDA 上下文。
    """
    import torch
    from multiprocessing import get_context

    ctx = get_context("spawn")
    parent_conn, child_conn = ctx.Pipe(duplex=False)

    # 只传可 pickl e 的参数：路径字符串等
    p = ctx.Process(
        target=_bench_worker_entry,
        args=(
            str(ind.code_path),  # type: ignore[attr-defined]
            str(ref_py),
            device_idx,
            warmup,
            repeat,
            tol,
            child_conn,
        ),
    )
    p.start()
    # 父进程不使用子端
    try:
        child_conn.close()
    except Exception:
        pass

    # 等待子进程结束并接收 payload
    p.join()
    payload = parent_conn.recv() if parent_conn.poll() else None
    try:
        parent_conn.close()
    except Exception:
        pass

    # —— 根据子进程返回更新 metrics/score（与原逻辑保持一致）——
    if isinstance(payload, tuple) and len(payload) == 2 and payload[0] in ("ok", "err"):
        tag, data = payload
        if tag == "ok":
            metrics = data
            metrics["runnable"] = True
            metrics["phase"] = phase
            speedup = metrics["ref_latency_ms"]["avg"] / max(1e-9, metrics["test_latency_ms"]["avg"])
            metrics["score"] = speedup

            ind.metrics = metrics
            ind.score = speedup
            print(f"[{phase}] score={speedup:.4f}")
        else:
            ind.metrics = {
                "runnable": False,
                "phase": phase,
                "error_type": "RuntimeError",
                "message": data,
            }
            ind.score = float("-inf")
            print(f"[{phase}] failed. See metrics.message for details.")
    else:
        # 子进程异常退出且未回传数据
        ind.metrics = {
            "runnable": False,
            "phase": phase,
            "error_type": "SubprocessCrashed",
            "message": "subprocess exited unexpectedly (no payload received)",
        }
        ind.score = float("-inf")
        print(f"[{phase}] failed. Subprocess crashed.")

    # —— 与原版一致：无论成功/失败都尝试保存 metrics ——
    if metrics_dir is not None:
        try:
            saved = ind.save_metrics(metrics_dir)
            print(f"[{phase}] metrics saved to: {saved}")
        except Exception as save_exc:
            print(f"[{phase}] WARNING: failed to save metrics: {save_exc}")

    # 父进程做一次轻量清理（非必须，但更稳）
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize(device_idx)
        except Exception:
            pass
        try:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except Exception:
            pass


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
def _run_single_task(task_path: Path, args, batch_dir: Path) -> Dict[str, Any]:
    # --- per-task directories under the SAME batch_dir
    task_root = (batch_dir / task_path.stem).resolve()
    code_dir = task_root / "code"
    eval_dir = task_root / "evaluation"
    fig_dir = task_root / "figures"
    io_dir  = eval_dir / "llm_io" 
    
    
    code_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    io_dir.mkdir(parents=True, exist_ok=True)
    
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
            prompt_file = io_dir / f"round{round_idx:03d}_seed_prompt.txt"
            prompt_file.write_text(seed_prompt, encoding="utf-8")
            ind = _llm_to_kernel(seed_prompt, code_dir, call_llm, io_dir, round_idx)
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
                
                
                problem_system_prompt, problem_prompt = build_correctness_prompts(error_log=error_log,
                                                                                  arch_path = task_path,
                                                                                  cuda_code = current_kernel.code)
                prompt_file = io_dir / f"round{round_idx:03d}_problem_identify_prompt.txt"
                prompt_file.write_text(problem_prompt, encoding="utf-8")
                raw = call_llm(problem_prompt,problem_system_prompt)
                reply_file = io_dir / f"{round_idx}_raw_problem_identify_reply.txt"
                reply_file.write_text(raw, encoding="utf-8")
                problem_json = extract_json(raw)
                
                repair_prompt = build_error_prompt(
                    old_code=current_kernel.code,
                    error_log=error_log,
                    problem = problem_json,
                    gpu_name=args.gpu,
                )
                prompt_file = io_dir / f"round{round_idx:03d}_repair_prompt.txt"
                prompt_file.write_text(repair_prompt, encoding="utf-8")
                ind = _llm_to_kernel(repair_prompt, code_dir, call_llm, io_dir, round_idx)
                _bench_and_score(
                    ind,
                    ref_py=task_path,
                    device_idx=args.device,
                    warmup=args.warmup,
                    repeat=args.repeat,
                    tol=args.tol,
                    phase="repair",
                    metrics_dir=eval_dir,
                )
            else:
                print("Optimizing start")
                ##TODO
                '''
                Add judge here for the optimization strategy, after get the strategy, give it to the opt_prompt
                '''
                history_block = _build_history_block(code_dir, keep_last=10)
                opt_prompt = build_optimization_prompt(
                    arch_path=current_kernel.code_path,  # type: ignore[union-attr]
                    gpu_name=args.gpu,
                    history_block=history_block,
                )
                prompt_file = io_dir / f"round{round_idx:03d}_opt_prompt.txt"
                prompt_file.write_text(opt_prompt, encoding="utf-8")
                ind = _llm_to_kernel(opt_prompt, code_dir, call_llm, io_dir, round_idx)
                _bench_and_score(
                    ind,
                    ref_py=task_path,
                    device_idx=args.device,
                    warmup=args.warmup,
                    repeat=args.repeat,
                    tol=args.tol,
                    phase="opt",
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
        "task_dir": str(task_root),
        "figure": str(fig_path),
    }


# --------------------- summary saving ------------------
def _save_global_summary(batch_dir: Path, summary: List[Dict[str, Any]], avg_speedup: float, accuracy: float) -> None:
    """Save summary.json and summary.csv under the batch_dir."""
    batch_dir.mkdir(parents=True, exist_ok=True)

    # JSON
    out_json = {
        "avg_speedup": avg_speedup,
        "accuracy": accuracy,
        "num_tasks": len(summary),
        "tasks": summary,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    (batch_dir / "summary.json").write_text(json.dumps(out_json, indent=2), encoding="utf-8")

    # CSV
    csv_path = batch_dir / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["task", "best_score", "best_runnable", "task_dir", "figure"])
        for s in summary:
            writer.writerow([s["task"], f'{s["best_score"]:.6f}', int(bool(s["best_runnable"])), s["task_dir"], s["figure"]])
        writer.writerow([])
        writer.writerow(["avg_speedup", f"{avg_speedup:.6f}"])
        writer.writerow(["accuracy", f"{accuracy:.6f}"])

    print(f"[GLOBAL] Saved: {batch_dir/'summary.json'}")
    print(f"[GLOBAL] Saved: {csv_path}")


# --------------------------- main ----------------------
def main():
    args = _build_arg_parser().parse_args()

    all_tasks = _collect_tasks(args.arch_py)

    # ---- Create ONE batch folder for this run ----
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # batch name hints: single file uses file stem; directory uses 'batch'
    if args.arch_py.is_file():
        batch_name = f"{stamp}_{args.arch_py.stem}"
    else:
        # include sampling info for traceability
        pick_note = f"first{args.first_n}" if (args.first_n and args.first_n > 0) else f"num{args.num_tasks}_seed{args.shuffle_seed}"
        batch_name = f"{stamp}_batch_{pick_note}"
    batch_dir = (args.work_dir / batch_name).resolve()
    batch_dir.mkdir(parents=True, exist_ok=True)
    print(f"[BATCH] Output folder: {batch_dir}")

    # single file → run once (still inside the same batch folder)
    if args.arch_py.is_file():
        res = _run_single_task(all_tasks[0], args, batch_dir=batch_dir)
        summary = [res]
        avg_speedup = res["best_score"]
        accuracy = 1.0 if res["best_runnable"] else 0.0
        print(f"[SUMMARY] {res}")
        print(f"[GLOBAL] Avg speedup={avg_speedup:.4f}, Accuracy={accuracy:.4f}")
        _save_global_summary(batch_dir, summary, avg_speedup, accuracy)
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
        res = _run_single_task(task, args, batch_dir=batch_dir)
        summary.append(res)

    # global summary using each task's best kernel
    if summary:
        avg_speedup = sum(s["best_score"] for s in summary) / len(summary)
        accuracy = sum(1 for s in summary if s["best_runnable"]) / len(summary)
        print("\n===== SUMMARY =====")
        for s in summary:
            print(f"{s['task']}: best_score={s['best_score']:.4f}  runnable={s['best_runnable']}  fig={s['figure']}")
        print(f"\n[GLOBAL] Avg speedup={avg_speedup:.4f}, Accuracy={accuracy:.4f}")

        # ---- save under the SAME batch folder ----
        _save_global_summary(batch_dir, summary, avg_speedup, accuracy)
    else:
        print("No tasks were run.")


if __name__ == "__main__":
    main()
