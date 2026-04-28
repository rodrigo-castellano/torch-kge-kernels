"""Profiling entrypoint for KGE training and evaluation."""
from __future__ import annotations

import argparse
import cProfile
import io
import os
import pstats
import time

from kge_kernels.training.config import TrainConfig
from kge_kernels.eval.checkpoint import evaluate_checkpoint
from kge_kernels.training.experiment import pipeline


def _emit_profile(
    profiler: cProfile.Profile,
    *,
    sort_by: str,
    top_k: int,
    output_path: str | None,
) -> None:
    buffer = io.StringIO()
    stats = pstats.Stats(profiler, stream=buffer).sort_stats(sort_by)
    stats.print_stats(top_k)
    report = buffer.getvalue()
    print(report, end="")
    if output_path:
        with open(output_path, "w", encoding="utf-8") as handle:
            handle.write(report)
        print(f"Saved profile report to {output_path}")


def _run_train(args: argparse.Namespace) -> None:
    os.makedirs(args.save_dir, exist_ok=True)
    cfg = TrainConfig(
        save_dir=args.save_dir,
        run_signature=args.run_signature,
        dataset=args.dataset,
        data_root=args.data_root,
        train_split=args.train_split,
        valid_split=args.valid_split,
        test_split=args.test_split,
        model=args.model,
        dim=args.dim,
        gamma=args.gamma,
        p=args.p,
        relation_dim=args.relation_dim,
        dropout=args.dropout,
        input_dropout=args.input_dropout,
        feature_map_dropout=args.feature_map_dropout,
        hidden_dropout=args.hidden_dropout,
        embedding_height=args.embedding_height,
        embedding_width=args.embedding_width,
        lr=args.lr,
        batch_size=args.batch_size,
        neg_ratio=args.neg_ratio,
        epochs=args.epochs,
        use_early_stopping=not args.no_early_stopping,
        patience=args.patience,
        num_workers=args.num_workers,
        amp=not args.no_amp,
        compile=not args.no_compile,
        compile_mode=args.compile_mode,
        compile_fullgraph=not args.no_compile_fullgraph,
        compile_warmup_steps=args.compile_warmup_steps,
        cpu=args.cpu,
        multi_gpu=False,
        seed=args.seed,
        eval_chunk_size=args.eval_chunk_size,
        eval_limit=args.eval_limit,
        valid_eval_every=args.valid_eval_every,
        valid_eval_queries=args.valid_eval_queries,
        report_train_mrr=args.report_train_mrr,
        use_reciprocal=args.use_reciprocal,
        adv_temp=args.adv_temp,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        warmup_ratio=args.warmup_ratio,
        scheduler=args.scheduler,
        eval_num_corruptions=args.eval_num_corruptions,
    )
    profiler = cProfile.Profile()
    start = time.perf_counter()
    profiler.enable()
    artifacts = pipeline(cfg)
    profiler.disable()
    elapsed = time.perf_counter() - start
    print(f"Train profiling finished in {elapsed:.2f}s")
    print(f"Config path: {artifacts.config_path}")
    print(f"Weights path: {artifacts.weights_path}")
    if artifacts.metrics:
        print(f"Metrics: {artifacts.metrics}")
    _emit_profile(profiler, sort_by=args.sort_by, top_k=args.top_k, output_path=args.profile_output)


def _run_eval(args: argparse.Namespace) -> None:
    profiler = cProfile.Profile()
    start = time.perf_counter()
    profiler.enable()
    metrics = evaluate_checkpoint(
        args.checkpoint_dir,
        weights_name=args.load_weights,
        split=args.eval_split,
        cpu=args.cpu,
        compile_model=not args.no_compile,
        compile_mode=args.compile_mode,
        compile_fullgraph=not args.no_compile_fullgraph,
        compile_warmup_steps=args.compile_warmup_steps,
        show_progress=args.show_progress,
        eval_limit=args.eval_limit,
    )
    profiler.disable()
    elapsed = time.perf_counter() - start
    print(f"Eval profiling finished in {elapsed:.2f}s")
    print(f"Metrics: {metrics}")
    _emit_profile(profiler, sort_by=args.sort_by, top_k=args.top_k, output_path=args.profile_output)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Profile KGE train/eval paths")
    parser.add_argument("--sort_by", default="cumulative", help="pstats sort key")
    parser.add_argument("--top_k", type=int, default=30, help="Number of profile rows to print")
    parser.add_argument("--profile_output", default=None, help="Optional file to write the profile report")

    subparsers = parser.add_subparsers(dest="mode", required=True)

    train_parser = subparsers.add_parser("train", help="Profile training")
    train_parser.add_argument("--save_dir", required=True)
    train_parser.add_argument("--run_signature", default="profile_train")
    train_parser.add_argument("--dataset", default="family")
    train_parser.add_argument(
        "--data_root",
        default=os.environ.get(
            "DATA_ROOT",
            os.path.expanduser("~/repos/data-swarm/main"),
        ),
    )
    train_parser.add_argument("--train_split", default="train.txt")
    train_parser.add_argument("--valid_split", default="valid.txt")
    train_parser.add_argument("--test_split", default="test.txt")
    train_parser.add_argument("--model", default="RotatE")
    train_parser.add_argument("--dim", type=int, default=1024)
    train_parser.add_argument("--gamma", type=float, default=12.0)
    train_parser.add_argument("--p", type=int, default=1)
    train_parser.add_argument("--relation_dim", type=int, default=None)
    train_parser.add_argument("--dropout", type=float, default=0.0)
    train_parser.add_argument("--input_dropout", type=float, default=0.2)
    train_parser.add_argument("--feature_map_dropout", type=float, default=0.2)
    train_parser.add_argument("--hidden_dropout", type=float, default=0.3)
    train_parser.add_argument("--embedding_height", type=int, default=10)
    train_parser.add_argument("--embedding_width", type=int, default=20)
    train_parser.add_argument("--lr", type=float, default=1e-3)
    train_parser.add_argument("--batch_size", type=int, default=512)
    train_parser.add_argument("--neg_ratio", type=int, default=1)
    train_parser.add_argument("--epochs", type=int, default=5)
    train_parser.add_argument("--no_early_stopping", action="store_true")
    train_parser.add_argument("--patience", type=int, default=4)
    train_parser.add_argument("--num_workers", type=int, default=2)
    train_parser.add_argument("--no_amp", action="store_true")
    train_parser.add_argument("--no_compile", action="store_true")
    train_parser.add_argument("--compile_mode", default="reduce-overhead")
    train_parser.add_argument("--no_compile_fullgraph", action="store_true")
    train_parser.add_argument("--compile_warmup_steps", type=int, default=0)
    train_parser.add_argument("--cpu", action="store_true")
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--eval_chunk_size", type=int, default=2048)
    train_parser.add_argument("--eval_limit", type=int, default=0)
    train_parser.add_argument("--valid_eval_every", type=int, default=0)
    train_parser.add_argument("--valid_eval_queries", type=int, default=0)
    train_parser.add_argument("--report_train_mrr", action="store_true")
    train_parser.add_argument("--use_reciprocal", action="store_true")
    train_parser.add_argument("--adv_temp", type=float, default=0.0)
    train_parser.add_argument("--weight_decay", type=float, default=0.0)
    train_parser.add_argument("--grad_clip", type=float, default=0.0)
    train_parser.add_argument("--warmup_ratio", type=float, default=0.0)
    train_parser.add_argument("--scheduler", default="none", choices=["none", "cosine"])
    train_parser.add_argument("--eval_num_corruptions", type=int, default=100)
    train_parser.set_defaults(handler=_run_train)

    eval_parser = subparsers.add_parser("eval", help="Profile checkpoint evaluation")
    eval_parser.add_argument("--checkpoint_dir", required=True)
    eval_parser.add_argument("--load_weights", default="best_weights.pth")
    eval_parser.add_argument("--eval_split", default="test", choices=["train", "valid", "test"])
    eval_parser.add_argument("--eval_limit", type=int, default=0)
    eval_parser.add_argument("--show_progress", action="store_true")
    eval_parser.add_argument("--cpu", action="store_true")
    eval_parser.add_argument("--no_compile", action="store_true")
    eval_parser.add_argument("--compile_mode", default="reduce-overhead")
    eval_parser.add_argument("--no_compile_fullgraph", action="store_true")
    eval_parser.add_argument("--compile_warmup_steps", type=int, default=0)
    eval_parser.set_defaults(handler=_run_eval)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
