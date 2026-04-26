"""PyTorch KGE training CLI.

Standalone trainer for the model architectures shipped in ``kge_kernels.models``.
Pick a single model with ``--model`` or train a list with ``--models foo,bar``.
Per-model defaults live in ``MODEL_CONFIGS``; CLI flags override them.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    "RotatE": {
        "lr": 1e-3, "embedding_dim": 1024, "gamma": 12.0, "p": 1,
        "weight_decay": 0.0, "use_reciprocal": False, "adv_temp": 0.0,
        "grad_clip": 0.0, "warmup_ratio": 0.0, "scheduler": "none",
        "description": "Distance-based rotation model",
    },
    "ComplEx": {
        "lr": 5e-4, "embedding_dim": 1024, "weight_decay": 1e-6,
        "use_reciprocal": False, "adv_temp": 0.0, "grad_clip": 0.0,
        "warmup_ratio": 0.0, "scheduler": "none",
        "description": "Complex bilinear model",
    },
    "DistMult": {
        "lr": 5e-4, "embedding_dim": 512, "weight_decay": 1e-6,
        "use_reciprocal": False, "adv_temp": 0.0, "grad_clip": 0.0,
        "warmup_ratio": 0.0, "scheduler": "none",
        "description": "Bilinear diagonal model",
    },
    "TuckER": {
        "lr": 5e-4, "embedding_dim": 512, "relation_dim": 256, "dropout": 0.3,
        "weight_decay": 1e-6, "use_reciprocal": False, "adv_temp": 0.0,
        "grad_clip": 1.0, "warmup_ratio": 0.0, "scheduler": "none",
        "description": "Tucker decomposition model",
    },
    "TransE": {
        "lr": 1e-3, "embedding_dim": 512, "p": 1, "weight_decay": 0.0,
        "use_reciprocal": False, "adv_temp": 0.0, "grad_clip": 0.0,
        "warmup_ratio": 0.0, "scheduler": "none",
        "description": "Translational embedding model",
    },
    "ConvE": {
        "lr": 1e-3, "embedding_dim": 200, "embedding_height": 10, "embedding_width": 20,
        "input_dropout": 0.2, "feature_map_dropout": 0.2, "hidden_dropout": 0.3,
        "weight_decay": 1e-6, "use_reciprocal": False, "adv_temp": 0.0,
        "grad_clip": 1.0, "warmup_ratio": 0.0, "scheduler": "none",
        "description": "Convolutional model",
    },
    "mrr_boost": {
        "lr": 1e-3, "embedding_dim": 1024, "gamma": 12.0, "p": 1,
        "weight_decay": 1e-6, "use_reciprocal": True, "adv_temp": 0.0,
        "grad_clip": 2.0, "warmup_ratio": 0.1, "scheduler": "cosine",
        "description": "RotatE with MRR boost (reciprocal + warmup + scheduler)",
    },
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train PyTorch KGE models")
    parser.add_argument("--dataset", type=str, default="countries_s3")
    parser.add_argument(
        "--data_root", type=str,
        default=os.environ.get(
            "DATA_ROOT",
            os.path.expanduser("~/repos/data-swarm/main"),
        ),
        help="Path to the shared data repo (defaults to $DATA_ROOT or "
             "~/repos/data-swarm/main/).",
    )
    parser.add_argument("--train_split", type=str, default="train.txt")
    parser.add_argument("--valid_split", type=str, default="valid.txt")
    parser.add_argument("--test_split", type=str, default="test.txt")
    parser.add_argument("--model", type=str, default="RotatE")
    parser.add_argument("--models", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=1500)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--embedding_dim", type=int, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--no_early_stopping", action="store_true")
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda:all", choices=["cpu", "cuda:1", "cuda:all"])
    parser.add_argument("--min_gpu_memory_gb", type=float, default=2.0)
    parser.add_argument("--save_dir", type=str, default=str(Path.cwd() / "checkpoints"))
    parser.add_argument("--save_models", action="store_true", default=True)
    parser.add_argument("--results", type=str, default=None)
    parser.add_argument("--verbose", action="store_true", default=True)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--p", type=int, default=None, choices=[1, 2])
    parser.add_argument("--relation_dim", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--neg_ratio", type=int, default=1)
    parser.add_argument("--use_reciprocal", action="store_true")
    parser.add_argument("--adv_temp", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.0)
    parser.add_argument("--scheduler", type=str, default="none", choices=["none", "cosine"])
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--compile", action="store_true", default=True)
    parser.add_argument("--no_compile", action="store_true")
    parser.add_argument("--compile_mode", type=str, default="reduce-overhead", choices=["default", "reduce-overhead", "max-autotune"])
    parser.add_argument("--compile_fullgraph", action="store_true", default=True)
    parser.add_argument("--no_compile_fullgraph", dest="compile_fullgraph", action="store_false")
    parser.add_argument("--compile_warmup_steps", type=int, default=0)
    parser.add_argument("--eval_num_corruptions", type=int, default=100)
    parser.add_argument("--valid_eval_every", type=int, default=0)
    parser.add_argument("--valid_eval_queries", type=int, default=0)
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--load_checkpoint_dir", type=str, default=None)
    parser.add_argument("--load_weights", type=str, default="best_weights.pth")
    parser.add_argument("--eval_split", type=str, default="test", choices=["train", "valid", "test"])
    return parser


def select_gpus(device_choice: str, min_free_gb: float) -> tuple[bool, bool]:
    """Returns (use_cpu, use_multi_gpu)."""
    if device_choice == "cpu":
        print("\n=== Using CPU ===\n")
        return True, False
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True,
        )
        available: List[int] = []
        for line in result.stdout.strip().splitlines():
            gpu_id_s, free_mb_s = [part.strip() for part in line.split(",")]
            free_gb = float(free_mb_s) / 1024.0
            gpu_id = int(gpu_id_s)
            print(f"GPU {gpu_id}: {free_gb:.2f} GB free")
            if free_gb >= min_free_gb:
                available.append(gpu_id)
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        print(f"Warning: Could not query GPUs: {exc}")
        available = []
    if not available:
        print(f"No GPUs with at least {min_free_gb} GB free memory found.\nFalling back to CPU\n")
        return True, False
    if device_choice == "cuda:1" or len(available) == 1:
        gpu_id = available[0]
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        print(f"Set CUDA_VISIBLE_DEVICES={gpu_id}\n")
        return False, False
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, available))
    print(f"Set CUDA_VISIBLE_DEVICES={','.join(map(str, available))}\n")
    return False, len(available) > 1


def _resolve(cli_value: Any, base_config: Dict[str, Any], key: str, default: Any) -> Any:
    """Resolve a config value: CLI override > model config > default."""
    if cli_value is not None:
        return cli_value
    return base_config.get(key, default)


def build_train_config(
    model_name: str,
    args: argparse.Namespace,
    use_cpu: bool,
    use_multi_gpu: bool,
) -> "TrainConfig":
    """Build a TrainConfig from CLI args merged with per-model defaults."""
    from kge_kernels.training.config import TrainConfig

    base = MODEL_CONFIGS.get(model_name, {})
    dim = _resolve(args.embedding_dim, base, "embedding_dim", 1024)
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_signature = f"torch_{args.dataset}_{model_name}_{dim}_{timestamp}_s{args.seed}"
    save_dir = os.path.join(args.save_dir, run_signature)

    use_amp = args.amp and not args.no_amp
    use_compile = args.compile and not args.no_compile

    return TrainConfig(
        save_dir=save_dir,
        run_signature=run_signature,
        dataset=args.dataset,
        data_root=args.data_root,
        train_split=args.train_split,
        valid_split=args.valid_split,
        test_split=args.test_split,
        model=model_name if model_name != "mrr_boost" else "RotatE",
        dim=dim,
        gamma=_resolve(args.gamma, base, "gamma", 12.0),
        p=_resolve(args.p, base, "p", 1),
        relation_dim=_resolve(args.relation_dim, base, "relation_dim", None),
        dropout=_resolve(args.dropout, base, "dropout", 0.0),
        input_dropout=base.get("input_dropout", 0.2),
        feature_map_dropout=base.get("feature_map_dropout", 0.2),
        hidden_dropout=base.get("hidden_dropout", 0.3),
        embedding_height=base.get("embedding_height", 10),
        embedding_width=base.get("embedding_width", 20),
        lr=_resolve(args.lr, base, "lr", 1e-3),
        batch_size=args.batch_size,
        neg_ratio=args.neg_ratio,
        epochs=args.epochs,
        use_early_stopping=not args.no_early_stopping,
        patience=args.patience,
        num_workers=2,
        amp=use_amp,
        compile=use_compile,
        compile_mode=args.compile_mode,
        compile_fullgraph=args.compile_fullgraph,
        compile_warmup_steps=args.compile_warmup_steps,
        cpu=use_cpu,
        multi_gpu=use_multi_gpu,
        seed=args.seed,
        eval_chunk_size=2048,
        valid_eval_every=args.valid_eval_every,
        valid_eval_queries=args.valid_eval_queries,
        report_train_mrr=False,
        use_reciprocal=args.use_reciprocal or base.get("use_reciprocal", False),
        adv_temp=args.adv_temp,
        weight_decay=_resolve(args.weight_decay, base, "weight_decay", 0.0),
        grad_clip=args.grad_clip or base.get("grad_clip", 0.0),
        warmup_ratio=args.warmup_ratio or base.get("warmup_ratio", 0.0),
        scheduler=args.scheduler if args.scheduler != "none" else base.get("scheduler", "none"),
        eval_num_corruptions=args.eval_num_corruptions,
    )


def parse_models(args: argparse.Namespace) -> List[str]:
    if args.models:
        if args.models.lower() == "all":
            return list(MODEL_CONFIGS.keys())
        return [item.strip() for item in args.models.split(",") if item.strip()]
    return [args.model]


def run(argv: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    parser = build_parser()
    args = parser.parse_args(argv)
    use_cpu, use_multi_gpu = select_gpus(args.device, args.min_gpu_memory_gb)
    verbose = args.verbose and not args.quiet

    if args.eval_only or args.load_checkpoint_dir:
        from kge_kernels.eval.checkpoint import evaluate_checkpoint

        if not args.load_checkpoint_dir:
            raise ValueError("--eval_only requires --load_checkpoint_dir")
        use_compile = args.compile and not args.no_compile
        metrics = evaluate_checkpoint(
            args.load_checkpoint_dir,
            weights_name=args.load_weights,
            split=args.eval_split,
            cpu=use_cpu,
            compile_model=use_compile,
            compile_mode=args.compile_mode,
            compile_fullgraph=args.compile_fullgraph,
            compile_warmup_steps=args.compile_warmup_steps,
            show_progress=True,
        )
        result: Dict[str, Any] = {
            "model": "checkpoint_eval",
            "dataset": args.dataset,
            "split": args.eval_split,
            "checkpoint_dir": args.load_checkpoint_dir,
            "weights_name": args.load_weights,
            "metrics": metrics,
            "timestamp": dt.datetime.now().isoformat(),
        }
        if args.results:
            with open(args.results, "w", encoding="utf-8") as handle:
                json.dump([result], handle, indent=2)
        return [result]

    from kge_kernels.training.pipeline import train_model

    results: List[Dict[str, Any]] = []
    for model_name in parse_models(args):
        if verbose:
            print(f"\n{'=' * 50}\nTraining {model_name} on {args.dataset}\n{'=' * 50}")

        train_cfg = build_train_config(model_name, args, use_cpu, use_multi_gpu)
        if verbose:
            print("Training config:")
            print(train_cfg)
            print()

        start = time.time()
        artifacts = train_model(train_cfg)
        duration = time.time() - start

        result = {
            "model": model_name,
            "dataset": args.dataset,
            "metrics": artifacts.metrics or {},
            "model_path": artifacts.weights_path if args.save_models else None,
            "config_path": artifacts.config_path,
            "duration": duration,
            "timestamp": dt.datetime.now().isoformat(),
        }
        if verbose:
            print(f"Finished {model_name} in {duration:.2f}s")
            print("Metrics:", result["metrics"])
        results.append(result)

    if args.results:
        with open(args.results, "w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2)
    return results


def main() -> None:
    print("\nAvailable PyTorch models:")
    for model_name, model_cfg in MODEL_CONFIGS.items():
        print(f"  - {model_name}: {model_cfg['description']}")
    print()
    run()


if __name__ == "__main__":
    main()
