"""Unified KGE inference wrapper: load a checkpoint, score atoms or top-k."""
from __future__ import annotations

import json
import os
import random
import re
import subprocess
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor

from kge_kernels.models import build_model


_BACKEND_ALIASES = {
    "torch": "pytorch",
    "pytorch": "pytorch",
    "pykeen": "pykeen",
}
_BACKEND_PREFIX = {
    "pytorch": "torch",
    "pykeen": "pykeen",
}


def _get_available_gpus(
    memory_threshold: float = 0.1,
    utilization_threshold: float = 0.3,
) -> List[int]:
    """Get GPU indices that are not currently busy."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return list(range(torch.cuda.device_count()))

        available_gpus = []
        for line in result.stdout.strip().splitlines():
            if not line.strip():
                continue
            parts = [part.strip() for part in line.split(",")]
            if len(parts) < 4:
                continue
            gpu_id = int(parts[0])
            mem_used = float(parts[1])
            mem_total = float(parts[2])
            utilization = float(parts[3])
            mem_fraction = mem_used / mem_total if mem_total > 0 else 1.0
            util_fraction = utilization / 100.0
            if mem_fraction <= memory_threshold and util_fraction <= utilization_threshold:
                available_gpus.append(gpu_id)
        return available_gpus
    except Exception:
        return list(range(torch.cuda.device_count()))


class _Atom:
    """Atom parser for functional notation ``pred(arg1,arg2)``."""

    def __init__(self, atom: str):
        normalized = re.sub(r"\b([(),\.])", r"\1", atom.strip())
        if normalized.endswith("."):
            normalized = normalized[:-1]
        tokens = normalized.replace("(", " ").replace(")", " ").replace(",", " ").split()
        self.r = tokens[0]
        self.args = tokens[1:]

    def to_tuple(self) -> Tuple[str, ...]:
        return (self.r,) + tuple(self.args)


class _PyTorchKGEInference:
    """PyTorch KGE inference engine: load a trained checkpoint and score atoms."""

    def __init__(
        self,
        dataset_name: str,
        base_path: str,
        run_signature: str,
        checkpoint_dir: str = "./../checkpoints/",
        seed: int = 0,
        scores_file_path: Optional[str] = None,
        runtime_cache_max_entries: Optional[int] = None,
        persist_runtime_scores: bool = True,
        device: Optional[str] = None,
    ) -> None:
        del dataset_name
        del base_path

        self.seed = seed
        self.set_seeds(self.seed)
        self.run_signature = run_signature
        self.checkpoint_dir = checkpoint_dir

        self.use_multi_gpu = False
        self.device = None
        self.available_gpu_ids = None

        if device == "cuda:all":
            if torch.cuda.is_available() and torch.cuda.device_count() > 1:
                available_gpu_indices = _get_available_gpus()
                if len(available_gpu_indices) > 1:
                    self.device = torch.device(f"cuda:{available_gpu_indices[0]}")
                    self.available_gpu_ids = available_gpu_indices
                    self.use_multi_gpu = True
                    print(
                        "PyTorch DataParallel with "
                        f"{len(available_gpu_indices)} GPUs: {available_gpu_indices}"
                    )
                else:
                    self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            else:
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif device == "cpu":
            self.device = torch.device("cpu")
        elif device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        self.model_dir = self._resolve_model_dir(checkpoint_dir, run_signature, seed)
        self.model = None
        self.config = None
        self.entity2id = None
        self.relation2id = None

        self.atom_scores: Dict[str, float] = {}
        self.persist_runtime_scores = persist_runtime_scores
        self._tuple_cache: Dict[str, Tuple[str, ...]] = {}

        if runtime_cache_max_entries == 0:
            self._runtime_cache_enabled = False
            self._score_cache = None
        else:
            self._runtime_cache_enabled = True
            self._runtime_cache_max_entries = runtime_cache_max_entries
            self._score_cache = OrderedDict()

        if scores_file_path:
            self._load_scores(scores_file_path)

    def set_seeds(self, seed: int) -> None:
        import numpy as np

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _load_scores(self, filepath: str) -> None:
        """Load pre-computed scores from a TSV file."""
        if not os.path.exists(filepath) or os.path.isdir(filepath):
            print(f"Warning: Scores file not found at {filepath}. KGE will perform live inference.")
            return

        import pandas as pd

        print(f"Loading pre-computed scores from {filepath}...")
        start_time = time.time()
        try:
            df = pd.read_csv(
                filepath,
                sep="\t",
                header=None,
                names=["atom", "score"],
                dtype={"atom": str, "score": float},
                engine="c",
            )
            self.atom_scores = pd.Series(df.score.values, index=df.atom).to_dict()
            print(f"Loaded {len(self.atom_scores)} scores in {time.time() - start_time:.2f}s.")
        except Exception as exc:
            print(f"Error loading scores: {exc}")

    def _get_runtime_cached_score(self, atom_str: str) -> Optional[float]:
        if not self._runtime_cache_enabled or self._score_cache is None:
            return None
        if atom_str in self._score_cache:
            score = self._score_cache[atom_str]
            self._score_cache.move_to_end(atom_str)
            return score
        return None

    def _store_runtime_score(self, atom_str: str, score: float) -> None:
        if self._runtime_cache_enabled and self._score_cache is not None:
            self._score_cache[atom_str] = score
            self._score_cache.move_to_end(atom_str)
            if self._runtime_cache_max_entries and len(self._score_cache) > self._runtime_cache_max_entries:
                self._score_cache.popitem(last=False)
        if self.persist_runtime_scores:
            self.atom_scores[atom_str] = score

    def _resolve_model_dir(self, checkpoint_dir: str, run_signature: str, seed: int) -> str:
        """Resolve run directory with backward-compatible fallbacks."""
        signature_path = os.path.expanduser(run_signature)
        if os.path.isdir(signature_path):
            return signature_path
        root = Path(checkpoint_dir)
        candidate = root / run_signature
        if candidate.is_dir():
            return str(candidate)
        seeded = root / f"{run_signature}_seed_{seed}"
        if seeded.is_dir():
            return str(seeded)
        return str(Path(checkpoint_dir) / f"{run_signature}_seed_{seed}")

    def _adapt_state_dict(self, state: Dict[str, Tensor], model: torch.nn.Module) -> Dict[str, Tensor]:
        """Adapt legacy torch-ns checkpoint keys to the current tkk model layout."""
        target = model.state_dict()
        if set(state.keys()) == set(target.keys()):
            return state

        adapted: Dict[str, Tensor] = {}
        for key in target:
            if key in state:
                adapted[key] = state[key]
                continue

            if key == "entity_embeddings.weight":
                if "ent_re.weight" in state and "ent_im.weight" in state:
                    adapted[key] = torch.cat([state["ent_re.weight"], state["ent_im.weight"]], dim=-1)
                    continue
                if "ent.weight" in state:
                    adapted[key] = state["ent.weight"]
                    continue

            if key == "relation_embeddings.weight":
                if "rel_re.weight" in state and "rel_im.weight" in state:
                    adapted[key] = torch.cat([state["rel_re.weight"], state["rel_im.weight"]], dim=-1)
                    continue
                if "rel.weight" in state:
                    adapted[key] = state["rel.weight"]
                    continue

            if key == "ent.weight" and "entity_embeddings.weight" in state:
                adapted[key] = state["entity_embeddings.weight"]
                continue

            if key == "rel.weight" and "relation_embeddings.weight" in state:
                adapted[key] = state["relation_embeddings.weight"]
                continue

        missing = [key for key in target if key not in adapted]
        if missing:
            preview = ", ".join(missing[:8])
            raise RuntimeError(
                "Could not adapt checkpoint state dict to current model layout. "
                f"Missing keys: {preview}"
            )
        return adapted

    def _build_and_load_model(self) -> torch.nn.Module:
        """Build and load the PyTorch KGE model."""
        print("Building model and loading weights...")

        config_path = os.path.join(self.model_dir, "config.json")
        with open(config_path, "r", encoding="utf-8") as handle:
            self.config = json.load(handle)

        with open(os.path.join(self.model_dir, "entity2id.json"), "r", encoding="utf-8") as handle:
            self.entity2id = json.load(handle)
        with open(os.path.join(self.model_dir, "relation2id.json"), "r", encoding="utf-8") as handle:
            self.relation2id = json.load(handle)

        model = build_model(
            self.config.get("model", "RotatE"),
            self.config["num_entities"],
            self.config["num_relations"],
            dim=self.config.get("dim") or self.config.get("entity_dim"),
            gamma=self.config.get("gamma", 12.0),
            p_norm=self.config.get("p", 1),
            relation_dim=self.config.get("relation_dim"),
            dropout=self.config.get("dropout", 0.0),
        )

        weights_path = os.path.join(self.model_dir, "weights.pth")
        state = torch.load(weights_path, map_location="cpu")
        if any(key.startswith("_orig_mod.") for key in state.keys()):
            state = {key.replace("_orig_mod.", ""): value for key, value in state.items()}
        state = self._adapt_state_dict(state, model)
        model.load_state_dict(state, strict=True)
        model.to(self.device)
        model.eval()

        if self.use_multi_gpu and self.available_gpu_ids:
            model = torch.nn.DataParallel(model, device_ids=self.available_gpu_ids)

        print("Weights loaded successfully.")
        return model

    def _normalize_atom_input(self, atom: Union[str, Tuple[str, ...]]) -> str:
        """Convert an atom to canonical string form."""
        if isinstance(atom, str):
            return atom
        if isinstance(atom, tuple):
            predicate, *args = atom
            return f"{predicate}({','.join(map(str, args))})"
        raise TypeError(f"Unsupported atom type: {type(atom)}")

    def _atom_str_to_tuple(self, atom_str: str) -> Tuple[str, ...]:
        """Convert an atom string to tuple form."""
        cached = self._tuple_cache.get(atom_str)
        if cached is not None:
            return cached
        atom_tuple = _Atom(atom_str).to_tuple()
        self._tuple_cache[atom_str] = atom_tuple
        return atom_tuple

    def _prepare_triple_ids(self, atom_tuples: List[Tuple[str, ...]]) -> torch.Tensor:
        """Convert atom tuples to ID tensor."""
        ids = []
        for atom_tuple in atom_tuples:
            predicate, *args = atom_tuple
            if len(args) != 2:
                raise ValueError(f"Expected binary predicate, got {len(args)} arguments")
            head, tail = args
            if head not in self.entity2id or tail not in self.entity2id or predicate not in self.relation2id:
                raise ValueError(f"Unknown entity or relation in {atom_tuple}")
            ids.append([self.relation2id[predicate], self.entity2id[head], self.entity2id[tail]])
        return torch.tensor(ids, dtype=torch.long)

    def _score_atoms_via_model(self, atom_tuples: Sequence[Tuple[str, ...]]) -> List[float]:
        """Score atoms using the loaded KGE model."""
        if not atom_tuples:
            return []

        if self.model is None:
            self.model = self._build_and_load_model()

        try:
            triple_ids = self._prepare_triple_ids(list(atom_tuples))
        except ValueError:
            return [0.0] * len(atom_tuples)

        scores = []
        batch_size = 2048

        with torch.no_grad():
            for start in range(0, triple_ids.size(0), batch_size):
                batch = triple_ids[start : start + batch_size].to(self.device)
                batch_scores = self.model.score(batch[:, 1], batch[:, 0], batch[:, 2])
                batch_scores = torch.sigmoid(batch_scores)
                scores.append(batch_scores.float().cpu())

        if scores:
            return torch.cat(scores, dim=0).tolist()
        return []

    def get_topk_tails(
        self,
        head: str,
        relation: str,
        k: int,
        return_scores: bool = True,
    ) -> List[Tuple[str, float]]:
        """Get top-k tail entities for (head, relation)."""
        del return_scores

        if self.model is None:
            self.model = self._build_and_load_model()

        if head not in self.entity2id or relation not in self.relation2id:
            return []

        h_idx = torch.tensor(self.entity2id[head], device=self.device)
        r_idx = torch.tensor(self.relation2id[relation], device=self.device)

        with torch.no_grad():
            if self.device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    scores = self.model.score(h_idx, r_idx, None)
            else:
                scores = self.model.score(h_idx, r_idx, None)

            scores = torch.sigmoid(scores.float())
            scores[h_idx] = float("-inf")
            top_scores, top_indices = torch.topk(scores, min(k, scores.size(0)))

        id2entity = {value: key for key, value in self.entity2id.items()}
        return [(id2entity[index.item()], score.item()) for index, score in zip(top_indices, top_scores)]

    def get_topk_heads(
        self,
        relation: str,
        tail: str,
        k: int,
        return_scores: bool = True,
    ) -> List[Tuple[str, float]]:
        """Get top-k head entities for (relation, tail)."""
        del return_scores

        if self.model is None:
            self.model = self._build_and_load_model()

        if tail not in self.entity2id or relation not in self.relation2id:
            return []

        r_idx = torch.tensor(self.relation2id[relation], device=self.device)
        t_idx = torch.tensor(self.entity2id[tail], device=self.device)

        with torch.no_grad():
            if self.device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    scores = self.model.score(None, r_idx, t_idx)
            else:
                scores = self.model.score(None, r_idx, t_idx)

            scores = torch.sigmoid(scores.float())
            scores[t_idx] = float("-inf")
            top_scores, top_indices = torch.topk(scores, min(k, scores.size(0)))

        id2entity = {value: key for key, value in self.entity2id.items()}
        return [(id2entity[index.item()], score.item()) for index, score in zip(top_indices, top_scores)]

    def predict_batch(self, atoms_for_ranking: Sequence[Union[str, Tuple[str, ...]]]) -> List[float]:
        """Score a batch of atoms and return scores in [0, 1]."""
        if not atoms_for_ranking:
            return []

        canonical_atoms = [self._normalize_atom_input(atom) for atom in atoms_for_ranking]
        scores: List[Optional[float]] = [None] * len(canonical_atoms)
        missing = []
        missing_indices = []

        for index, atom_str in enumerate(canonical_atoms):
            cached = self.atom_scores.get(atom_str)
            if cached is None:
                cached = self._get_runtime_cached_score(atom_str)
            if cached is not None:
                scores[index] = float(cached)
            else:
                missing.append(atom_str)
                missing_indices.append(index)

        if missing:
            unique_missing = list(dict.fromkeys(missing))
            atom_tuples = [self._atom_str_to_tuple(atom) for atom in unique_missing]
            new_scores = self._score_atoms_via_model(atom_tuples)

            newly_scored = {atom: float(score) for atom, score in zip(unique_missing, new_scores)}
            for atom, score in newly_scored.items():
                self._store_runtime_score(atom, score)

            for index, atom_str in zip(missing_indices, missing):
                scores[index] = newly_scored[atom_str]

        return [float(score) for score in scores]


def normalize_backend(backend: Optional[str]) -> str:
    """Normalize backend name and validate supported options."""
    if backend is None:
        return "pytorch"
    normalized = backend.strip().lower()
    if normalized in _BACKEND_ALIASES:
        return _BACKEND_ALIASES[normalized]
    raise ValueError(
        f"Unsupported KGE backend '{backend}'. Use 'pytorch' or 'pykeen'."
    )


def find_latest_run(checkpoint_dir: str, prefix: Optional[str] = None) -> Optional[str]:
    """Find the most recent run directory under the checkpoint root."""
    root = Path(checkpoint_dir)
    if not root.is_dir():
        return None
    candidates = [p for p in root.iterdir() if p.is_dir()]
    if prefix:
        candidates = [p for p in candidates if p.name.startswith(prefix)]
    if not candidates:
        return None
    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    return latest.name


def _get_backend_class(backend: str) -> Tuple[type, str]:
    """Load the appropriate KGE backend class based on the specified backend."""
    backend = normalize_backend(backend)
    if backend == "pytorch":
        return _PyTorchKGEInference, "pytorch"
    if backend == "pykeen":
        from kge_pykeen.kge_inference_pykeen import KGEInference as BackendKGEInference
        return BackendKGEInference, "pykeen"
    raise ValueError(f"Unsupported KGE backend '{backend}'.")


class KGEInference:
    """Wrapper class that delegates to the appropriate backend implementation."""

    def __init__(
        self,
        dataset_name: str,
        base_path: str,
        checkpoint_dir: str,
        run_signature: str,
        seed: int = 0,
        scores_file_path: Optional[str] = None,
        backend: str = "pytorch",
        **kwargs: Any,
    ) -> None:
        BackendClass, self.backend = _get_backend_class(backend)
        self._backend_engine = BackendClass(
            dataset_name=dataset_name,
            base_path=base_path,
            checkpoint_dir=checkpoint_dir,
            run_signature=run_signature,
            seed=seed,
            scores_file_path=scores_file_path,
            **kwargs,
        )
        print(f"KGE Engine initialized with backend: {self.backend} (signature: {run_signature})")

    def __getattr__(self, name: str) -> Any:
        """Delegate all attribute access to the backend engine."""
        return getattr(self._backend_engine, name)

    def __repr__(self) -> str:
        return f"KGEInference(backend={self.backend}, engine={self._backend_engine})"


def current_backend(engine: Optional[KGEInference] = None) -> str:
    """Return the name of the active backend implementation."""
    if engine is not None:
        return engine.backend
    return "pytorch"
