"""KGE model factory and training-model builder.

``build_model``           — name → ``nn.Module`` factory
``build_training_model``  — factory + optional DataParallel/compile wrapping
"""
from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from .complex import ComplEx
from .conve import ConvE
from .distmult import DistMult
from .rotate import RotatE
from .transe import TransE
from .tucker import TuckER


def build_model(
    name: str,
    num_entities: int,
    num_relations: int,
    dim: int,
    gamma: float = 12.0,
    p_norm: int = 1,
    relation_dim: Optional[int] = None,
    dropout: float = 0.0,
    input_dropout: float = 0.2,
    feature_map_dropout: float = 0.2,
    hidden_dropout: float = 0.3,
    embedding_height: int = 10,
    embedding_width: int = 20,
) -> nn.Module:
    """Factory that returns the requested KGE model."""
    name = name.lower()
    if name == "rotate":
        return RotatE(num_entities, num_relations, dim=dim, gamma=gamma, p_norm=p_norm)
    if name == "complex":
        return ComplEx(num_entities, num_relations, dim=dim)
    if name == "tucker":
        return TuckER(
            num_entities,
            num_relations,
            entity_dim=dim,
            relation_dim=relation_dim or dim,
            dropout=dropout,
        )
    if name == "transe":
        return TransE(num_entities, num_relations, dim=dim, p_norm=p_norm)
    if name == "distmult":
        return DistMult(num_entities, num_relations, dim=dim)
    if name == "conve":
        return ConvE(
            num_entities,
            num_relations,
            dim=dim,
            input_dropout=input_dropout,
            feature_map_dropout=feature_map_dropout,
            hidden_dropout=hidden_dropout,
            embedding_height=embedding_height,
            embedding_width=embedding_width,
        )
    raise ValueError(f"Unknown model name: {name}")


def build_training_model(
    cfg,
    num_entities: int,
    num_relations: int,
    device: torch.device,
) -> nn.Module:
    """Build a KGE model and apply training wrappers (DataParallel, compile).

    ``cfg`` is duck-typed: reads ``model``, ``dim``, ``gamma``, ``p``,
    ``relation_dim``, ``dropout``, ``input_dropout``, ``feature_map_dropout``,
    ``hidden_dropout``, ``embedding_height``, ``embedding_width``,
    ``multi_gpu``, ``compile``, ``compile_mode``, ``compile_fullgraph``.
    """
    from ..training.loop import wrap_model_for_training

    model = build_model(
        cfg.model,
        num_entities,
        num_relations,
        dim=cfg.dim,
        gamma=cfg.gamma,
        p_norm=cfg.p,
        relation_dim=cfg.relation_dim,
        dropout=cfg.dropout,
        input_dropout=cfg.input_dropout,
        feature_map_dropout=cfg.feature_map_dropout,
        hidden_dropout=cfg.hidden_dropout,
        embedding_height=cfg.embedding_height,
        embedding_width=cfg.embedding_width,
    )
    if cfg.multi_gpu and device.type == "cuda" and torch.cuda.device_count() <= 1:
        print("Warning: multi_gpu=True but multiple GPUs not available. Using single GPU/CPU.")

    return wrap_model_for_training(model, device, cfg)


__all__ = ["build_model", "build_training_model"]
