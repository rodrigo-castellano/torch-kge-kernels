"""Dataset path resolution for KGE experiments.

Conventions:
  - Training data lives under ``<data_root>/<dataset>/<split_filename>``
  - Splits default to ``train.txt``, ``valid.txt``, ``test.txt``
  - Explicit paths override the convention
"""
from __future__ import annotations

import os
from typing import Optional


def load_dataset_split(
    data_root: str,
    dataset_name: str,
    split_filename: str,
) -> str:
    """Resolve ``<data_root>/<dataset_name>/<split_filename>``."""
    path = os.path.join(data_root, dataset_name, split_filename)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Could not find split '{split_filename}' for dataset "
            f"'{dataset_name}' at {path}"
        )
    return path


def resolve_train_path(
    train_path: str | None,
    dataset: str | None,
    data_root: str,
    train_split: str,
) -> str:
    """Pick an explicit train path or resolve via dataset/split convention."""
    if train_path:
        return train_path
    if dataset:
        return load_dataset_split(data_root, dataset, train_split)
    raise ValueError("Provide either train_path or dataset")


def resolve_split_path(
    *,
    split_name: str,
    explicit_path: str | None,
    dataset: str | None,
    data_root: str,
    split_filename: str | None,
) -> Optional[str]:
    """Resolve an optional eval split; returns ``None`` if not available."""
    if explicit_path:
        if not os.path.isfile(explicit_path):
            raise FileNotFoundError(
                f"Provided {split_name} path '{explicit_path}' does not exist"
            )
        return explicit_path
    if dataset and split_filename:
        try:
            return load_dataset_split(data_root, dataset, split_filename)
        except FileNotFoundError as err:
            print(
                f"Warning: {split_name} split not found ({err}); continuing without it."
            )
    return None


__all__ = [
    "load_dataset_split",
    "resolve_split_path",
    "resolve_train_path",
]
