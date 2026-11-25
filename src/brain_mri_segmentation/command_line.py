"""Command-line options shared by the numbered scripts."""

from __future__ import annotations

import argparse

from .config import load_paths


def add_path_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--paths-config", help="YAML with data_root and results_dir (default: configs/paths.yaml)")
    parser.add_argument("--data-root", help="override data_root from the paths config")
    parser.add_argument("--results-dir", help="override results_dir from the paths config")


def resolve_paths(arguments: argparse.Namespace) -> dict:
    return load_paths(arguments.paths_config, arguments.data_root, arguments.results_dir)


def default_device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"
