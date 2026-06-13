"""Benchmark runner for contrastive, masked, and VICReg modules."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from ssl_repr.contrastive.simclr import SimCLRTrainConfig, fit_simclr
from ssl_repr.data.cluster_dgp import ClusterDGPConfig, generate_cluster_data
from ssl_repr.data.structured_dgp import StructuredDGPConfig, generate_structured_data
from ssl_repr.masked.trainer import MaskedAETrainConfig, fit_masked_ae
from ssl_repr.utils.seed import config_hash
from ssl_repr.vicreg.trainer import VICRegTrainConfig, fit_vicreg


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def _aggregate(results: list[dict]) -> dict[str, float]:
    if not results:
        return {}
    keys = results[0].keys()
    return {
        k: float(np.mean([r[k] for r in results]))
        for k in keys
        if isinstance(results[0][k], (int, float))
    }


def _aggregate_std(results: list[dict]) -> dict[str, float]:
    if not results:
        return {}
    keys = results[0].keys()
    return {
        k: float(np.std([r[k] for r in results]))
        for k in keys
        if isinstance(results[0][k], (int, float))
    }


def run_contrastive_benchmark(config: dict[str, Any]) -> dict[str, Any]:
    """Run SimCLR sweep on Gaussian cluster data."""
    seeds = config.get("seeds", [42])
    models = config.get("models", ["SimCLR"])
    n_samples_list = config.get("n_samples_list", [2000])
    all_results = []

    for n_samples in n_samples_list:
        for model_name in models:
            seed_results = []
            for seed in seeds:
                data = generate_cluster_data(
                    ClusterDGPConfig(
                        n_samples=n_samples,
                        n_features=config.get("n_features", 32),
                        n_clusters=config.get("n_clusters", 5),
                        cluster_separation=config.get("cluster_separation", 3.0),
                        nuisance_dim=config.get("nuisance_dim", 8),
                        train_ratio=config.get("train_ratio", 0.7),
                        val_ratio=config.get("val_ratio", 0.15),
                        seed=seed,
                    )
                )
                if model_name == "SimCLR":
                    result = fit_simclr(
                        data,
                        SimCLRTrainConfig(
                            epochs=config.get("epochs", 80),
                            batch_size=config.get("batch_size", 256),
                            lr=config.get("lr", 0.001),
                            hidden_dim=config.get("hidden_dim", 128),
                            repr_dim=config.get("repr_dim", 64),
                            projection_dim=config.get("projection_dim", 64),
                            n_layers=config.get("n_layers", 2),
                            dropout=config.get("dropout", 0.1),
                            temperature=config.get("temperature", 0.5),
                            seed=seed,
                        ),
                    )
                else:
                    raise ValueError(f"Unknown contrastive model: {model_name}")

                seed_results.append(
                    {
                        "linear_probe_acc": result.metrics.linear_probe_acc,
                        "nmi": result.metrics.nmi,
                        "silhouette": result.metrics.silhouette,
                        "embedding_std": result.metrics.embedding_std,
                        "train_loss": result.train_loss,
                    }
                )
            mean = _aggregate(seed_results)
            std = _aggregate_std(seed_results)
            all_results.append(
                {
                    "model": model_name,
                    "n_samples": n_samples,
                    **{f"{k}_mean": v for k, v in mean.items()},
                    **{f"{k}_std": v for k, v in std.items()},
                }
            )
    return {"module": "contrastive", "results": all_results}


def run_masked_benchmark(config: dict[str, Any]) -> dict[str, Any]:
    """Run masked autoencoder sweep on structured latent-factor data."""
    seeds = config.get("seeds", [42])
    models = config.get("models", ["MaskedAE"])
    n_samples_list = config.get("n_samples_list", [3000])
    all_results = []

    for n_samples in n_samples_list:
        for model_name in models:
            seed_results = []
            for seed in seeds:
                data = generate_structured_data(
                    StructuredDGPConfig(
                        n_samples=n_samples,
                        observed_dim=config.get("observed_dim", 48),
                        latent_dim=config.get("latent_dim", 8),
                        noise_std=config.get("noise_std", 0.15),
                        train_ratio=config.get("train_ratio", 0.7),
                        val_ratio=config.get("val_ratio", 0.15),
                        seed=seed,
                    )
                )
                if model_name == "MaskedAE":
                    result = fit_masked_ae(
                        data,
                        MaskedAETrainConfig(
                            epochs=config.get("epochs", 80),
                            batch_size=config.get("batch_size", 256),
                            lr=config.get("lr", 0.001),
                            hidden_dim=config.get("hidden_dim", 128),
                            latent_dim=config.get("latent_dim", 32),
                            n_layers=config.get("n_layers", 2),
                            dropout=config.get("dropout", 0.1),
                            mask_ratio=config.get("mask_ratio", 0.4),
                            seed=seed,
                        ),
                    )
                else:
                    raise ValueError(f"Unknown masked model: {model_name}")

                seed_results.append(
                    {
                        "recon_mse": result.metrics.recon_mse,
                        "latent_r2": result.metrics.latent_r2,
                        "linear_probe_acc": result.metrics.linear_probe_acc,
                        "train_loss": result.train_loss,
                    }
                )
            mean = _aggregate(seed_results)
            std = _aggregate_std(seed_results)
            all_results.append(
                {
                    "model": model_name,
                    "n_samples": n_samples,
                    **{f"{k}_mean": v for k, v in mean.items()},
                    **{f"{k}_std": v for k, v in std.items()},
                }
            )
    return {"module": "masked", "results": all_results}


def run_vicreg_benchmark(config: dict[str, Any]) -> dict[str, Any]:
    """Run VICReg sweep on Gaussian cluster data."""
    seeds = config.get("seeds", [42])
    models = config.get("models", ["VICReg"])
    n_samples_list = config.get("n_samples_list", [2000])
    all_results = []

    for n_samples in n_samples_list:
        for model_name in models:
            seed_results = []
            for seed in seeds:
                data = generate_cluster_data(
                    ClusterDGPConfig(
                        n_samples=n_samples,
                        n_features=config.get("n_features", 32),
                        n_clusters=config.get("n_clusters", 5),
                        cluster_separation=config.get("cluster_separation", 3.0),
                        nuisance_dim=config.get("nuisance_dim", 8),
                        train_ratio=config.get("train_ratio", 0.7),
                        val_ratio=config.get("val_ratio", 0.15),
                        seed=seed,
                    )
                )
                if model_name == "VICReg":
                    result = fit_vicreg(
                        data,
                        VICRegTrainConfig(
                            epochs=config.get("epochs", 80),
                            batch_size=config.get("batch_size", 256),
                            lr=config.get("lr", 0.001),
                            hidden_dim=config.get("hidden_dim", 128),
                            repr_dim=config.get("repr_dim", 64),
                            projection_dim=config.get("projection_dim", 64),
                            n_layers=config.get("n_layers", 2),
                            dropout=config.get("dropout", 0.1),
                            sim_coeff=config.get("sim_coeff", 25.0),
                            var_coeff=config.get("var_coeff", 25.0),
                            cov_coeff=config.get("cov_coeff", 1.0),
                            seed=seed,
                        ),
                    )
                else:
                    raise ValueError(f"Unknown VICReg model: {model_name}")

                seed_results.append(
                    {
                        "linear_probe_acc": result.metrics.linear_probe_acc,
                        "nmi": result.metrics.nmi,
                        "silhouette": result.metrics.silhouette,
                        "repr_std": result.repr_std,
                        "train_loss": result.train_loss,
                    }
                )
            mean = _aggregate(seed_results)
            std = _aggregate_std(seed_results)
            all_results.append(
                {
                    "model": model_name,
                    "n_samples": n_samples,
                    **{f"{k}_mean": v for k, v in mean.items()},
                    **{f"{k}_std": v for k, v in std.items()},
                }
            )
    return {"module": "vicreg", "results": all_results}


def run_benchmark(
    config_path: str | Path,
    module: str = "all",
    output_dir: str | Path | None = None,
) -> Path:
    """Run benchmark(s) and write results."""
    config = load_config(config_path)
    default_path = Path(config_path).parent / "default.yaml"
    merged = {**load_config(default_path), **config} if default_path.exists() else config

    results: dict[str, Any] = {
        "config_hash": config_hash(merged),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "modules": {},
    }

    if module in ("contrastive", "all"):
        contrastive_cfg = merged
        if module == "all":
            contrastive_path = Path(config_path).parent / "contrastive_benchmark.yaml"
            if contrastive_path.exists():
                contrastive_cfg = {**merged, **load_config(contrastive_path)}
        results["modules"]["contrastive"] = run_contrastive_benchmark(contrastive_cfg)

    if module in ("masked", "all"):
        masked_cfg = merged
        if module == "all":
            masked_path = Path(config_path).parent / "masked_benchmark.yaml"
            if masked_path.exists():
                masked_cfg = {**merged, **load_config(masked_path)}
        results["modules"]["masked"] = run_masked_benchmark(masked_cfg)

    if module in ("vicreg", "all"):
        vicreg_cfg = merged
        if module == "all":
            vicreg_path = Path(config_path).parent / "vicreg_benchmark.yaml"
            if vicreg_path.exists():
                vicreg_cfg = {**merged, **load_config(vicreg_path)}
        results["modules"]["vicreg"] = run_vicreg_benchmark(vicreg_cfg)

    out = Path(output_dir or "results")
    run_dir = out / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    from ssl_repr.evaluation.report import write_report

    write_report(results, run_dir / "summary.md")
    return run_dir
