"""Tables and figures from the saved results (validation always; test only if the one-time evaluation exists)."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from .config import ALWAYS_EMPTY_REFERENCE, ANALYSIS_CONFIG, condition_name, load_yaml  # noqa: E402
from .data import load_manifest, normalise, read_image, read_mask  # noqa: E402
from .select import load_trained_model  # noqa: E402

LOGGER = logging.getLogger(__name__)
SETTINGS = load_yaml(ANALYSIS_CONFIG)
BASELINE_CONDITION = SETTINGS["baseline_condition"]
PROPOSED_CONDITION = SETTINGS["proposed_condition"]
KEY_METRICS = [
    "pooled_foreground_dice",
    "pooled_foreground_iou",
    "positive_slice_dice",
    "positive_slice_iou",
    "empty_slices_with_false_positives",
    "empty_slice_false_positive_rate",
    "empty_slice_false_positive_pixels_total",
    "patient_dice_mean",
    "patient_dice_median",
    "patient_dice_min",
    "pixel_accuracy",
]


def _save_figure(figure: plt.Figure, path: Path) -> None:
    figure.tight_layout()
    figure.savefig(path, dpi=SETTINGS["figure_dpi"])
    plt.close(figure)


def _write_table(table: pd.DataFrame, path: Path, index: bool = False) -> None:
    table.to_csv(path.with_suffix(".csv"), index=index)
    path.with_suffix(".md").write_text(table.to_markdown(index=index, floatfmt=".4f") + "\n")


def summary_by_run(per_run: dict[str, dict]) -> pd.DataFrame:
    rows = [
        {"run_name": name, "condition": condition_name(name), **{m: summary[m] for m in KEY_METRICS}}
        for name, summary in per_run.items()
    ]
    return pd.DataFrame(rows)


def summary_by_condition(by_run: pd.DataFrame) -> pd.DataFrame:
    """Mean and sample standard deviation over seeds for every loss condition."""
    grouped = by_run.groupby("condition", sort=False)[KEY_METRICS]
    table = grouped.mean().add_suffix("_mean").join(grouped.std(ddof=1).add_suffix("_std"))
    table.insert(0, "n_seeds", grouped.size())
    ordered = ["n_seeds"] + [f"{m}_{s}" for m in KEY_METRICS for s in ("mean", "std")]
    return table[ordered].reset_index()


def paired_patient_dice(patient_dice_by_run: dict[str, pd.Series]) -> pd.DataFrame:
    """Per-patient Dice averaged over seeds for baseline and proposed conditions, and their difference."""

    def condition_mean(condition: str) -> pd.Series:
        series = [s for run, s in patient_dice_by_run.items() if condition_name(run) == condition]
        return pd.concat(series, axis=1).mean(axis=1)

    table = pd.DataFrame(
        {BASELINE_CONDITION: condition_mean(BASELINE_CONDITION), PROPOSED_CONDITION: condition_mean(PROPOSED_CONDITION)}
    )
    table["difference_proposed_minus_baseline"] = table[PROPOSED_CONDITION] - table[BASELINE_CONDITION]
    return table.sort_values("difference_proposed_minus_baseline")


def lesion_size_cutoffs(train_manifest: pd.DataFrame) -> tuple[float, float]:
    """Tertile cut-offs of lesion area (pixels) over TRAINING slices that contain a lesion."""
    areas = train_manifest.loc[train_manifest.has_foreground == 1, "foreground_pixels"]
    return float(np.quantile(areas, 1 / 3)), float(np.quantile(areas, 2 / 3))


def dice_by_lesion_size(slices_by_run: dict[str, pd.DataFrame], cutoffs: tuple[float, float]) -> pd.DataFrame:
    rows = []
    for run_name, slices in slices_by_run.items():
        positive = slices[slices.ground_truth_empty == 0]
        groups = pd.cut(
            positive.ground_truth_pixels,
            [-np.inf, cutoffs[0], cutoffs[1], np.inf],
            labels=SETTINGS["lesion_size_group_names"],
        )
        for group, members in positive.groupby(groups, observed=False):
            rows.append(
                {
                    "run_name": run_name,
                    "condition": condition_name(run_name),
                    "lesion_size_group": group,
                    "n_slices": len(members),
                    "mean_slice_dice": members.dice.mean(),
                }
            )
    return pd.DataFrame(rows)


def plot_training_curves(results_dir: Path, run_names: list[str], output: Path) -> None:
    figure, (loss_axis, dice_axis) = plt.subplots(1, 2, figsize=(12, 4.5))
    for condition in dict.fromkeys(condition_name(r) for r in run_names):
        histories = [
            pd.read_csv(results_dir / "runs" / r / "metrics.csv") for r in run_names if condition_name(r) == condition
        ]
        losses = np.vstack([h.train_loss for h in histories])
        dices = np.vstack([h.validation_pooled_foreground_dice_at_best_threshold for h in histories])
        epochs = histories[0].epoch + 1
        for axis, values in ((loss_axis, losses), (dice_axis, dices)):
            (line,) = axis.plot(epochs, values.mean(0), label=f"{condition} (n={len(histories)})")
            axis.fill_between(epochs, values.min(0), values.max(0), color=line.get_color(), alpha=0.15)
    loss_axis.set(title="Training loss (mean, min-max over seeds)", xlabel="epoch", yscale="log")
    dice_axis.set(title="Validation pooled Dice at best threshold", xlabel="epoch")
    dice_axis.legend(fontsize=8)
    _save_figure(figure, output)


def plot_threshold_curves(results_dir: Path, run_names: list[str], output: Path) -> None:
    figure, axis = plt.subplots(figsize=(7, 4.5))
    for run_name in run_names:
        sweep = pd.read_csv(results_dir / "validation" / run_name / "threshold_sweep.csv")
        axis.plot(sweep.threshold, sweep.pooled_foreground_dice, label=run_name, linewidth=1)
    axis.set(title="Validation pooled Dice by threshold", xlabel="threshold", ylabel="pooled foreground Dice")
    axis.legend(fontsize=6, ncol=2)
    _save_figure(figure, output)


def plot_trade_off(by_run: pd.DataFrame, split_label: str, output: Path) -> None:
    figure, axis = plt.subplots(figsize=(7, 4.5))
    for condition, rows in by_run.groupby("condition", sort=False):
        axis.scatter(rows.empty_slices_with_false_positives, rows.positive_slice_dice, label=condition)
    axis.set(
        title=f"{split_label}: positive-slice Dice vs empty-slice false positives",
        xlabel="empty slices with any false-positive pixel",
        ylabel="mean positive-slice Dice",
    )
    axis.legend(fontsize=8)
    _save_figure(figure, output)


def plot_paired_patients(paired: pd.DataFrame, split_label: str, output: Path) -> None:
    figure, axis = plt.subplots(figsize=(7, 4.5))
    for _, row in paired.iterrows():
        axis.plot([0, 1], [row[BASELINE_CONDITION], row[PROPOSED_CONDITION]], color="grey", alpha=0.6, marker="o")
    axis.set_xticks([0, 1], [BASELINE_CONDITION, PROPOSED_CONDITION])
    axis.set(title=f"{split_label}: per-patient Dice (mean over seeds)", ylabel="patient Dice")
    _save_figure(figure, output)


def choose_failure_slices(results_dir: Path) -> list[int]:
    """Worst positive slices of the proposed run and the empty slices with most false-positive pixels (baseline)."""
    baseline, proposed = _overlay_run_names()
    proposed_slices = pd.read_csv(results_dir / "validation" / proposed / "validation_slices.csv")
    baseline_slices = pd.read_csv(results_dir / "validation" / baseline / "validation_slices.csv")
    worst_positive = (
        proposed_slices[proposed_slices.ground_truth_empty == 0]
        .sort_values(["dice", "ground_truth_pixels"], ascending=[True, False])
        .index[: SETTINGS["overlay_worst_positive_slices"]]
    )
    worst_empty = (
        baseline_slices[baseline_slices.ground_truth_empty == 1]
        .sort_values("fp", ascending=False)
        .index[: SETTINGS["overlay_worst_empty_slices"]]
    )
    return list(worst_positive) + list(worst_empty)


def choose_representative_slices(results_dir: Path) -> list[int]:
    """Positive slices whose proposed-run Dice is closest to chosen quantiles (typical, not cherry-picked)."""
    _, proposed = _overlay_run_names()
    slices = pd.read_csv(results_dir / "validation" / proposed / "validation_slices.csv")
    positive = slices[slices.ground_truth_empty == 0]
    chosen = []
    for quantile in SETTINGS["overlay_representative_dice_quantiles"]:
        target = positive.dice.quantile(quantile)
        candidates = positive.drop(index=chosen, errors="ignore")
        chosen.append(int((candidates.dice - target).abs().idxmin()))
    return chosen


def _overlay_run_names() -> tuple[str, str]:
    seed = SETTINGS["overlay_seed"]
    return f"{BASELINE_CONDITION}_seed{seed}", f"{PROPOSED_CONDITION}_seed{seed}"


def plot_overlays(results_dir: Path, data_root: str, slice_indices: list[int], output: Path) -> None:
    """FLAIR slice with ground truth, baseline prediction and proposed prediction (same seed) for each slice."""
    manifest = load_manifest(data_root, "val")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    predictions = {}
    for run_name in _overlay_run_names():
        run_dir = results_dir / "runs" / run_name
        statistics = json.load(open(run_dir / "channel_statistics.json"))
        threshold = json.load(open(results_dir / "validation" / run_name / "selection.json"))["selected_threshold"]
        base_channels = yaml.safe_load(open(run_dir / "config.yaml"))["base_channels"]
        model = load_trained_model(run_dir, base_channels, device)
        images = np.stack(
            [
                normalise(read_image(os.path.join(data_root, manifest.image_path[i])), statistics).transpose(2, 0, 1)
                for i in slice_indices
            ]
        )
        with torch.no_grad():
            probabilities = torch.sigmoid(model(torch.from_numpy(images).to(device))).cpu().numpy()[:, 0]
        predictions[run_name] = probabilities >= threshold
    figure, axes = plt.subplots(len(slice_indices), 3, figsize=(9, 3 * len(slice_indices)), squeeze=False)
    for row, index in enumerate(slice_indices):
        background = read_image(os.path.join(data_root, manifest.image_path[index]))[
            ..., SETTINGS["overlay_background_channel"]
        ]
        truth = read_mask(os.path.join(data_root, manifest.mask_path[index]))
        panels = [("ground truth", truth)] + [(name, predictions[name][row]) for name in _overlay_run_names()]
        for column, (title, mask) in enumerate(panels):
            axis = axes[row, column]
            axis.imshow(background, cmap="gray")
            axis.imshow(np.ma.masked_equal(mask, 0), cmap="autumn", alpha=0.5)
            axis.set_title(f"{title}\n{manifest.patient_id[index]} {manifest.slice_id[index]}", fontsize=7)
            axis.axis("off")
    _save_figure(figure, output)


def experiment_record(results_dir: Path, run_names: list[str]) -> pd.DataFrame:
    """Configuration, seed, hardware, parameter count, training time and selection for every run."""
    rows = []
    for run_name in run_names:
        run_dir = results_dir / "runs" / run_name
        config = yaml.safe_load(open(run_dir / "config.yaml"))
        summary = json.load(open(run_dir / "run_summary.json"))
        selection = json.load(open(results_dir / "validation" / run_name / "selection.json"))
        rows.append(
            {
                "run_name": run_name,
                "study_group": config["study_group"],
                "loss": config["loss"],
                "dice_weight": config["dice_weight"],
                "positive_class_weight_used": summary["positive_class_weight_used"],
                "seed": config["seed"],
                "epochs": config["epochs"],
                "batch_size": config["batch_size"],
                "learning_rate": config["learning_rate"],
                "weight_decay": config["weight_decay"],
                "trainable_parameters": summary["trainable_parameters"],
                "gpu_name": summary["gpu_name"],
                "torch": summary["environment"]["torch"],
                "torch_cuda": summary["environment"]["torch_cuda"],
                "training_seconds": summary["training_seconds"],
                "best_epoch": selection["best_epoch"],
                "selected_threshold": selection["selected_threshold"],
                "selected_threshold_is_grid_edge": selection["selected_threshold_is_grid_edge"],
                "is_smoke_test": summary["is_smoke_test"],
            }
        )
    return pd.DataFrame(rows)


def plot_lesion_area_distribution(data_root: str, output: Path) -> None:
    """Lesion area histogram for training and validation slices that contain a lesion (data understanding)."""
    figure, axis = plt.subplots(figsize=(7, 4.5))
    for split in ("train", "val"):
        manifest = load_manifest(data_root, split)
        areas = manifest.loc[manifest.has_foreground == 1, "foreground_pixels"]
        empty = int((manifest.has_foreground == 0).sum())
        axis.hist(
            areas,
            bins=SETTINGS["lesion_area_histogram_bins"],
            alpha=0.5,
            label=f"{split}: {len(areas)} lesion slices, {empty} empty slices",
        )
    axis.set(title="Lesion area per slice (pixels)", xlabel="lesion pixels", ylabel="slices", xscale="linear")
    axis.legend(fontsize=8)
    _save_figure(figure, output)


def data_audit_tables(results_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split/label summary and per-channel intensity summary from results/data_preparation/data_audit.json."""
    audit = json.load(open(results_dir / "data_preparation" / "data_audit.json"))
    splits = pd.DataFrame(audit["splits"]).T
    labels = pd.DataFrame(
        {
            split: {k: v for k, v in stats.items() if not isinstance(v, (dict, list))}
            for split, stats in audit["label_statistics"].items()
        }
    ).T
    split_table = splits.join(labels).rename_axis("split").reset_index()
    intensity_rows = [
        {"split": split, "channel": channel, **values}
        for split, channels in audit["channel_intensity_statistics"].items()
        for channel, values in channels.items()
    ]
    return split_table, pd.DataFrame(intensity_rows)


def _load_validation(results_dir: Path, run_names: list[str]):
    summaries = {
        ALWAYS_EMPTY_REFERENCE: json.load(open(results_dir / "validation" / ALWAYS_EMPTY_REFERENCE / "summary.json"))
    }
    slices, patients = {}, {}
    for run_name in run_names:
        folder = results_dir / "validation" / run_name
        summaries[run_name] = json.load(open(folder / "selection.json"))["validation_summary_at_selected_threshold"]
        slices[run_name] = pd.read_csv(folder / "validation_slices.csv")
        patients[run_name] = pd.read_csv(folder / "validation_patient_dice.csv", index_col=0).iloc[:, 0]
    return summaries, slices, patients


def make_all_figures_and_tables(results_dir: str | Path, data_root: str, run_names: list[str]) -> None:
    results_dir = Path(results_dir)
    figures, tables = results_dir / "figures", results_dir / "tables"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    main_comparison_available = all(
        f"{c}_seed{SETTINGS['overlay_seed']}" in run_names for c in (BASELINE_CONDITION, PROPOSED_CONDITION)
    )

    plot_lesion_area_distribution(data_root, figures / "data_lesion_area_distribution.png")
    split_table, intensity_table = data_audit_tables(results_dir)
    _write_table(split_table, tables / "data_audit_splits_and_labels")
    _write_table(intensity_table, tables / "data_audit_channel_intensities")
    _write_table(experiment_record(results_dir, run_names), tables / "experiment_record")
    summaries, slices, patients = _load_validation(results_dir, run_names)
    validation_by_run = summary_by_run(summaries)
    _write_table(validation_by_run, tables / "validation_summary_by_run")
    _write_table(summary_by_condition(validation_by_run), tables / "validation_summary_by_condition")
    cutoffs = lesion_size_cutoffs(load_manifest(data_root, "train"))
    size_table = dice_by_lesion_size(slices, cutoffs)
    size_table["cutoffs_pixels_from_training"] = f"{cutoffs[0]:.2f} / {cutoffs[1]:.2f}"
    _write_table(size_table, tables / "validation_dice_by_lesion_size")
    plot_training_curves(results_dir, run_names, figures / "training_curves.png")
    plot_threshold_curves(results_dir, run_names, figures / "validation_threshold_curves.png")
    plot_trade_off(
        validation_by_run[validation_by_run.run_name != ALWAYS_EMPTY_REFERENCE],
        "Validation",
        figures / "validation_positive_dice_vs_empty_false_positives.png",
    )
    if main_comparison_available:
        paired = paired_patient_dice(patients)
        _write_table(paired, tables / "validation_paired_patient_dice", index=True)
        plot_paired_patients(paired, "Validation", figures / "validation_paired_patient_dice.png")
        plot_overlays(
            results_dir, data_root, choose_failure_slices(results_dir), figures / "validation_failure_overlays.png"
        )
        plot_overlays(
            results_dir,
            data_root,
            choose_representative_slices(results_dir),
            figures / "validation_representative_overlays.png",
        )

    test_results_path = results_dir / "test" / "test_results.json"
    if test_results_path.exists():
        test_by_run = summary_by_run(json.load(open(test_results_path)))
        _write_table(test_by_run, tables / "test_summary_by_run")
        _write_table(summary_by_condition(test_by_run), tables / "test_summary_by_condition")
        plot_trade_off(
            test_by_run[test_by_run.run_name != ALWAYS_EMPTY_REFERENCE],
            "Test",
            figures / "test_positive_dice_vs_empty_false_positives.png",
        )
        test_patients = pd.read_csv(results_dir / "test" / "test_patient_dice.csv", index_col=0)
        if main_comparison_available:
            paired = paired_patient_dice({c: test_patients[c] for c in test_patients.columns if c in run_names})
            _write_table(paired, tables / "test_paired_patient_dice", index=True)
            plot_paired_patients(paired, "Test", figures / "test_paired_patient_dice.png")
    else:
        LOGGER.info("No test results yet; only validation tables and figures were made.")
    LOGGER.info("Tables in %s, figures in %s", tables, figures)
