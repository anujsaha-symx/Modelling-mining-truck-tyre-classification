from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

def _save_figure(output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()

def plot_class_distribution(dataset_df: pd.DataFrame, output_dir: Path) -> None:
    counts = dataset_df["label"].value_counts().sort_index()
    plt.figure(figsize=(7, 5))
    ax = counts.plot(kind="bar", color=["#2b8cbe", "#de2d26"])
    ax.set_title("Class Distribution")
    ax.set_xlabel("Label")
    ax.set_ylabel("Image Count")
    ax.tick_params(axis="x", rotation=0)
    for patch in ax.patches:
        height = patch.get_height()
        ax.annotate(
            f"{int(height)}",
            (patch.get_x() + patch.get_width() / 2, height),
            ha="center",
            va="bottom",
        )
    _save_figure(output_dir / "class_distribution.png")
def plot_image_size_distribution(dataset_df: pd.DataFrame, output_dir: Path) -> None:
    plt.figure(figsize=(8, 6))
    colors = dataset_df["label"].map({"good": "#2b8cbe", "bad": "#de2d26"}).fillna("#636363")
    plt.scatter(dataset_df["width"], dataset_df["height"], c=colors, alpha=0.7, edgecolors="none")
    plt.title("Image Size Distribution")
    plt.xlabel("Width (px)")
    plt.ylabel("Height (px)")
    _save_figure(output_dir / "image_size_distribution.png")
def plot_sample_grid(dataset_df: pd.DataFrame, output_dir: Path, samples_per_class: int = 6, random_state: int = 42) -> None:
    sample_frames: list[pd.DataFrame] = []
    for label in sorted(dataset_df["label"].unique()):
        class_df = dataset_df[dataset_df["label"] == label]
        sample_frames.append(class_df.sample(n=min(samples_per_class, len(class_df)), random_state=random_state))
    sample_df = pd.concat(sample_frames, ignore_index=True)
    total_samples = len(sample_df)
    columns = min(4, total_samples)
    rows = (total_samples + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(4 * columns, 4 * rows))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]
    for axis, (_, row) in zip(axes, sample_df.iterrows()):
        with Image.open(row["processed_filepath"]) as image:
            axis.imshow(image.convert("RGB"))
        axis.set_title(f"{row['label']} | {row['source_dataset']}")
        axis.axis("off")
    for axis in axes[total_samples:]:
        axis.axis("off")
    fig.suptitle("Sample Image Grid", fontsize=14)
    _save_figure(output_dir / "sample_grid.png")
def plot_duplicate_analysis(dataset_df: pd.DataFrame, summary: dict, output_dir: Path) -> None:
    duplicates = pd.Series(summary.get("duplicate_filenames", {}), dtype="int64").sort_values(ascending=False).head(10)
    duplicate_groups_found = summary.get("duplicate_groups_found")
    removed_duplicates = summary.get("removed_duplicates")
    remaining_duplicates = summary.get("remaining_duplicates")
    plt.figure(figsize=(10, 5))
    if duplicates.empty:
        plt.text(0.5, 0.5, "No duplicate filenames found", ha="center", va="center", fontsize=14)
        plt.axis("off")
    else:
        ax = duplicates.plot(kind="bar", color="#756bb1")
        ax.set_title("Top Duplicate Filenames")
        ax.set_xlabel("Filename")
        ax.set_ylabel("Occurrences")
        ax.tick_params(axis="x", rotation=45, labelsize=8)
    if duplicate_groups_found is not None:
        plt.gcf().text(
            0.02,
            0.02,
            (
                f"Duplicate-content groups: {duplicate_groups_found} | "
                f"Removed duplicates: {removed_duplicates} | Remaining duplicates: {remaining_duplicates}"
            ),
            fontsize=9,
        )
    _save_figure(output_dir / "duplicate_analysis.png")
def generate_eda_reports(dataset_df: pd.DataFrame, summary: dict, output_dir: Path) -> None:
    plot_class_distribution(dataset_df, output_dir)
    plot_image_size_distribution(dataset_df, output_dir)
    plot_sample_grid(dataset_df, output_dir)
    plot_duplicate_analysis(dataset_df, summary, output_dir)