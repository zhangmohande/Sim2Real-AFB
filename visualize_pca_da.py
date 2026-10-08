# -*- coding: utf-8 -*-
"""O4 PCA visualization for the proposed AFB alignment."""

import argparse
import csv
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data import SourceDataset, TargetDataset
from net import SFD_Mamba2Net


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def ensure_parent(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def region_feature_to_vector(feature, region, min_pixels=8):
    if feature.dim() != 4:
        raise RuntimeError(
            f"O4 PCA expects [B,C,H,W], got {tuple(feature.shape)}"
        )
    if region.shape[2:] != feature.shape[2:]:
        region = F.interpolate(
            region.float(), size=feature.shape[2:], mode="nearest"
        )
    else:
        region = region.float()

    vectors = []
    valid_indices = []
    for index in range(feature.shape[0]):
        current_region = region[index : index + 1]
        if current_region.sum().item() < min_pixels:
            continue
        vector = (feature[index : index + 1] * current_region).sum(
            dim=(2, 3)
        ) / (current_region.sum(dim=(2, 3)) + 1e-8)
        vectors.append(F.normalize(vector, p=2, dim=1))
        valid_indices.append(index)

    if not vectors:
        return None, []
    return torch.cat(vectors, dim=0), valid_indices


def pca_2d(features):
    features = features.astype(np.float64)
    mean = features.mean(axis=0, keepdims=True)
    standard_deviation = features.std(axis=0, keepdims=True)
    standard_deviation = np.where(
        standard_deviation < 1e-12, 1.0, standard_deviation
    )
    features = (features - mean) / standard_deviation
    features = features - features.mean(axis=0, keepdims=True)

    _, singular_values, right_vectors = np.linalg.svd(
        features, full_matrices=False
    )
    component_count = min(2, right_vectors.shape[0])
    coordinates = features @ right_vectors[:component_count].T
    if component_count < 2:
        coordinates = np.pad(
            coordinates,
            ((0, 0), (0, 2 - component_count)),
            mode="constant",
        )

    eigenvalues = singular_values ** 2
    eigenvalues = eigenvalues / max(features.shape[0] - 1, 1)
    explained = eigenvalues[:component_count] / (
        eigenvalues.sum() + 1e-12
    ) * 100.0
    if component_count < 2:
        explained = np.pad(
            explained, (0, 2 - component_count), mode="constant"
        )
    return coordinates, explained


def domain_metrics(coordinates, domains):
    domains = np.asarray(domains)
    source = coordinates[domains == "Source"]
    target = coordinates[domains == "Target"]
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    return {
        "source_count": int(source.shape[0]),
        "target_count": int(target.shape[0]),
        "centroid_distance": float(
            np.linalg.norm(source_center - target_center)
        ),
        "source_spread": float(
            np.mean(np.linalg.norm(source - source_center, axis=1))
        ),
        "target_spread": float(
            np.mean(np.linalg.norm(target - target_center, axis=1))
        ),
    }


@torch.no_grad()
def extract_o4_features(
    model,
    loader,
    domain_name,
    region_mode,
    max_samples=None,
    min_pixels=8,
):
    model.eval()
    all_features = []
    all_domains = []
    all_names = []
    skipped = 0

    for image, mask, names in loader:
        if max_samples is not None and len(all_domains) >= max_samples:
            break

        image = image.to(device)
        mask = mask.to(device)
        _, feature_dictionary = model(image, return_feature=True)
        o4 = feature_dictionary["O4"]

        if region_mode == "foreground":
            region = (mask > 0.5).float()
        elif region_mode == "background":
            region = (mask <= 0.5).float()
        else:
            raise ValueError(f"Unsupported region mode: {region_mode}")

        vectors, valid_indices = region_feature_to_vector(
            o4, region, min_pixels=min_pixels
        )
        if vectors is None:
            skipped += image.shape[0]
            continue

        vectors = vectors.cpu().numpy()
        valid_names = [names[index] for index in valid_indices]
        if max_samples is not None:
            remaining = max_samples - len(all_domains)
            vectors = vectors[:remaining]
            valid_names = valid_names[:remaining]

        all_features.append(vectors)
        all_domains.extend([domain_name] * vectors.shape[0])
        all_names.extend(valid_names)

    if not all_features:
        raise RuntimeError(
            f"No O4 features extracted from {domain_name}; skipped={skipped}"
        )
    return (
        np.concatenate(all_features, axis=0),
        all_domains,
        all_names,
        skipped,
    )


def load_model(model_path, stage_name):
    model = SFD_Mamba2Net().to(device)
    if str(model_path).lower() in {
        "untrained",
        "none",
        "random",
        "__untrained__",
    }:
        print(f"[PCA] Random model: {stage_name}")
        model.eval()
        return model

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def collect_result(
    model_path,
    stage_name,
    source_path,
    target_path,
    region_mode,
    batch_size,
    max_samples,
    min_pixels,
):
    model = load_model(model_path, stage_name)
    source_loader = DataLoader(
        SourceDataset(source_path, augment=False),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    target_loader = DataLoader(
        TargetDataset(target_path, with_label=True, augment=False),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    source_feature, source_domain, source_names, source_skipped = (
        extract_o4_features(
            model,
            source_loader,
            "Source",
            region_mode,
            max_samples,
            min_pixels,
        )
    )
    target_feature, target_domain, target_names, target_skipped = (
        extract_o4_features(
            model,
            target_loader,
            "Target",
            region_mode,
            max_samples,
            min_pixels,
        )
    )

    features = np.concatenate([source_feature, target_feature], axis=0)
    domains = source_domain + target_domain
    names = source_names + target_names
    coordinates, explained = pca_2d(features)
    metrics = domain_metrics(coordinates, domains)

    row = {
        "stage": stage_name,
        "region": region_mode,
        "layer": "O4",
        **metrics,
        "pc1_explained_percent": float(explained[0]),
        "pc2_explained_percent": float(explained[1]),
        "model_path": model_path,
        "source_skipped": int(source_skipped),
        "target_skipped": int(target_skipped),
    }
    return {
        "stage": stage_name,
        "region": region_mode,
        "coordinates": coordinates,
        "domains": domains,
        "names": names,
        "explained": explained,
        "metrics": row,
    }


def save_metrics(path, results):
    ensure_parent(path)
    fields = [
        "stage",
        "region",
        "layer",
        "source_count",
        "target_count",
        "centroid_distance",
        "source_spread",
        "target_spread",
        "pc1_explained_percent",
        "pc2_explained_percent",
        "model_path",
        "source_skipped",
        "target_skipped",
    ]
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for result in results:
            writer.writerow(result["metrics"])


def save_points(path, results):
    ensure_parent(path)
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["stage", "region", "layer", "domain", "name", "pc1", "pc2"]
        )
        for result in results:
            for index, domain in enumerate(result["domains"]):
                writer.writerow(
                    [
                        result["stage"],
                        result["region"],
                        "O4",
                        domain,
                        result["names"][index],
                        result["coordinates"][index, 0],
                        result["coordinates"][index, 1],
                    ]
                )


def plot_results(results, stage_names, output_prefix, region_mode):
    lookup = {
        (result["stage"], result["region"]): result
        for result in results
    }
    figure, axes = plt.subplots(
        len(stage_names),
        1,
        figsize=(5.0, 3.8 * len(stage_names)),
        squeeze=False,
    )

    for row, stage_name in enumerate(stage_names):
        axis = axes[row][0]
        result = lookup[(stage_name, region_mode)]
        coordinates = result["coordinates"]
        domains = np.asarray(result["domains"])
        source = domains == "Source"
        target = domains == "Target"
        axis.scatter(
            coordinates[source, 0],
            coordinates[source, 1],
            s=16,
            alpha=0.75,
            marker="o",
            label="Source",
        )
        axis.scatter(
            coordinates[target, 0],
            coordinates[target, 1],
            s=16,
            alpha=0.75,
            marker="^",
            label="Target",
        )
        metrics = result["metrics"]
        axis.set_title(
            f"{stage_name} | {region_mode} O4\n"
            f"Centroid distance={metrics['centroid_distance']:.3f}"
        )
        axis.set_xlabel(f"PC1 ({result['explained'][0]:.1f}%)")
        axis.set_ylabel(f"PC2 ({result['explained'][1]:.1f}%)")
        axis.grid(alpha=0.25)
        axis.legend()

    figure.suptitle(f"AFB O4 PCA: {region_mode}")
    figure.tight_layout(rect=[0, 0, 1, 0.96])
    figure.savefig(f"{output_prefix}_{region_mode}.png", dpi=300)
    plt.close(figure)


def run_pca(args):
    model_paths = list(args.model_paths)
    stage_names = list(args.stage_names)
    if args.include_untrained:
        model_paths = ["untrained"] + model_paths
        stage_names = [args.untrained_stage_name] + stage_names

    if len(model_paths) != len(stage_names):
        raise ValueError(
            "--model_paths and --stage_names must have the same length."
        )

    results = []
    for stage_name, model_path in zip(stage_names, model_paths):
        for region_mode in ["foreground", "background"]:
            print(f"[PCA] {stage_name} | {region_mode} | O4")
            results.append(
                collect_result(
                    model_path,
                    stage_name,
                    args.source_path,
                    args.target_path,
                    region_mode,
                    args.batch_size,
                    args.max_samples,
                    args.min_pixels,
                )
            )

    save_metrics(args.output_prefix + "_metrics.csv", results)
    save_points(args.output_prefix + "_points.csv", results)
    plot_results(results, stage_names, args.output_prefix, "foreground")
    plot_results(results, stage_names, args.output_prefix, "background")
    print(f"Saved O4 AFB PCA outputs: {args.output_prefix}")


def parse_args():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(
        description="O4 PCA visualization for AFB"
    )
    parser.add_argument(
        "--model_paths",
        nargs="+",
        default=[
            os.path.join(
                current_dir, "weight", "joint_da_best_model.pth"
            )
        ],
    )
    parser.add_argument("--stage_names", nargs="+", default=["AFB"])
    parser.add_argument("--include_untrained", action="store_true")
    parser.add_argument("--untrained_stage_name", default="Untrained")
    parser.add_argument(
        "--source_path",
        default=os.path.join(current_dir, "dataset", "source", "train"),
    )
    parser.add_argument(
        "--target_path",
        default=os.path.join(
            current_dir, "dataset", "target", "labeled_train"
        ),
    )
    parser.add_argument(
        "--output_prefix",
        default=os.path.join(
            current_dir, "pca_results", "source_target_afb_lmmd_o4"
        ),
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--min_pixels", type=int, default=8)
    return parser.parse_args()


if __name__ == "__main__":
    run_pca(parse_args())
