# -*- coding: utf-8 -*-

import os
import csv
import argparse
from time import time

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

try:
    from scipy import ndimage as ndi
except ImportError:
    ndi = None

try:
    from skimage.morphology import skeletonize as skimage_skeletonize
except ImportError:
    skimage_skeletonize = None

from data import TargetDataset
from net import SFD_Mamba2Net


# -----------------------------------------------------------------------------
# All summary metrics use the same batch-level aggregation principle as Dice
# -----------------------------------------------------------------------------

def dice_coefficient(logits, target, threshold=0.55, smooth=1e-10):
    """Exactly the same Dice calculation used by train_ssda.py.

    All images and pixels in one batch are accumulated together first.
    """
    with torch.no_grad():
        pred = (torch.sigmoid(logits) >= threshold).float()
        gt = (target > 0.5).float()
        inter = (pred * gt).sum()
        return (2.0 * inter + smooth) / (pred.sum() + gt.sum() + smooth)


def binary_batch_metrics_from_logits(logits, target, threshold=0.55):
    """Calculate one set of metrics over the whole batch.

    Dice, sensitivity and specificity follow train_ssda.py exactly.
    IoU and precision use the same batch-level TP/FP/FN aggregation so the
    retained test outputs follow the same aggregation principle.
    """
    with torch.no_grad():
        pred = (torch.sigmoid(logits) >= threshold).float()
        gt = (target > 0.5).float()

        tp = ((gt == 1) & (pred == 1)).sum().float()
        tn = ((gt == 0) & (pred == 0)).sum().float()
        fp = ((gt == 0) & (pred == 1)).sum().float()
        fn = ((gt == 1) & (pred == 0)).sum().float()

        dice = dice_coefficient(logits, target, threshold=threshold)
        iou = tp / (tp + fp + fn + 1e-10)
        precision = tp / (tp + fp + 1e-10)
        sensitivity = tp / (tp + fn + 1e-10)
        specificity = tn / (tn + fp + 1e-10)

        return {
            "dice": float(dice.item()),
            "iou": float(iou.item()),
            "precision": float(precision.item()),
            "sensitivity": float(sensitivity.item()),
            "specificity": float(specificity.item()),
        }


# -----------------------------------------------------------------------------
# Per-image metrics retained for per_image_metrics.csv
# -----------------------------------------------------------------------------

def binary_dice(pred, label, smooth=1e-10):
    """Per-image Dice using the same smoothing rule as train_ssda.py."""
    pred = (pred > 0).astype(np.float32)
    label = (label > 0.5).astype(np.float32)
    intersection = float(np.sum(pred * label))
    return (2.0 * intersection + smooth) / (
        float(np.sum(pred)) + float(np.sum(label)) + smooth
    )


def binary_iou(pred, label):
    pred = pred > 0
    label = label > 0.5
    intersection = np.sum(np.logical_and(pred, label))
    union = np.sum(np.logical_or(pred, label))
    return intersection / (union + 1e-10)


def binary_confusion_metrics(gt, pred):
    gt = gt > 0.5
    pred = pred > 0

    tp = int(np.sum(np.logical_and(gt, pred)))
    tn = int(np.sum(np.logical_and(~gt, ~pred)))
    fp = int(np.sum(np.logical_and(~gt, pred)))
    fn = int(np.sum(np.logical_and(gt, ~pred)))

    sensitivity = tp / (tp + fn + 1e-10)
    specificity = tn / (tn + fp + 1e-10)
    precision = tp / (tp + fp + 1e-10)

    return precision, sensitivity, specificity, tp, tn, fp, fn


def zhang_suen_thinning(mask):
    img = (mask > 0).astype(np.uint8)
    changed = True
    while changed:
        changed = False
        for step in (0, 1):
            padded = np.pad(img, 1, mode="constant")
            p2 = padded[:-2, 1:-1]
            p3 = padded[:-2, 2:]
            p4 = padded[1:-1, 2:]
            p5 = padded[2:, 2:]
            p6 = padded[2:, 1:-1]
            p7 = padded[2:, :-2]
            p8 = padded[1:-1, :-2]
            p9 = padded[:-2, :-2]
            neighbors = [p2, p3, p4, p5, p6, p7, p8, p9]
            n_count = sum(neighbors)
            transitions = (
                ((p2 == 0) & (p3 == 1)).astype(np.uint8)
                + ((p3 == 0) & (p4 == 1)).astype(np.uint8)
                + ((p4 == 0) & (p5 == 1)).astype(np.uint8)
                + ((p5 == 0) & (p6 == 1)).astype(np.uint8)
                + ((p6 == 0) & (p7 == 1)).astype(np.uint8)
                + ((p7 == 0) & (p8 == 1)).astype(np.uint8)
                + ((p8 == 0) & (p9 == 1)).astype(np.uint8)
                + ((p9 == 0) & (p2 == 1)).astype(np.uint8)
            )
            if step == 0:
                step_cond = (p2 * p4 * p6 == 0) & (p4 * p6 * p8 == 0)
            else:
                step_cond = (p2 * p4 * p8 == 0) & (p2 * p6 * p8 == 0)
            remove = (
                (img == 1)
                & (n_count >= 2)
                & (n_count <= 6)
                & (transitions == 1)
                & step_cond
            )
            if np.any(remove):
                img[remove] = 0
                changed = True
    return img.astype(bool)


def binary_cldice(pred, label):
    pred = pred > 0
    label = label > 0.5
    if not pred.any() and not label.any():
        return 1.0
    if not pred.any() or not label.any():
        return 0.0
    if skimage_skeletonize is not None:
        skel_pred = skimage_skeletonize(pred)
        skel_label = skimage_skeletonize(label)
    else:
        skel_pred = zhang_suen_thinning(pred)
        skel_label = zhang_suen_thinning(label)
    tprec = np.sum(skel_pred & label) / (np.sum(skel_pred) + 1e-10)
    tsens = np.sum(skel_label & pred) / (np.sum(skel_label) + 1e-10)
    return 2.0 * tprec * tsens / (tprec + tsens + 1e-10)


def binary_surface(mask):
    mask = mask > 0
    if not mask.any():
        return mask
    eroded = ndi.binary_erosion(mask) if ndi is not None else mask
    return mask ^ eroded


def surface_distances(pred, label):
    if ndi is None:
        return np.asarray([], dtype=np.float32)
    pred = pred > 0
    label = label > 0.5
    if not pred.any() and not label.any():
        return np.asarray([0.0], dtype=np.float32)
    if not pred.any() or not label.any():
        return np.asarray([np.nan], dtype=np.float32)
    pred_surface = binary_surface(pred)
    label_surface = binary_surface(label)
    dist_to_label = ndi.distance_transform_edt(~label_surface)
    dist_to_pred = ndi.distance_transform_edt(~pred_surface)
    distances = np.concatenate(
        [dist_to_label[pred_surface], dist_to_pred[label_surface]]
    )
    return distances.astype(np.float32)


def binary_assd(pred, label):
    distances = surface_distances(pred, label)
    if distances.size == 0 or np.isnan(distances).any():
        return np.nan
    return float(np.mean(distances))


def binary_batch_cldice(pred_batch, label_batch):
    """Compute one clDice value over a complete mini-batch.

    Skeletonization is still performed independently for each 2-D image so
    that different images are never connected. The topology intersections and
    skeleton lengths are then accumulated across the batch before clDice is
    calculated. This mirrors batch Dice, which accumulates TP/FP/FN-like pixel
    counts across all images before producing one value.
    """
    pred_batch = np.asarray(pred_batch) > 0
    label_batch = np.asarray(label_batch) > 0.5

    pred_centerline_in_label = 0.0
    pred_centerline_total = 0.0
    label_centerline_in_pred = 0.0
    label_centerline_total = 0.0

    for pred, label in zip(pred_batch, label_batch):
        if skimage_skeletonize is not None:
            skel_pred = skimage_skeletonize(pred) if pred.any() else pred
            skel_label = skimage_skeletonize(label) if label.any() else label
        else:
            skel_pred = zhang_suen_thinning(pred) if pred.any() else pred
            skel_label = zhang_suen_thinning(label) if label.any() else label

        pred_centerline_in_label += float(np.sum(skel_pred & label))
        pred_centerline_total += float(np.sum(skel_pred))
        label_centerline_in_pred += float(np.sum(skel_label & pred))
        label_centerline_total += float(np.sum(skel_label))

    # Match the original empty-mask convention: both empty -> 1, only one
    # side empty -> 0.
    if pred_centerline_total == 0 and label_centerline_total == 0:
        return 1.0
    if pred_centerline_total == 0 or label_centerline_total == 0:
        return 0.0

    tprec = pred_centerline_in_label / (pred_centerline_total + 1e-10)
    tsens = label_centerline_in_pred / (label_centerline_total + 1e-10)
    return float(
        2.0 * tprec * tsens / (tprec + tsens + 1e-10)
    )


def binary_batch_assd(pred_batch, label_batch):
    """Compute one ASSD value over a complete mini-batch.

    Surface distances are calculated image by image, then all valid directed
    surface distances in the batch are concatenated and averaged. Therefore,
    the batch contributes one ASSD value, following the same summary unit used
    by batch Dice. Empty/non-empty pairs remain invalid and are ignored, as in
    the original per-image ASSD summary.
    """
    valid_distances = []
    for pred, label in zip(pred_batch, label_batch):
        distances = surface_distances(pred, label)
        if distances.size == 0 or np.isnan(distances).any():
            continue
        valid_distances.append(distances)

    if not valid_distances:
        return np.nan
    return float(np.mean(np.concatenate(valid_distances)))


def mean_std(values, ignore_nan=False):
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return np.nan, np.nan, 0
    if ignore_nan:
        valid = arr[~np.isnan(arr)]
        if valid.size == 0:
            return np.nan, np.nan, 0
        return float(valid.mean()), float(valid.std(ddof=0)), int(valid.size)
    return float(arr.mean()), float(arr.std(ddof=0)), int(arr.size)


def save_csv(path, rows, fields):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def prediction(
    data_path,
    model_path,
    threshold=0.55,
    save_outputs=True,
    output_dir=None,
    batch_size=2,
    num_workers=0,
):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    current_dir = os.path.dirname(os.path.abspath(__file__))

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Test dataset not found: {data_path}")

    model_name = os.path.splitext(os.path.basename(model_path))[0]
    if output_dir is None:
        threshold_tag = f"{threshold:.3f}".replace(".", "p")
        output_dir = os.path.join(
            current_dir, "outputs", f"{model_name}_thr{threshold_tag}"
        )
    os.makedirs(output_dir, exist_ok=True)
    prediction_dir = os.path.join(output_dir, "predictions")
    if save_outputs:
        os.makedirs(prediction_dir, exist_ok=True)

    print(
        f"Test | Model {model_name} | Device {device} | "
        f"Threshold {threshold:.3f}"
    )

    # Use the same target-domain dataset settings as train_ssda.py.
    target_test_dataset = TargetDataset(
        data_path,
        with_label=True,
        augment=False,
    )
    data_loader = DataLoader(
        target_test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    net = SFD_Mamba2Net().to(device)
    state = torch.load(model_path, map_location=device)
    missing, unexpected = net.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(
            f"Load warning | Missing {len(missing)} | "
            f"Unexpected {len(unexpected)}"
        )
    net.eval()

    # Every final summary metric is represented by one value per mini-batch.
    # The reported mean/std are then calculated across those batch values.
    batch_metric_names = [
        "dice",
        "iou",
        "precision",
        "sensitivity",
        "specificity",
        "cldice",
        "assd",
    ]
    batch_records = {metric: [] for metric in batch_metric_names}
    per_image_rows = []
    st = time()

    for data, label, names in data_loader:
        with torch.no_grad():
            data = data.to(device)
            label_device = label.to(device)
            logits = net(data)

            # Region/confusion metrics are calculated once over all pixels in
            # the complete mini-batch, exactly like the training Dice.
            current_batch_metrics = binary_batch_metrics_from_logits(
                logits,
                label_device,
                threshold=threshold,
            )
            probs = torch.sigmoid(logits).cpu().numpy()

        # Convert [B, 1, H, W] to [B, H, W] without mixing different images.
        predictions_np = (probs >= threshold).astype(np.uint8)
        labels_np = (label.cpu().numpy() > 0.5).astype(np.uint8)
        if predictions_np.ndim == 4 and predictions_np.shape[1] == 1:
            predictions_np = predictions_np[:, 0]
        if labels_np.ndim == 4 and labels_np.shape[1] == 1:
            labels_np = labels_np[:, 0]

        # clDice and ASSD also produce exactly one value for this complete
        # batch. Their internal 2-D operations remain image-wise, while the
        # sufficient counts/distances are accumulated before one batch result
        # is produced.
        current_batch_metrics["cldice"] = binary_batch_cldice(
            predictions_np, labels_np
        )
        current_batch_metrics["assd"] = binary_batch_assd(
            predictions_np, labels_np
        )
        for metric in batch_metric_names:
            batch_records[metric].append(current_batch_metrics[metric])

        batch_size_now = predictions_np.shape[0]

        for i in range(batch_size_now):
            pred = predictions_np[i]
            label_np = labels_np[i]
            name = names[i]

            if save_outputs:
                Image.fromarray((pred * 255).astype(np.uint8)).save(
                    os.path.join(prediction_dir, name)
                )

            # Per-image rows remain available in the same CSV format.
            dice = binary_dice(pred, label_np)
            iou = binary_iou(pred, label_np)
            precision, sensitivity, specificity, tp, tn, fp, fn = (
                binary_confusion_metrics(label_np, pred)
            )
            cldice = binary_cldice(pred, label_np)
            assd = binary_assd(pred, label_np)
            gt_pixels = int(label_np.sum())
            pred_pixels = int(pred.sum())

            per_image_rows.append(
                {
                    "image_name": name,
                    "threshold": threshold,
                    "dice": dice,
                    "iou": iou,
                    "precision": precision,
                    "sensitivity": sensitivity,
                    "specificity": specificity,
                    "cldice": cldice,
                    "assd": assd,
                    "gt_pixels": gt_pixels,
                    "pred_pixels": pred_pixels,
                    "empty_prediction": int(pred_pixels == 0),
                    "empty_label": int(gt_pixels == 0),
                    "tp": tp,
                    "tn": tn,
                    "fp": fp,
                    "fn": fn,
                }
            )

    if not per_image_rows:
        raise RuntimeError(f"No test images found in: {data_path}")

    elapsed = time() - st
    summary = {
        "model_name": model_name,
        "model_path": os.path.abspath(model_path),
        "data_path": os.path.abspath(data_path),
        "threshold": threshold,
        "num_images": len(per_image_rows),
        "empty_prediction_count": sum(
            r["empty_prediction"] for r in per_image_rows
        ),
        "empty_label_count": sum(r["empty_label"] for r in per_image_rows),
        "elapsed_seconds": elapsed,
        "seconds_per_image": elapsed / len(per_image_rows),
    }

    # All metrics use the same final aggregation level:
    # one metric value per batch, followed by mean/std across test batches.
    for metric in batch_metric_names:
        mean, std, valid_count = mean_std(
            batch_records[metric],
            ignore_nan=(metric == "assd"),
        )
        summary[f"{metric}_mean"] = mean
        summary[f"{metric}_std"] = std
        summary[f"{metric}_valid_count"] = valid_count

    per_image_path = os.path.join(output_dir, "per_image_metrics.csv")
    summary_path = os.path.join(output_dir, "summary_metrics.csv")
    per_image_fields = [
        "image_name",
        "threshold",
        "dice",
        "iou",
        "precision",
        "sensitivity",
        "specificity",
        "cldice",
        "assd",
        "gt_pixels",
        "pred_pixels",
        "empty_prediction",
        "empty_label",
        "tp",
        "tn",
        "fp",
        "fn",
    ]
    summary_fields = [
        "model_name",
        "model_path",
        "data_path",
        "threshold",
        "num_images",
        "dice_mean",
        "dice_std",
        "dice_valid_count",
        "iou_mean",
        "iou_std",
        "iou_valid_count",
        "precision_mean",
        "precision_std",
        "precision_valid_count",
        "sensitivity_mean",
        "sensitivity_std",
        "sensitivity_valid_count",
        "specificity_mean",
        "specificity_std",
        "specificity_valid_count",
        "cldice_mean",
        "cldice_std",
        "cldice_valid_count",
        "assd_mean",
        "assd_std",
        "assd_valid_count",
        "empty_prediction_count",
        "empty_label_count",
        "elapsed_seconds",
        "seconds_per_image",
    ]
    save_csv(per_image_path, per_image_rows, per_image_fields)
    save_csv(summary_path, [summary], summary_fields)

    if ndi is None:
        print("Metric warning | scipy not installed | ASSD is NaN")

    print(
        f"Result | Dice:{summary['dice_mean']:.4f}, "
        f"IoU:{summary['iou_mean']:.4f}, "
        f"Precision:{summary['precision_mean']:.4f}, "
        f"Sensitivity:{summary['sensitivity_mean']:.4f}, "
        f"Specificity:{summary['specificity_mean']:.4f}, "
        f"clDice:{summary['cldice_mean']:.4f}, "
        f"ASSD:{summary['assd_mean']:.4f}"
    )
    print(
        f"Saved | Per-image {per_image_path} | Summary {summary_path} | "
        f"Predictions {'saved' if save_outputs else 'not saved'}"
    )
    return summary


def parse_args():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(
        description=(
            "Target-domain vessel segmentation testing with all summary "
            "metrics aggregated at batch level"
        )
    )
    parser.add_argument(
        "--data_path",
        default=os.path.join(current_dir, "dataset", "target", "test"),
    )
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--no_save", action="store_true")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help=(
            "All summary metrics are computed per batch; use the same "
            "batch size for every compared model."
        ),
    )
    parser.add_argument("--num_workers", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    prediction(
        data_path=args.data_path,
        model_path=args.model_path,
        threshold=args.threshold,
        save_outputs=not args.no_save,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
