# -*- coding: utf-8 -*-

import argparse
import csv
import os
import random

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import ConcatDataset, DataLoader

from AFB import AFB
from data import SourceDataset, TargetDataset
from net import SFD_Mamba2Net


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SOURCE_TRAIN = os.path.join(CURRENT_DIR, "dataset", "source", "train")
DEFAULT_TARGET_TRAIN = os.path.join(
    CURRENT_DIR, "dataset", "target", "labeled_train"
)
DEFAULT_TARGET_VAL = os.path.join(CURRENT_DIR, "dataset", "target", "val")
DEFAULT_TARGET_TEST = os.path.join(CURRENT_DIR, "dataset", "target", "test")
DEFAULT_WEIGHT_DIR = os.path.join(CURRENT_DIR, "weight")
DEFAULT_LOG_DIR = os.path.join(CURRENT_DIR, "logs")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def dice_coefficient(logits, target, threshold=0.55, smooth=1e-10):
    with torch.no_grad():
        prediction = (torch.sigmoid(logits) >= threshold).float()
        target = (target > 0.5).float()
        intersection = (prediction * target).sum()
        return (2.0 * intersection + smooth) / (
            prediction.sum() + target.sum() + smooth
        )


def sensitivity_specificity(logits, target, threshold=0.55):
    with torch.no_grad():
        prediction = (torch.sigmoid(logits) >= threshold).float()
        target = (target > 0.5).float()
        true_positive = ((target == 1) & (prediction == 1)).sum().float()
        true_negative = ((target == 0) & (prediction == 0)).sum().float()
        false_positive = ((target == 0) & (prediction == 1)).sum().float()
        false_negative = ((target == 1) & (prediction == 0)).sum().float()
        sensitivity = true_positive / (
            true_positive + false_negative + 1e-10
        )
        specificity = true_negative / (
            true_negative + false_positive + 1e-10
        )
        return sensitivity, specificity


@torch.no_grad()
def validate_labeled(loader, model, device, loss_fn, threshold=0.55):
    model.eval()
    loss_sum = 0.0
    dice_sum = 0.0
    sensitivity_sum = 0.0
    specificity_sum = 0.0
    batch_count = 0

    for image, mask, _ in loader:
        image = image.to(device)
        mask = mask.to(device)
        logits = model(image)
        loss_sum += loss_fn(logits, mask).item()
        dice_sum += dice_coefficient(logits, mask, threshold).item()
        sensitivity, specificity = sensitivity_specificity(
            logits, mask, threshold
        )
        sensitivity_sum += sensitivity.item()
        specificity_sum += specificity.item()
        batch_count += 1

    batch_count = max(batch_count, 1)
    return {
        "loss": loss_sum / batch_count,
        "dice": dice_sum / batch_count,
        "sensitivity": sensitivity_sum / batch_count,
        "specificity": specificity_sum / batch_count,
    }


def build_optimizer(
    model,
    encoder_lr=1e-4,
    decoder_lr=1e-4,
    weight_decay=1e-4,
):
    encoder_prefix = (
        "c1",
        "c2",
        "c3",
        "c4",
        "c5",
        "d1",
        "d2",
        "d3",
        "d4",
        "mamba2",
        "CASE",
        "channel_attn",
    )
    encoder_parameters = []
    decoder_parameters = []

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith(encoder_prefix):
            encoder_parameters.append(parameter)
        else:
            decoder_parameters.append(parameter)

    return optim.Adam(
        [
            {"params": encoder_parameters, "lr": encoder_lr},
            {"params": decoder_parameters, "lr": decoder_lr},
        ],
        weight_decay=weight_decay,
    )


def build_afb_lmmd(args):
    return AFB(
        sigma=args.afb_lmmd_sigma,
        fg_weight=args.afb_fg_weight,
        bg_weight=args.afb_bg_weight,
        downsample_size=args.afb_downsample_size,
        max_samples=args.afb_max_samples,
        min_pixels=args.afb_min_pixels,
        normalize_feature=True,
        fg_direction=args.afb_fg_direction,
        bg_direction=args.afb_bg_direction,
    )


def zero_afb_lmmd_logs(device):
    zero = torch.tensor(0.0, device=device)
    return {
        "afb_lmmd": zero,
        "afb_lmmd_fg": zero,
        "afb_lmmd_bg": zero,
    }


def current_afb_lmmd_weight(args):
    if args.mode != "joint_da" or args.disable_afb_lmmd:
        return 0.0
    return min(
        args.afb_lmmd_weight_max,
        args.afb_lmmd_weight_max
        * args.current_epoch
        / max(args.afb_lmmd_warmup_epochs, 1),
    )


def init_csv(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fields = [
        "mode",
        "epoch",
        "train_loss",
        "source_loss",
        "target_loss",
        "mixed_loss",
        "source_dice",
        "target_dice",
        "mixed_dice",
        "afb_lmmd",
        "afb_lmmd_fg",
        "afb_lmmd_bg",
        "afb_lmmd_weight",
        "target_val_loss",
        "target_val_dice",
        "target_val_sens",
        "target_val_spec",
        "best_target_val_dice",
        "lr_encoder",
        "lr_decoder",
    ]
    with open(path, "w", newline="", encoding="utf-8") as file:
        csv.DictWriter(file, fieldnames=fields).writeheader()
    return fields


def append_csv(path, fields, row):
    with open(path, "a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writerow({key: row.get(key, "") for key in fields})


def average_meter(keys):
    return {key: 0.0 for key in keys}


def add_to_meter(meter, values):
    for key, value in values.items():
        if key in meter:
            meter[key] += float(value)


def divide_meter(meter, count):
    count = max(count, 1)
    return {key: value / count for key, value in meter.items()}


def clip_gradients(optimizer, grad_clip):
    if grad_clip is not None and grad_clip > 0:
        parameters = [
            parameter
            for group in optimizer.param_groups
            for parameter in group["params"]
        ]
        torch.nn.utils.clip_grad_norm_(parameters, grad_clip)


def train_single_domain_epoch(
    model,
    loader,
    loss_fn,
    optimizer,
    device,
    args,
):
    """Train target_only, source_only or naturally mixed joint_no_da."""
    model.train()
    meter = average_meter(
        [
            "train_loss",
            "source_loss",
            "target_loss",
            "mixed_loss",
            "source_dice",
            "target_dice",
            "mixed_dice",
        ]
    )
    batch_count = 0

    for image, mask, _ in loader:
        image = image.to(device)
        mask = mask.to(device)
        logits = model(image)
        loss = loss_fn(logits, mask)

        optimizer.zero_grad()
        loss.backward()
        clip_gradients(optimizer, args.grad_clip)
        optimizer.step()

        dice = dice_coefficient(logits.detach(), mask, args.threshold)
        values = {"train_loss": loss.item()}
        if args.mode == "target_only":
            values.update(
                {"target_loss": loss.item(), "target_dice": dice.item()}
            )
        elif args.mode == "source_only":
            values.update(
                {"source_loss": loss.item(), "source_dice": dice.item()}
            )
        elif args.mode == "joint_no_da":
            values.update(
                {"mixed_loss": loss.item(), "mixed_dice": dice.item()}
            )
        else:
            raise ValueError(
                f"Unsupported single-loader mode: {args.mode}"
            )

        add_to_meter(meter, values)
        batch_count += 1

    statistics = divide_meter(meter, batch_count)
    statistics.update(
        {
            "afb_lmmd": 0.0,
            "afb_lmmd_fg": 0.0,
            "afb_lmmd_bg": 0.0,
            "afb_lmmd_weight": 0.0,
        }
    )
    return statistics


def train_joint_epoch(
    model,
    source_loader,
    target_loader,
    loss_fn,
    optimizer,
    device,
    args,
    afb_lmmd=None,
):
    """Train paired source/target batches.

    Target-domain segmentation loss always has a fixed coefficient of 1:
        source_loss + target_loss + w_afb_lmmd * afb_lmmd_loss

    Only the O4 feature map is used by AFB.
    """
    model.train()
    meter = average_meter(
        [
            "train_loss",
            "source_loss",
            "target_loss",
            "mixed_loss",
            "source_dice",
            "target_dice",
            "mixed_dice",
            "afb_lmmd",
            "afb_lmmd_fg",
            "afb_lmmd_bg",
        ]
    )

    source_iterator = iter(source_loader)
    target_iterator = iter(target_loader)
    max_batches = max(len(source_loader), len(target_loader))
    afb_lmmd_weight = current_afb_lmmd_weight(args)
    use_alignment = afb_lmmd_weight > 0.0

    for _ in range(max_batches):
        try:
            source_image, source_mask, _ = next(source_iterator)
        except StopIteration:
            source_iterator = iter(source_loader)
            source_image, source_mask, _ = next(source_iterator)

        try:
            target_image, target_mask, _ = next(target_iterator)
        except StopIteration:
            target_iterator = iter(target_loader)
            target_image, target_mask, _ = next(target_iterator)

        source_image = source_image.to(device)
        source_mask = source_mask.to(device)
        target_image = target_image.to(device)
        target_mask = target_mask.to(device)

        if use_alignment:
            source_logits, source_features = model(
                source_image, return_feature=True
            )
            target_logits, target_features = model(
                target_image, return_feature=True
            )
        else:
            source_logits = model(source_image)
            target_logits = model(target_image)
            source_features = None
            target_features = None

        source_loss = loss_fn(source_logits, source_mask)
        target_loss = loss_fn(target_logits, target_mask)
        source_dice = dice_coefficient(
            source_logits.detach(), source_mask, args.threshold
        )
        target_dice = dice_coefficient(
            target_logits.detach(), target_mask, args.threshold
        )

        afb_lmmd_loss = torch.tensor(0.0, device=device)
        alignment_logs = zero_afb_lmmd_logs(device)
        if use_alignment:
            afb_lmmd_loss, alignment_logs = afb_lmmd(
                source_features["O4"],
                target_features["O4"],
                source_mask,
                target_mask,
            )

        # The target-domain supervised loss is always weighted by exactly 1.
        total_loss = (
            source_loss
            + target_loss
            + afb_lmmd_weight * afb_lmmd_loss
        )

        optimizer.zero_grad()
        total_loss.backward()
        clip_gradients(optimizer, args.grad_clip)
        optimizer.step()

        add_to_meter(
            meter,
            {
                "train_loss": total_loss.item(),
                "source_loss": source_loss.item(),
                "target_loss": target_loss.item(),
                "source_dice": source_dice.item(),
                "target_dice": target_dice.item(),
                "afb_lmmd": alignment_logs["afb_lmmd"].item(),
                "afb_lmmd_fg": alignment_logs["afb_lmmd_fg"].item(),
                "afb_lmmd_bg": alignment_logs["afb_lmmd_bg"].item(),
            },
        )

    statistics = divide_meter(meter, max_batches)
    statistics["afb_lmmd_weight"] = afb_lmmd_weight
    return statistics


def run_training(args):
    set_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.weight_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    if args.run_name is None or not args.run_name.strip():
        args.run_name = args.mode

    best_model_path = os.path.join(
        args.weight_dir, f"{args.run_name}_best_model.pth"
    )
    final_model_path = os.path.join(
        args.weight_dir, f"{args.run_name}_final_model.pth"
    )
    log_csv_path = os.path.join(
        args.log_dir, f"{args.run_name}_training_log.csv"
    )

    loss_fn = nn.BCEWithLogitsLoss()

    source_loader = None
    target_train_loader = None
    train_loader = None
    source_dataset = None
    target_train_dataset = None
    mixed_train_dataset = None
    source_count = 0
    target_train_count = 0
    mixed_train_count = 0

    if args.mode in ["source_only", "joint_no_da", "dual_no_da", "joint_da"]:
        source_dataset = SourceDataset(args.source_train_path, augment=True)
        source_count = len(source_dataset)
        if args.mode in ["source_only", "dual_no_da", "joint_da"]:
            source_loader = DataLoader(
                source_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
            )

    if args.mode in ["target_only", "joint_no_da", "dual_no_da", "joint_da"]:
        target_train_dataset = TargetDataset(
            args.target_train_path,
            with_label=True,
            augment=True,
        )
        target_train_count = len(target_train_dataset)
        if args.mode in ["target_only", "dual_no_da", "joint_da"]:
            target_train_loader = DataLoader(
                target_train_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
            )

    if args.mode == "joint_no_da":
        mixed_train_dataset = ConcatDataset(
            [source_dataset, target_train_dataset]
        )
        mixed_train_count = len(mixed_train_dataset)
        train_loader = DataLoader(
            mixed_train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
        )

    target_val_dataset = TargetDataset(
        args.target_val_path, with_label=True, augment=False
    )
    target_test_dataset = TargetDataset(
        args.target_test_path, with_label=True, augment=False
    )
    target_val_loader = DataLoader(
        target_val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    target_test_loader = DataLoader(
        target_test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = SFD_Mamba2Net().to(device)
    if args.pretrained_path:
        if not os.path.exists(args.pretrained_path):
            raise FileNotFoundError(
                f"Pretrained checkpoint not found: {args.pretrained_path}"
            )
        state = torch.load(args.pretrained_path, map_location=device)
        model.load_state_dict(state, strict=False)

    optimizer = build_optimizer(
        model,
        args.encoder_lr,
        args.decoder_lr,
        args.weight_decay,
    )
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=args.step_size, gamma=args.gamma
    )

    afb_lmmd = None
    if args.mode == "joint_da":
        afb_lmmd = build_afb_lmmd(args).to(device)

    fields = init_csv(log_csv_path)
    best_target_dice = -1.0
    patience_counter = 0
    best_epoch = 0

    if args.mode == "target_only":
        train_loader = target_train_loader
    elif args.mode == "source_only":
        train_loader = source_loader

    print(
        f"Run {args.run_name} | Mode {args.mode} | Device {device} | "
        f"Source {source_count} | TargetTrain {target_train_count} | "
        f"MixedTrain {mixed_train_count} | Val {len(target_val_dataset)} | "
        f"Test {len(target_test_dataset)}"
    )
    print(
        f"Parameters {sum(p.numel() for p in model.parameters()):,} | "
        f"Best {best_model_path} | Log {log_csv_path}"
    )
    print(
        f"Early stopping | min_epochs={args.min_epochs} | "
        f"patience={args.patience} | min_delta={args.min_delta}"
    )
    if args.mode in ["dual_no_da", "joint_da"]:
        print("Target supervised loss weight: 1.0 (fixed)")
    if args.mode == "joint_da":
        print(
            "Alignment: O4 AFB only | "
            f"FG direction={args.afb_fg_direction} | "
            f"BG direction={args.afb_bg_direction}"
        )

    for epoch in range(1, args.epochs + 1):
        args.current_epoch = epoch
        if args.mode in ["target_only", "source_only", "joint_no_da"]:
            train_statistics = train_single_domain_epoch(
                model,
                train_loader,
                loss_fn,
                optimizer,
                device,
                args,
            )
        else:
            train_statistics = train_joint_epoch(
                model,
                source_loader,
                target_train_loader,
                loss_fn,
                optimizer,
                device,
                args,
                afb_lmmd,
            )

        validation_statistics = validate_labeled(
            target_val_loader,
            model,
            device,
            loss_fn,
            threshold=args.threshold,
        )
        lr_encoder = optimizer.param_groups[0]["lr"]
        lr_decoder = (
            optimizer.param_groups[1]["lr"]
            if len(optimizer.param_groups) > 1
            else lr_encoder
        )

        improved = (
            validation_statistics["dice"]
            > best_target_dice + args.min_delta
        )
        if improved:
            best_target_dice = validation_statistics["dice"]
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1

        append_csv(
            log_csv_path,
            fields,
            {
                "mode": args.mode,
                "epoch": epoch,
                **train_statistics,
                "target_val_loss": validation_statistics["loss"],
                "target_val_dice": validation_statistics["dice"],
                "target_val_sens": validation_statistics["sensitivity"],
                "target_val_spec": validation_statistics["specificity"],
                "best_target_val_dice": best_target_dice,
                "lr_encoder": lr_encoder,
                "lr_decoder": lr_decoder,
            },
        )

        marker = " | Saved" if improved else ""
        if args.mode == "joint_no_da":
            train_dice_text = (
                f"MixedDice {train_statistics['mixed_dice']:.4f}"
            )
        else:
            train_dice_text = (
                f"SrcDice {train_statistics['source_dice']:.4f} | "
                f"TgtDice {train_statistics['target_dice']:.4f}"
            )

        alignment_text = ""
        if args.mode == "joint_da":
            alignment_text = (
                f" | AFB {train_statistics['afb_lmmd']:.6f}"
                f" (w={train_statistics['afb_lmmd_weight']:.3f})"
            )

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"Train {train_statistics['train_loss']:.6f} | "
            f"{train_dice_text}{alignment_text} | "
            f"ValDice {validation_statistics['dice']:.4f} | "
            f"Best {best_target_dice:.4f} (epoch {best_epoch}) | "
            f"EarlyStop {patience_counter}/{args.patience} | "
            f"LR {lr_encoder:.2e}{marker}"
        )

        if epoch >= args.min_epochs and patience_counter >= args.patience:
            print(
                f"Early stopping | Epoch {epoch} | "
                f"Best epoch {best_epoch} | "
                f"BestValDice {best_target_dice:.4f}"
            )
            break
        scheduler.step()

    torch.save(model.state_dict(), final_model_path)

    if os.path.exists(best_model_path):
        best_model = SFD_Mamba2Net().to(device)
        best_model.load_state_dict(
            torch.load(best_model_path, map_location=device), strict=True
        )
        test_statistics = validate_labeled(
            target_test_loader,
            best_model,
            device,
            loss_fn,
            threshold=args.threshold,
        )
        print(
            f"Test | Dice {test_statistics['dice']:.4f} | "
            f"Sens {test_statistics['sensitivity']:.4f} | "
            f"Spec {test_statistics['specificity']:.4f} | "
            f"Model {best_model_path}"
        )
        return best_model_path, test_statistics

    print(f"Training finished | Model {final_model_path}")
    return final_model_path, {}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Supervised sim-to-real vessel segmentation with O4 AFB"
        )
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=[
            "target_only",
            "source_only",
            "joint_no_da",
            "dual_no_da",
            "joint_da",
        ],
    )
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--source_train_path", default=DEFAULT_SOURCE_TRAIN)
    parser.add_argument("--target_train_path", default=DEFAULT_TARGET_TRAIN)
    parser.add_argument("--target_val_path", default=DEFAULT_TARGET_VAL)
    parser.add_argument("--target_test_path", default=DEFAULT_TARGET_TEST)
    parser.add_argument("--weight_dir", default=DEFAULT_WEIGHT_DIR)
    parser.add_argument("--log_dir", default=DEFAULT_LOG_DIR)
    parser.add_argument("--pretrained_path", default=None)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--min_epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--min_delta", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--encoder_lr", type=float, default=1e-4)
    parser.add_argument("--decoder_lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--step_size", type=int, default=50)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--afb_lmmd_weight_max",
        "--gamma_da_max",
        dest="afb_lmmd_weight_max",
        type=float,
        default=1.0,
        help="Maximum external weight of O4 AFB.",
    )
    parser.add_argument(
        "--afb_lmmd_warmup_epochs",
        "--gamma_da_warmup_epochs",
        dest="afb_lmmd_warmup_epochs",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--disable_afb_lmmd",
        "--disable_afb_da",
        dest="disable_afb_lmmd",
        action="store_true",
    )
    parser.add_argument("--afb_lmmd_sigma", type=float, default=2.0)
    parser.add_argument(
        "--afb_fg_weight",
        "--align_fg_weight",
        dest="afb_fg_weight",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--afb_bg_weight",
        "--align_bg_weight",
        dest="afb_bg_weight",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--afb_fg_direction",
        "--align_fg_direction",
        dest="afb_fg_direction",
        choices=["t2s", "s2t", "bidir"],
        default="t2s",
    )
    parser.add_argument(
        "--afb_bg_direction",
        "--align_bg_direction",
        dest="afb_bg_direction",
        choices=["t2s", "s2t", "bidir"],
        default="s2t",
    )
    parser.add_argument(
        "--afb_downsample_size",
        "--align_downsample_size",
        dest="afb_downsample_size",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--afb_max_samples",
        "--align_max_samples",
        dest="afb_max_samples",
        type=int,
        default=2048,
    )
    parser.add_argument(
        "--afb_min_pixels",
        "--align_min_pixels",
        dest="afb_min_pixels",
        type=int,
        default=16,
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_training(parse_args())
