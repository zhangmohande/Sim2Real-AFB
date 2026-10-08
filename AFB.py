# -*- coding: utf-8 -*-
"""O4-only Asymmetric Foreground-Background alignment (AFB).

This module contains only the proposed local distribution-alignment loss:
    - O4 local feature alignment only
    - foreground/background class-aware RBF-MMD
    - optional asymmetric optimization directions

Ordinary global MMD, CORAL and O3 alignment are intentionally removed.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AFB(nn.Module):
    """Asymmetric Foreground-Background Local Maximum Mean Discrepancy."""

    def __init__(
        self,
        sigma=2.0,
        sigmas=None,
        fg_weight=1.0,
        bg_weight=1.0,
        downsample_size=64,
        max_samples=2048,
        min_pixels=16,
        normalize_feature=True,
        fg_direction="t2s",
        bg_direction="s2t",
    ):
        super().__init__()
        self.sigma = float(sigma)
        if sigmas is None:
            self.sigmas = [
                self.sigma * 0.5,
                self.sigma,
                self.sigma * 2.0,
                self.sigma * 4.0,
            ]
        else:
            self.sigmas = [float(value) for value in sigmas]

        self.fg_weight = float(fg_weight)
        self.bg_weight = float(bg_weight)
        self.downsample_size = int(downsample_size)
        self.max_samples = int(max_samples)
        self.min_pixels = int(min_pixels)
        self.normalize_feature = bool(normalize_feature)
        self.fg_direction = self._normalize_direction(
            fg_direction, "fg_direction"
        )
        self.bg_direction = self._normalize_direction(
            bg_direction, "bg_direction"
        )

    @staticmethod
    def _normalize_direction(direction, name):
        direction = str(direction).lower()
        aliases = {
            "target_to_source": "t2s",
            "target-source": "t2s",
            "target2source": "t2s",
            "source_to_target": "s2t",
            "source-target": "s2t",
            "source2target": "s2t",
            "bi": "bidir",
            "both": "bidir",
            "bidirectional": "bidir",
        }
        direction = aliases.get(direction, direction)
        if direction not in {"t2s", "s2t", "bidir"}:
            raise ValueError(
                f"{name} must be one of ['t2s', 's2t', 'bidir'], "
                f"got {direction!r}"
            )
        return direction

    @staticmethod
    def _zero(reference):
        return reference.new_tensor(0.0)

    def _normalize(self, features):
        if self.normalize_feature:
            return F.normalize(features, p=2, dim=1)
        return features

    def _sample(self, features):
        if features.shape[0] <= self.max_samples:
            return features
        indices = torch.randperm(
            features.shape[0], device=features.device
        )[: self.max_samples]
        return features[indices]

    def _downsample_regions(self, feature, mask):
        if feature.dim() != 4:
            raise RuntimeError(
                f"O4 feature must be [B,C,H,W], got {tuple(feature.shape)}"
            )
        if mask.dim() != 4:
            raise RuntimeError(
                f"Mask must be [B,1,H,W], got {tuple(mask.shape)}"
            )

        size = self.downsample_size
        feature = F.interpolate(
            feature, size=(size, size), mode="area"
        )
        mask = F.interpolate(
            mask.float(), size=(size, size), mode="nearest"
        )
        foreground = mask > 0.5
        background = ~foreground

        batch, channels, _, _ = feature.shape
        feature = (
            feature.flatten(2)
            .permute(0, 2, 1)
            .reshape(batch * size * size, channels)
        )
        feature = self._normalize(feature)
        foreground = foreground.reshape(batch * size * size)
        background = background.reshape(batch * size * size)
        return feature, foreground, background

    def _rbf_kernel(self, x, y):
        squared_distance = torch.cdist(x, y, p=2).pow(2)
        kernel = torch.zeros_like(squared_distance)
        for sigma in self.sigmas:
            kernel = kernel + torch.exp(
                -squared_distance / (2.0 * sigma * sigma + 1e-8)
            )
        return kernel / max(len(self.sigmas), 1)

    def _mmd(self, x, y):
        x = self._sample(self._normalize(x))
        y = self._sample(self._normalize(y))
        if x.shape[0] < 1 or y.shape[0] < 1:
            return self._zero(x if x.numel() > 0 else y)

        kernel_xx = self._rbf_kernel(x, x)
        kernel_yy = self._rbf_kernel(y, y)
        kernel_xy = self._rbf_kernel(x, y)
        return torch.clamp(
            kernel_xx.mean() + kernel_yy.mean() - 2.0 * kernel_xy.mean(),
            min=0.0,
        )

    def _select_direction(self, source, target, direction):
        if direction == "t2s":
            return source.detach(), target
        if direction == "s2t":
            return source, target.detach()
        return source, target

    def _region_loss(
        self,
        source_feature,
        target_feature,
        source_region,
        target_region,
        direction,
    ):
        source_selected = source_feature[source_region]
        target_selected = target_feature[target_region]

        if (
            source_selected.shape[0] < self.min_pixels
            or target_selected.shape[0] < self.min_pixels
        ):
            return self._zero(source_feature), 0.0

        source_selected, target_selected = self._select_direction(
            source_selected, target_selected, direction
        )
        return self._mmd(source_selected, target_selected), 1.0

    def forward(
        self,
        source_o4,
        target_o4,
        source_mask,
        target_mask,
    ):
        source_feature, source_fg, source_bg = self._downsample_regions(
            source_o4, source_mask
        )
        target_feature, target_fg, target_bg = self._downsample_regions(
            target_o4, target_mask
        )

        foreground_loss, foreground_valid = self._region_loss(
            source_feature,
            target_feature,
            source_fg,
            target_fg,
            self.fg_direction,
        )
        background_loss, background_valid = self._region_loss(
            source_feature,
            target_feature,
            source_bg,
            target_bg,
            self.bg_direction,
        )

        valid_weight = (
            self.fg_weight * foreground_valid
            + self.bg_weight * background_valid
        )
        if valid_weight <= 0:
            total_loss = self._zero(source_o4)
        else:
            total_loss = (
                self.fg_weight * foreground_loss
                + self.bg_weight * background_loss
            ) / valid_weight

        logs = {
            "afb_lmmd": total_loss.detach(),
            "afb_lmmd_fg": foreground_loss.detach(),
            "afb_lmmd_bg": background_loss.detach(),
            "afb_lmmd_fg_valid": float(foreground_valid),
            "afb_lmmd_bg_valid": float(background_valid),
        }
        return total_loss, logs


__all__ = ["AFB"]
