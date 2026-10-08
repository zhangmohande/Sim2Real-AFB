# -*- coding: utf-8 -*-
"""
Clean dataset utilities for supervised vessel segmentation experiments.

Supported experiments:
    1) target_only   : real target-domain labeled training
    2) source_only   : simulated source-domain labeled training
    3) joint_no_da   : source + target supervised training without alignment
    4) joint_da      : source + target supervised training with supervised alignment

Expected folder structure for every labeled split:
    split_root/
        ICA_PNG/*.png
        label/*.png

Image and label filenames must be identical.
"""

import os
import random
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

transform = transforms.Compose([transforms.ToTensor()])


# ---------------------------------------------------------------------
# Basic IO
# ---------------------------------------------------------------------
def _list_image_names(path):
    img_dir = os.path.join(path, "ICA_PNG")
    if not os.path.isdir(img_dir):
        raise FileNotFoundError(f"ICA_PNG folder not found: {img_dir}")
    names = [n for n in os.listdir(img_dir) if not n.startswith(".")]
    return sorted(names)


def resize_image(path, size=(512, 512)):
    img = Image.open(path).convert("L")
    img = img.resize(size, Image.BILINEAR)
    return img


def resize_mask(path, size=(512, 512)):
    mask = Image.open(path).convert("L")
    mask = mask.resize(size, Image.NEAREST)
    return mask


def _to_tensor_mask(mask):
    m = transform(mask)
    return (m > 0.5).float()


# ---------------------------------------------------------------------
# Synchronized image/mask augmentation
# ---------------------------------------------------------------------
class JointVesselAugment:
    """
    Apply synchronized geometry augmentation to image and mask.
    Intensity augmentation is applied to image only.
    """
    def __init__(
        self,
        enable=True,
        flip_prob=0.5,
        max_rotate=10.0,
        translate=0.04,
        scale_range=(0.92, 1.08),
        shear=0.0,
        brightness=0.12,
        contrast=0.12,
        gamma_range=(0.90, 1.10),
        noise_std=0.015,
        blur_prob=0.10,
        blur_radius=(0.3, 0.8),
        intensity_prob=0.8,
    ):
        self.enable = bool(enable)
        self.flip_prob = float(flip_prob)
        self.max_rotate = float(max_rotate)
        self.translate = float(translate)
        self.scale_range = tuple(scale_range)
        self.shear = float(shear)
        self.brightness = float(brightness)
        self.contrast = float(contrast)
        self.gamma_range = tuple(gamma_range)
        self.noise_std = float(noise_std)
        self.blur_prob = float(blur_prob)
        self.blur_radius = tuple(blur_radius)
        self.intensity_prob = float(intensity_prob)

    def _apply_affine(self, img, mask=None):
        w, h = img.size
        angle = random.uniform(-self.max_rotate, self.max_rotate)
        max_dx = int(round(self.translate * w))
        max_dy = int(round(self.translate * h))
        translations = (random.randint(-max_dx, max_dx), random.randint(-max_dy, max_dy))
        scale = random.uniform(self.scale_range[0], self.scale_range[1])
        shear = random.uniform(-self.shear, self.shear) if self.shear > 0 else 0.0

        img = TF.affine(
            img,
            angle=angle,
            translate=translations,
            scale=scale,
            shear=[shear, 0.0],
            interpolation=InterpolationMode.BILINEAR,
            fill=0,
        )
        if mask is not None:
            mask = TF.affine(
                mask,
                angle=angle,
                translate=translations,
                scale=scale,
                shear=[shear, 0.0],
                interpolation=InterpolationMode.NEAREST,
                fill=0,
            )
        return img, mask

    def _apply_intensity(self, img):
        if random.random() > self.intensity_prob:
            return img

        if self.brightness > 0:
            factor = random.uniform(1.0 - self.brightness, 1.0 + self.brightness)
            img = ImageEnhance.Brightness(img).enhance(factor)
        if self.contrast > 0:
            factor = random.uniform(1.0 - self.contrast, 1.0 + self.contrast)
            img = ImageEnhance.Contrast(img).enhance(factor)

        if self.gamma_range is not None:
            gamma = random.uniform(self.gamma_range[0], self.gamma_range[1])
            arr = np.asarray(img).astype(np.float32) / 255.0
            arr = np.power(np.clip(arr, 0.0, 1.0), gamma)
            img = Image.fromarray((arr * 255.0).astype(np.uint8), mode="L")

        if self.noise_std > 0:
            arr = np.asarray(img).astype(np.float32) / 255.0
            arr = arr + np.random.normal(0.0, self.noise_std, size=arr.shape).astype(np.float32)
            arr = np.clip(arr, 0.0, 1.0)
            img = Image.fromarray((arr * 255.0).astype(np.uint8), mode="L")

        if self.blur_prob > 0 and random.random() < self.blur_prob:
            radius = random.uniform(self.blur_radius[0], self.blur_radius[1])
            img = img.filter(ImageFilter.GaussianBlur(radius=radius))
        return img

    def __call__(self, image, mask=None):
        if not self.enable:
            return image, mask

        if random.random() < self.flip_prob:
            image = TF.hflip(image)
            if mask is not None:
                mask = TF.hflip(mask)
        if random.random() < self.flip_prob * 0.5:
            image = TF.vflip(image)
            if mask is not None:
                mask = TF.vflip(mask)

        image, mask = self._apply_affine(image, mask)
        image = self._apply_intensity(image)
        return image, mask


SOURCE_AUG = JointVesselAugment(
    enable=True,
    max_rotate=12.0,
    translate=0.05,
    scale_range=(0.90, 1.10),
    brightness=0.18,
    contrast=0.18,
    gamma_range=(0.85, 1.15),
    noise_std=0.020,
    blur_prob=0.15,
    intensity_prob=0.9,
)

TARGET_AUG = JointVesselAugment(
    enable=True,
    max_rotate=8.0,
    translate=0.03,
    scale_range=(0.94, 1.06),
    brightness=0.10,
    contrast=0.10,
    gamma_range=(0.92, 1.08),
    noise_std=0.010,
    blur_prob=0.08,
    intensity_prob=0.7,
)

NO_AUG = JointVesselAugment(enable=False)


# ---------------------------------------------------------------------
# Labeled datasets
# ---------------------------------------------------------------------
class LabeledVesselDataset(Dataset):
    """Generic labeled image/mask dataset."""
    def __init__(self, path, augment=False, aug_type="target"):
        self.path = path
        self.name = _list_image_names(path)
        self.augment = bool(augment)
        if not self.augment:
            self.aug = NO_AUG
        elif aug_type == "source":
            self.aug = SOURCE_AUG
        elif aug_type == "target":
            self.aug = TARGET_AUG
        else:
            raise ValueError(f"Unsupported aug_type: {aug_type}")

    def __len__(self):
        return len(self.name)

    def __getitem__(self, index):
        segment_name = self.name[index]
        image_path = os.path.join(self.path, "ICA_PNG", segment_name)
        mask_path = os.path.join(self.path, "label", segment_name)
        image = resize_image(image_path)
        mask = resize_mask(mask_path)
        image, mask = self.aug(image, mask)
        return transform(image), _to_tensor_mask(mask), segment_name


class SourceDataset(LabeledVesselDataset):
    def __init__(self, path, augment=False):
        super().__init__(path, augment=augment, aug_type="source")


class TargetDataset(LabeledVesselDataset):
    def __init__(self, path, with_label=True, augment=False):
        if not with_label:
            raise ValueError("Clean supervised version only supports labeled target data.")
        super().__init__(path, augment=augment, aug_type="target")


class MyDataset(LabeledVesselDataset):
    """Evaluation dataset. No augmentation."""
    def __init__(self, path):
        super().__init__(path, augment=False, aug_type="target")
