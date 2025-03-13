import os
import torch
import torch.utils.data as data
import nibabel as nib
import numpy as np
import json
import scipy.ndimage
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

from typing import Tuple, List

from monai.transforms import (
    Compose,
    RandAffine,
    RandGaussianNoise,
    RandAdjustContrast,
    ScaleIntensity,
)

# Define spatial transformations for 3D volumes
spatial_transforms = Compose(
    [
        RandAffine(
            prob=0.5,
            rotate_range=(
                np.pi / 18,
                0,
                0,
            ),  # rotation
            scale_range=(0.1, 0.1, 0.1),  # zoom
            padding_mode="border",
            mode="bilinear",
        ),
    ]
)

# Define intensity transformations (applied only to the MRI volume)
intensity_transforms = Compose(
    [
        ScaleIntensity(),
        RandGaussianNoise(prob=0.5, std=0.05),
        RandAdjustContrast(prob=0.5, gamma=(0.5, 2.0)),
    ]
)


class Dataset(data.Dataset):
    def __init__(
        self,
        mri_dir: str,
        annotation_file: str,
        volume_shape: Tuple[int, int, int] = (32, 256, 256),
        sigma: float = 1.5,
        transform: bool = False,
    ):
        """
        MRI Dataset with heatmap-based point annotations.

        Args:
            mri_dir (str): Directory containing .nii MRI volumes.
            annotation_file (str): Path to the JSON file with annotations.
            volume_shape (Tuple[int, int, int]): Target shape of the volume (Depth, Height, Width).
            sigma (float): Gaussian standard deviation for heatmap smoothing.
            transform (bool): Whether to apply data augmentation transforms.
        """
        self.mri_dir = mri_dir
        self.sigma = sigma
        self.volume_shape = volume_shape
        self.transform = transform

        # Load annotation file
        with open(annotation_file, "r") as f:
            self.annotations = json.load(f)

        self.image_data = {img["id"]: img for img in self.annotations["images"]}
        self.annotation_data = {
            ann["image_id"]: [] for ann in self.annotations["annotations"]
        }

        for ann in self.annotations["annotations"]:
            self.annotation_data[ann["image_id"]].append(ann)

        self.file_list = list(self.image_data.keys())

    def __len__(self):
        return len(self.file_list)

    def load_nifti(self, file_path: str) -> np.ndarray:
        """
        Loads and normalizes a NIfTI MRI volume.

        Args:
            file_path (str): Path to the NIfTI file.

        Returns:
            np.ndarray: Normalized MRI volume.
        """
        nii = nib.load(file_path)
        data = nii.get_fdata()
        data = np.clip(data, np.percentile(data, 1), np.percentile(data, 99))
        data = (data - data.min()) / (data.max() - data.min())
        return data.astype(np.float32)

    def generate_heatmap(self, points: List[List[float]]) -> np.ndarray:
        """
        Generates a 3D heatmap for given points.

        Args:
            points (List[List[float]]): List of points [x, y, z].

        Returns:
            np.ndarray: Heatmap of the same shape as the MRI volume.
        """
        heatmap = np.zeros(self.volume_shape, dtype=np.float32)

        for x, y, z in points:
            x, y, z = int(x), int(y), int(z)

            if (
                0 <= z < self.volume_shape[0]
                and 0 <= y < self.volume_shape[1]
                and 0 <= x < self.volume_shape[2]
            ):
                heatmap[z, y, x] = 1

        heatmap = scipy.ndimage.gaussian_filter(heatmap, sigma=self.sigma)

        if heatmap.max() > 0:
            heatmap /= heatmap.max()

        heatmap[heatmap < 0.1] = 0

        return heatmap

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Loads an MRI volume and its corresponding heatmap.

        Args:
            idx (int): Index of the sample.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: MRI volume and heatmap.
        """
        file_id = self.file_list[idx]
        file_path = os.path.join(self.mri_dir, f"{file_id}.nii")

        mri_volume = self.load_nifti(file_path)

        points = [x["point"] for x in self.annotation_data.get(file_id, [])]
        heatmap = self.generate_heatmap(points)

        mri_volume = torch.tensor(mri_volume).unsqueeze(0)  # (1, D, H, W)
        heatmap = torch.tensor(heatmap).unsqueeze(0)  # (1, D, H, W)

        if self.transform:
            combined = torch.cat([mri_volume, heatmap], dim=0)
            combined = spatial_transforms(combined)

            mri_volume, heatmap = combined[0:1, :, :, :], combined[1:2, :, :, :]

            mri_volume = intensity_transforms(mri_volume)

        return mri_volume, heatmap

    def visualize_sample(self, idx: int) -> None:
        """Visualize an MRI sample along with its generated heatmap."""
        mri_volume, heatmap = self[idx]
        mri_volume, heatmap = mri_volume.squeeze().numpy(), heatmap.squeeze().numpy()
        num_slices = mri_volume.shape[0]

        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        plt.subplots_adjust(bottom=0.25)

        slice_idx = num_slices // 2

        img_display = ax[0].imshow(mri_volume[slice_idx], cmap="gray")
        ax[0].set_title(f"MRI Slice {slice_idx}")

        heatmap_display = ax[1].imshow(mri_volume[slice_idx], cmap="gray")
        heatmap_overlay = ax[1].imshow(heatmap[slice_idx], cmap="Reds", alpha=0.5)
        ax[1].set_title(f"Heatmap Slice {slice_idx}")

        ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
        slider = Slider(
            ax_slider, "Slice", 0, num_slices - 1, valinit=slice_idx, valstep=1
        )

        def update(val):
            slice_idx = int(slider.val)
            img_display.set_data(mri_volume[slice_idx])
            heatmap_display.set_data(mri_volume[slice_idx])
            heatmap_overlay.set_data(heatmap[slice_idx])

            ax[0].set_title(f"MRI Slice {slice_idx}")
            ax[1].set_title(f"Heatmap Slice {slice_idx}")

            fig.canvas.draw_idle()

        slider.on_changed(update)
        plt.show()


if __name__ == "__main__":
    dataset = Dataset("dataset", "annotations.json", transform=True)
    for i in range(180, 200):
        dataset.visualize_sample(i)
