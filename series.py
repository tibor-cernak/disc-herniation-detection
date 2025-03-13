import os

import cv2

from typing import Dict, List

import numpy as np
from pydicom.dataset import FileDataset
import nibabel as nib

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

from mri_type import MRIType


class Series:
    """Represents a single series in study."""

    def __init__(
        self,
        study_uid: str,
        series_uid: str,
        frames: List[FileDataset],
        mri_type: MRIType,
    ):
        """
        Initializes a Series object with study metadata and converts dicom frames to NIfTI format.

        Args:
            study_uid (str): Unique identifier of the study.
            series_uid (str): Unique identifier of the series.
            frames (List[FileDataset]): List of DICOM frames in the series.
            mri_type (MRIType): Type of MRI scan.
        """
        self.study_uid = study_uid
        self.series_uid = series_uid
        self.frames = sorted(frames, key=lambda x: getattr(x, "InstanceNumber", 0))
        self.mri_type = mri_type
        self.body_part_examined = self.get_body_part_examined()
        self.human_readable_name = self.get_human_readable_name()
        self.nifti_file = self.convert_to_nifti()

    def __str__(self) -> str:
        return f"Study ID: {self.study_uid} | Series ID: {self.series_uid} | {self.human_readable_name} | Frames: {len(self.frames)} | MRI Type: {self.mri_type.name}"

    def __len__(self) -> int:
        return len(self.frames)

    def convert_to_nifti(self):
        volume = np.stack(
            [frame.pixel_array.astype(np.float32) for frame in self.frames], axis=0
        )

        slice_thickness = float(self.frames[0].SliceThickness)
        pixel_spacing = [float(s) for s in self.frames[0].PixelSpacing]

        affine = np.diag([pixel_spacing[0], pixel_spacing[1], slice_thickness, 1])

        nifti_file = nib.Nifti1Image(volume, affine)

        return nifti_file

    def export_as_nifti(self, output_filename):
        """
        Saves the NIfTI file.

        Args:
            output_filename (str): Path where the NIfTI file should be saved.
        """
        nib.save(self.nifti_file, output_filename)

    def get_human_readable_name(self):
        first_frame = self.frames[0]
        if hasattr(first_frame, "SeriesDescription"):
            return first_frame.SeriesDescription
        elif hasattr(first_frame, "ProtocolName"):
            first_frame.ProtocolName
        else:
            return "Series"

    def get_body_part_examined(self):
        first_frame = self.frames[0]
        if hasattr(first_frame, "BodyPartExamined"):
            return first_frame.BodyPartExamined
        return "Unknown"

    def visualize(self):
        data = self.nifti_file.get_fdata()

        fig, ax = plt.subplots()
        fig.canvas.manager.set_window_title(
            f"Study: {self.study_uid} - {self.series_uid}"
        )
        plt.subplots_adjust(bottom=0.25)
        slice_index = data.shape[0] // 2
        img_display = ax.imshow(data[slice_index, :, :], cmap="gray")
        ax.set_title(f"Slice {slice_index}")

        ax_slider = plt.axes([0.2, 0.1, 0.65, 0.03])
        slider = Slider(
            ax_slider, "Slice", 0, data.shape[0] - 1, valinit=slice_index, valstep=1
        )

        def update(val):
            slice_idx = int(slider.val)
            img_display.set_data(data[slice_idx, :, :])
            ax.set_title(f"Slice {slice_idx}")
            fig.canvas.draw_idle()

        slider.on_changed(update)
        plt.show()
