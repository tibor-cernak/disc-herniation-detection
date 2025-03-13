import scipy.ndimage
from series import Series
from annotation import Annotation

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

import nibabel as nib
from typing import List
import json
import os
import numpy as np
from typing import List
import scipy

from skimage.transform import resize

PADDING_METHODS = ["constant", "edge"]


class AnnotatedSeries:
    """Represents a DICOM series with corresponding annotations"""

    def __init__(self, series: Series, annotations: List[Annotation]):
        """
        Initializes an AnnotatedSeries with an MRI series and its annotations.
        Resizes the height and width and also pads the slices to constant depth.

        Args:
            series (Series): The MRI series object.
            annotations (List[Annotation]): List of annotations for the series.
        """
        self.series: Series = series
        self.annotations: List[Annotation] = annotations
        self.resize_nifti(256, 256)
        self.adjust_depth("constant", 32)

    def resize_nifti(self, new_width: int = 256, new_height: int = 256) -> None:
        """
        Resizes the width and height of each slice in the NIfTI volume.

        Args:
            new_width (int): Desired width.
            new_height (int): Desired height.
        """
        data = self.series.nifti_file.get_fdata()
        original_shape = data.shape
        resized_data = np.array(
            [
                resize(
                    slice,
                    (new_width, new_height),
                    anti_aliasing=True,
                    preserve_range=True,
                )
                for slice in data
            ]
        )

        resized_data = resized_data.astype(data.dtype)

        affine = self.series.nifti_file.affine.copy()
        scale_x = original_shape[1] / new_width
        scale_y = original_shape[2] / new_height
        affine[0, 0] *= scale_x
        affine[1, 1] *= scale_y

        self.series.nifti_file = nib.Nifti1Image(resized_data, affine)

    def adjust_depth(self, method: str = "constant", target_depth: int = 32) -> None:
        """
        Adjusts the number of slices in the MRI volume by either padding or trimming.

        Args:
            method (str): Padding method ('constant' or 'replicate').
            target_depth (int): Desired number of slices.
        """
        if method not in PADDING_METHODS:
            raise Exception(
                f"Unknown padding method: {method}. Should be 'constant' or 'replicate'"
            )

        data = self.series.nifti_file.get_fdata()
        num_slices = data.shape[0]

        if num_slices == target_depth:
            return

        if num_slices > target_depth:
            trim_before = (num_slices - target_depth) // 2
            trim_after = num_slices - target_depth - trim_before
            data = data[trim_before : num_slices - trim_after]

            new_affine = self.series.nifti_file.affine.copy()
            new_affine[:3, 3] += trim_before * new_affine[:3, 2]

            self.annotations = [
                annotation
                for annotation in self.annotations
                if trim_before <= annotation.slice < num_slices - trim_after
            ]
            for annotation in self.annotations:
                annotation.slice -= trim_before

        else:
            pad_before = (target_depth - num_slices) // 2
            pad_after = target_depth - num_slices - pad_before
            padding = ((pad_before, pad_after), (0, 0), (0, 0))

            if method == "constant":
                data = np.pad(data, padding, mode="constant", constant_values=0)
            elif method == "replicate":
                data = np.pad(data, padding, mode="edge")

            new_affine = self.series.nifti_file.affine.copy()
            new_affine[:3, 3] -= pad_before * new_affine[:3, 2]

            for annotation in self.annotations:
                annotation.slice += pad_before

        self.series.nifti_file = nib.Nifti1Image(data, new_affine)

    def export_annotations(self, output_path: str = "annotations.json") -> None:
        """
        Exports the annotations for the series.

        Args:
            output_path (str): Path to the output JSON file.
        """
        data = self.series.nifti_file.get_fdata()
        depth, height, width = data.shape

        if os.path.exists(output_path):
            with open(output_path, "r") as f:
                annotation_file = json.load(f)
        else:
            annotation_file = {
                "images": [],
                "annotations": [],
            }

        existing_image_ids = {img["id"] for img in annotation_file["images"]}

        if self.series.series_uid not in existing_image_ids:
            annotation_file["images"].append(
                {
                    "id": self.series.series_uid,
                    "file_name": f"{self.series.series_uid}.nii",
                    "depth": depth,
                    "height": height,
                    "width": width,
                }
            )

        existing_annotation_ids = {ann["id"] for ann in annotation_file["annotations"]}
        next_annotation_id = max(existing_annotation_ids, default=0) + 1

        for ann in self.annotations:
            x1, y1 = ann.p1[0], ann.p1[1]
            x2, y2 = ann.p2[0], ann.p2[1]
            slice = ann.slice - 1

            # Compute center of the annotated line
            x = round((x1 + x2) / 2 * width, 2)
            y = round((y1 + y2) / 2 * height, 2)

            point = [x, y, slice]

            annotation_file["annotations"].append(
                {
                    "id": next_annotation_id,
                    "image_id": self.series.series_uid,
                    "point": point,
                }
            )

            next_annotation_id += 1

        with open(output_path, "w") as f:
            json.dump(annotation_file, f, indent=4)

    def export_nifti(self, output_dir: str = "dataset"):
        """
        Saves the modified NIfTI volume file to the specified directory.

        Args:
            output_dir (str): Target directory for saving the file.
        """
        os.makedirs(output_dir, exist_ok=True)

        output_path = os.path.join(output_dir, f"{self.series.series_uid}.nii")

        nib.save(self.series.nifti_file, output_path)

    def visualize(self) -> None:
        """
        Displays the series along with annotations.
        """
        data = self.series.nifti_file.get_fdata()
        img_height, img_width = (
            data.shape[1],
            data.shape[2],
        )
        num_slices = data.shape[0]

        fig, ax = plt.subplots()
        fig.canvas.manager.set_window_title(
            f"Study: {self.series.study_uid} - {self.series.series_uid}"
        )
        plt.subplots_adjust(bottom=0.25)

        slice_index = num_slices // 2
        img_display = ax.imshow(data[slice_index - 1, :, :], cmap="gray")
        ax.set_title(f"Slice {slice_index}")

        def denormalize(point):
            return int(point[0] * img_width), int(point[1] * img_height)

        annotation_lines = []
        for annotation in self.annotations:
            if annotation.slice == slice_index:
                p1 = denormalize(annotation.p1)
                p2 = denormalize(annotation.p2)
                (line,) = ax.plot([p1[0], p2[0]], [p1[1], p2[1]], "r-", linewidth=2)
                annotation_lines.append(line)

        ax_slider = plt.axes([0.2, 0.1, 0.65, 0.03])
        slider = Slider(
            ax_slider, "Slice", 1, num_slices, valinit=slice_index, valstep=1
        )

        def update(val):
            slice_idx = int(slider.val)
            img_display.set_data(data[slice_idx - 1, :, :])
            ax.set_title(f"Slice {slice_idx}")

            for line in annotation_lines:
                line.remove()
            annotation_lines.clear()

            for annotation in self.annotations:
                if annotation.slice == slice_idx:
                    p1 = denormalize(annotation.p1)
                    p2 = denormalize(annotation.p2)
                    (line,) = ax.plot([p1[0], p2[0]], [p1[1], p2[1]], "r-", linewidth=2)
                    annotation_lines.append(line)

            fig.canvas.draw_idle()

        slider.on_changed(update)
        plt.show()
