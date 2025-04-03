import os
import torch

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np

from scipy.ndimage import label

from dataset import Dataset
from vnet import VNetModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def binary_threshold(heatmap, threshold=0.25):
    heatmap = torch.as_tensor(heatmap)
    return (heatmap >= threshold).float()


def extract_bounding_boxes(thresholded_heatmap):
    """Extract bounding boxes from the thresholded heatmap."""
    labeled_array, num_features = label(thresholded_heatmap)
    bounding_boxes = []

    for region_id in range(1, num_features + 1):
        positions = np.argwhere(labeled_array == region_id)
        min_z, min_y, min_x = positions.min(axis=0)
        max_z, max_y, max_x = positions.max(axis=0)
        bounding_boxes.append((min_x, min_y, min_z, max_x, max_y, max_z))

    return bounding_boxes


def plot_bounding_box(ax, box, color="r"):
    """Helper function to plot a single bounding box."""
    x_min, y_min, z_min = box[0], box[1], box[2]
    x_max, y_max, z_max = box[3], box[4], box[5]

    corners = [
        (x_min, y_min, z_min),
        (x_max, y_min, z_min),
        (x_max, y_max, z_min),
        (x_min, y_max, z_min),
        (x_min, y_min, z_max),
        (x_max, y_min, z_max),
        (x_max, y_max, z_max),
        (x_min, y_max, z_max),
    ]

    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]

    for edge in edges:
        p1, p2 = edge
        ax.plot(
            [corners[p1][0], corners[p2][0]],
            [corners[p1][1], corners[p2][1]],
            [corners[p1][2], corners[p2][2]],
            color,
        )


def visualize_3d_bounding_boxes(
    ground_truth_boxes, predicted_boxes, volume_shape=(256, 256, 32)
):
    """Visualize predicted and ground truth bounding boxes in 3D space."""
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")

    for box in predicted_boxes:
        plot_bounding_box(ax, box, color="r")

    for box in ground_truth_boxes:
        plot_bounding_box(ax, box, color="b")

    ax.set_xlabel("X (0-256)")
    ax.set_ylabel("Y (0-256)")
    ax.set_zlabel("Z (0-32)")
    ax.set_xlim(0, volume_shape[0])
    ax.set_ylim(0, volume_shape[1])
    ax.set_zlim(0, volume_shape[2])

    plt.show()


def visualize_prediction(model, dataset, index=0, threshold=0.25):
    """Visualize the MRI slice, ground truth heatmap, predicted heatmap, thresholded heatmap"""
    model.eval()
    mri_volume, true_heatmap = dataset[index]
    mri_volume = mri_volume.unsqueeze(0).to(device)

    with torch.no_grad():
        predicted_heatmap = model(mri_volume).squeeze(0).cpu().numpy()

    mri_volume = mri_volume.squeeze().cpu().numpy()
    true_heatmap = true_heatmap.squeeze().cpu().numpy()
    predicted_heatmap = predicted_heatmap.squeeze(0)

    thresholded_truth_heatmap = binary_threshold(true_heatmap, threshold)
    thresholded_pred_heatmap = binary_threshold(predicted_heatmap, threshold)
    true_bboxes = extract_bounding_boxes(thresholded_truth_heatmap)
    pred_bboxes = extract_bounding_boxes(thresholded_pred_heatmap)

    visualize_3d_bounding_boxes(true_bboxes, pred_bboxes)

    num_slices = mri_volume.shape[0]
    slice_idx = min(num_slices // 2, predicted_heatmap.shape[0] - 1)

    # Increase the figure size
    fig, axes = plt.subplots(1, 4, figsize=(80, 16))  # Increased figure size
    plt.subplots_adjust(bottom=0.2)

    img_mri = axes[0].imshow(mri_volume[slice_idx], cmap="gray")
    img_true = axes[1].imshow(mri_volume[slice_idx], cmap="gray")
    heatmap_true = axes[1].imshow(true_heatmap[slice_idx], cmap="Reds", alpha=0.5)
    img_pred = axes[2].imshow(mri_volume[slice_idx], cmap="gray")
    heatmap_pred = axes[2].imshow(predicted_heatmap[slice_idx], cmap="Reds", alpha=0.5)
    img_thresh = axes[3].imshow(mri_volume[slice_idx], cmap="gray")
    heatmap_thresh = axes[3].imshow(
        thresholded_pred_heatmap[slice_idx], cmap="Reds", alpha=0.5
    )

    axes[0].set_title("MRI Slice")
    axes[1].set_title("Ground Truth Heatmap")
    axes[2].set_title("Predicted Heatmap")
    axes[3].set_title("Thresholded Heatmap")

    ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
    slider = Slider(ax_slider, "Slice", 0, num_slices - 1, valinit=slice_idx, valstep=1)

    def update(val):
        slice_idx = int(slider.val)
        slice_idx = min(slice_idx, predicted_heatmap.shape[0] - 1)
        img_mri.set_data(mri_volume[slice_idx])
        img_true.set_data(mri_volume[slice_idx])
        heatmap_true.set_data(true_heatmap[slice_idx])
        img_pred.set_data(mri_volume[slice_idx])
        heatmap_pred.set_data(predicted_heatmap[slice_idx])
        img_thresh.set_data(mri_volume[slice_idx])
        heatmap_thresh.set_data(thresholded_pred_heatmap[slice_idx])

        fig.canvas.draw_idle()

    slider.on_changed(update)

    plt.show()


if __name__ == "__main__":
    save_path = "trained_models/disc_herniation_detection_model_uran4.pth"
    model = VNetModel().to(device)
    if os.path.exists(save_path):
        print(f"Loading model from {save_path}")
        model.load_state_dict(torch.load(save_path, map_location=device))
        print("Model loaded.")
        test_dataset = Dataset("test", "test_annotations.json")
        indices = [0, 2, 3, 4, 5]
        for i in indices:
            visualize_prediction(model, test_dataset, i)
    else:
        print(f"No model found at {save_path}.")
