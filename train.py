import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np
import os
from tqdm import tqdm

from models.vnet import VnetModel
from dataset import Dataset


from scipy.ndimage import gaussian_filter, maximum_filter


def find_strongest_peaks(heatmap):
    """
    Find the strongest peaks in a 3D heatmap, filtering out weak peaks.

    Args:
        heatmap (np.ndarray): 3D heatmap (z, y, x).

    Returns:
        List of tuples: Coordinates of the strongest peaks (z, y, x).
    """
    max_filtered = maximum_filter(heatmap, size=3)

    # Find local maxima and filter by minimum intensity
    peaks = heatmap == max_filtered
    z_coords, y_coords, x_coords = np.where(peaks)

    # Keep only the strongest peak for each (y, x) coordinate
    peak_dict = {}
    for z, y, x in zip(z_coords, y_coords, x_coords):
        key = (y, x)
        if key not in peak_dict or heatmap[z, y, x] > heatmap[peak_dict[key], y, x]:
            peak_dict[key] = z

    final_peaks = [(z, y, x) for (y, x), z in peak_dict.items()]
    return final_peaks


def visualize_prediction(model, dataset, index=0, threshold=0.3):
    """Visualize the MRI slice, ground truth heatmap, predicted heatmap, thresholded heatmap, and peaks."""
    model.eval()
    mri_volume, true_heatmap = dataset[index]

    mri_volume = mri_volume.unsqueeze(0).to(device)
    with torch.no_grad():
        predicted_heatmap = model(mri_volume).squeeze(0).cpu().numpy()

    mri_volume = mri_volume.squeeze().cpu().numpy()
    true_heatmap = true_heatmap.squeeze().cpu().numpy()
    predicted_heatmap = predicted_heatmap.squeeze(0)

    # Apply thresholding to the predicted heatmap
    thresholded_heatmap = np.where(predicted_heatmap >= threshold, predicted_heatmap, 0)

    # Find the strongest peaks in the thresholded heatmap
    strongest_peaks = find_strongest_peaks(thresholded_heatmap)

    num_slices = mri_volume.shape[0]
    slice_idx = min(num_slices // 2, predicted_heatmap.shape[0] - 1)

    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    plt.subplots_adjust(bottom=0.2)

    img_mri = axes[0].imshow(mri_volume[slice_idx], cmap="gray")
    img_true = axes[1].imshow(mri_volume[slice_idx], cmap="gray")
    heatmap_true = axes[1].imshow(true_heatmap[slice_idx], cmap="Reds", alpha=0.5)
    img_pred = axes[2].imshow(mri_volume[slice_idx], cmap="gray")
    heatmap_pred = axes[2].imshow(predicted_heatmap[slice_idx], cmap="Reds", alpha=0.5)
    img_thresh = axes[3].imshow(mri_volume[slice_idx], cmap="gray")
    heatmap_thresh = axes[3].imshow(
        thresholded_heatmap[slice_idx], cmap="Reds", alpha=0.5
    )
    img_peaks = axes[4].imshow(mri_volume[slice_idx], cmap="gray")

    # Overlay detected peaks
    for peak in strongest_peaks:
        if peak[0] == slice_idx:  # Only show peaks in the current slice
            axes[4].scatter(peak[2], peak[1], c="blue", marker="x", s=50)

    axes[0].set_title("MRI Slice")
    axes[1].set_title("Ground Truth Heatmap")
    axes[2].set_title("Predicted Heatmap")
    axes[3].set_title("Thresholded Heatmap")
    axes[4].set_title("Detected Peaks")

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
        heatmap_thresh.set_data(thresholded_heatmap[slice_idx])
        img_peaks.set_data(mri_volume[slice_idx])

        # Clear previous peaks and plot new ones
        axes[4].clear()
        axes[4].imshow(mri_volume[slice_idx], cmap="gray")
        for peak in strongest_peaks:
            if peak[0] == slice_idx:
                axes[4].scatter(peak[2], peak[1], c="blue", marker="x", s=50)

        fig.canvas.draw_idle()

    slider.on_changed(update)

    plt.show()


def train_model(
    model,
    dataloader,
    val_dataloader,
    criterion,
    optimizer,
    num_epochs,
    device,
    save_path,
    patience=5,
):
    """Train the model"""
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        for mri_volumes, heatmaps in tqdm(
            dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"
        ):
            mri_volumes = mri_volumes.to(device)
            heatmaps = heatmaps.to(device)

            outputs = model(mri_volumes)

            loss = criterion(outputs, heatmaps)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_epoch_loss = epoch_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_epoch_loss:.8f}")

        val_loss = evaluate_model(model, val_dataloader, criterion, device)
        print(f"Validation Loss: {val_loss:.8f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"Model saved to {save_path}")
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            print(f"No improvement for {epochs_without_improvement} epochs.")

            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch+1}.")
                break


def evaluate_model(model, dataloader, criterion, device):
    """Evaluate the model on the validation set."""
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for mri_volumes, heatmaps in dataloader:
            mri_volumes = mri_volumes.to(device)
            heatmaps = heatmaps.to(device)

            outputs = model(mri_volumes)

            loss = criterion(outputs, heatmaps)
            val_loss += loss.item()

    return val_loss / len(dataloader)


if __name__ == "__main__":
    num_epochs = 200
    batch_size = 1
    learning_rate = 1e-4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_path = "disc_herniation_detection_model.pth"
    patience = 5

    dataset = Dataset(
        mri_dir="train", annotation_file="train_annotations.json", transform=True
    )

    val_size = int(0.1 * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = VnetModel().to(device)

    if os.path.exists(save_path):
        print(f"Loading model from {save_path}...")
        model.load_state_dict(torch.load(save_path, map_location=device))
        print("Model loaded.")
        test_dataset = Dataset("test", "test_annotations.json")
        for i in range(len(test_dataset)):
            visualize_prediction(model, test_dataset, i)
    else:
        print(f"No model found at {save_path}. Training from scratch...")
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        train_model(
            model,
            train_dataloader,
            val_dataloader,
            criterion,
            optimizer,
            num_epochs,
            device,
            save_path,
            patience,
        )
