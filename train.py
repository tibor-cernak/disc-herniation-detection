import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
from eval import evaluate_model
from visualize import visualize_prediction

from vnet import VNetModel, AttentionVNetModel
from dataset import Dataset

import csv


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
    log_file_path="loss_log.csv",
):
    """Train the model"""
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    if not os.path.exists(log_file_path):
        with open(log_file_path, mode="w", newline="") as file:
            writer = csv.writer(file, delimiter=";")
            writer.writerow(["epoch", "training_loss", "validation_loss", "dice_score"])

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

        val_loss, dice_score = evaluate_model(model, val_dataloader, criterion, device)
        print(f"Validation Loss: {val_loss:.8f} Dice Score: {dice_score:4f}")

        with open(log_file_path, mode="a", newline="") as file:
            writer = csv.writer(file, delimiter=";")
            writer.writerow(
                [
                    epoch + 1,
                    f"{avg_epoch_loss:.8f}",
                    f"{val_loss:.8f}",
                    f"{dice_score:.4f}",
                ]
            )

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


if __name__ == "__main__":
    num_epochs = 200
    batch_size = 2
    learning_rate = 1e-4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_path = "trained_models/disc_herniation_detection_model_uran4.pth"
    patience = 5

    model = VNetModel().to(device)

    if os.path.exists(save_path):
        print(f"Loading model from {save_path}...")
        model.load_state_dict(torch.load(save_path, map_location=device))
        print("Model loaded.")
        test_dataset = Dataset("test", "test_annotations.json")
        test_dataloader = DataLoader(test_dataset, 1, shuffle=False)
        print(evaluate_model(model, test_dataloader, nn.MSELoss(), device))
    else:
        print(f"No model found at {save_path}. Training from scratch...")

        dataset = Dataset(
            mri_dir="train", annotation_file="train_annotations.json", transform=True
        )

        val_size = int(0.1 * len(dataset))
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )

        train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
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
