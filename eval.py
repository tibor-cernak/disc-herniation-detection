import os
import numpy as np
import torch
import torch.nn as nn
from dataset import Dataset
from visualize import binary_threshold, extract_bounding_boxes
from vnet import VNetModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def aabb_intersects(box1, box2):
    """
    Checks if two axis-aligned bounding boxes (AABBs) intersect.

    Each box is represented as a tuple of (xmin, ymin, zmin, xmax, ymax, zmax).
    """
    x1_min, y1_min, z1_min, x1_max, y1_max, z1_max = box1
    x2_min, y2_min, z2_min, x2_max, y2_max, z2_max = box2

    return (
        x1_min <= x2_max
        and x1_max >= x2_min
        and y1_min <= y2_max
        and y1_max >= y2_min
        and z1_min <= z2_max
        and z1_max >= z2_min
    )


def compute_dice(pred_mask, gt_mask):
    """
    Compute the Dice score between two binary masks (PyTorch tensors).

    Args:
        pred_mask (torch.Tensor): Predicted binary mask (0s and 1s).
        gt_mask (torch.Tensor): Ground truth binary mask (0s and 1s).

    Returns:
        float: Dice score between the two masks.
    """
    pred_mask = (pred_mask > 0).int()
    gt_mask = (gt_mask > 0).int()

    intersection = torch.sum(pred_mask * gt_mask).item()
    pred_sum = torch.sum(pred_mask).item()
    gt_sum = torch.sum(gt_mask).item()

    if pred_sum + gt_sum == 0:
        return 1.0

    dice = (2.0 * intersection) / (pred_sum + gt_sum)
    return dice


def evaluate_model(model, dataloader, criterion, device):
    """Evaluate the model on the set given by dataloader based on criterion."""
    model.eval()
    val_loss = 0.0
    dice_scores = []
    with torch.no_grad():
        for mri_volumes, heatmaps in dataloader:
            mri_volumes = mri_volumes.to(device)
            heatmaps = heatmaps.to(device)

            outputs = model(mri_volumes)

            loss = criterion(outputs, heatmaps)
            val_loss += loss.item()

            for i in range(heatmaps.shape[2]):
                dice = compute_dice(
                    binary_threshold(outputs[:, :, i]),
                    binary_threshold(heatmaps[:, :, i]),
                )
                dice_scores.append(dice)

    return val_loss / len(dataloader), np.mean(dice_scores)


def evaluate_intersections(model, dataloader, threshold=0.25):
    model.eval()
    TP, FP, FN = 0, 0, 0

    total = 0
    total_detected = 0
    with torch.no_grad():
        for mri_volume, heatmap in dataloader:
            mri_volume = mri_volume.to(device)
            true_heatmap = heatmap.to(device).squeeze().cpu().numpy()
            predicted_heatmap = (
                model(mri_volume).squeeze(0).squeeze(0).squeeze(0).cpu().numpy()
            )

            thresholded_truth_heatmap = binary_threshold(true_heatmap, threshold)
            thresholded_pred_heatmap = binary_threshold(predicted_heatmap, threshold)

            true_bboxes = extract_bounding_boxes(thresholded_truth_heatmap)
            pred_bboxes = extract_bounding_boxes(thresholded_pred_heatmap)

            matched = set()
            for pred_bbox in pred_bboxes:
                detected = False
                for i, true_bbox in enumerate(true_bboxes):
                    if aabb_intersects(true_bbox, pred_bbox):
                        TP += 1
                        matched.add(i)
                        detected = True
                        break
                if not detected:
                    FP += 1
            total += len(true_bboxes)
            total_detected += len(matched)
            FN += len(true_bboxes) - len(matched)

    print(f"{total_detected}/{total} = {total_detected/total}")
    # When the model predicts a herniation, how often is it correct?
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    # How many actual herniations did the model successfully detect?
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    print(f"TP: {TP}, FP: {FP}, FN: {FN}")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1_score:.4f}")

    return TP, FP, FN, precision, recall, f1_score


if __name__ == "__main__":
    save_path = "trained_models/disc_herniation_detection_model_uran4.pth"
    model = VNetModel().to(device)
    if os.path.exists(save_path):
        print(f"Loading model from {save_path}...")
        model.load_state_dict(torch.load(save_path, map_location=device))
        print("Model loaded.")
        test_dataset = Dataset("test", "test_annotations.json")
        test_dataloader = torch.utils.data.DataLoader(test_dataset, 1, shuffle=False)
        # print(evaluate_model(model, test_dataloader, nn.MSELoss(), device))
        evaluate_intersections(model, test_dataloader)
    else:
        print(f"No model found at {save_path}.")
