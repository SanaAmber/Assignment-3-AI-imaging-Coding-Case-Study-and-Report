"""Reproducible biomedical image-analysis pipeline for the nuclei dataset.

Run from this directory with: python pipeline.py --epochs 8
Use --ollama to query a local llama3.2-vision model.
"""
from __future__ import annotations

import argparse
import base64
import json
import random
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import torch
from skimage import io
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops_table
from skimage.morphology import binary_closing, binary_opening, disk, remove_small_objects
from skimage.transform import resize
from torch import nn
from torch.utils.data import DataLoader, Dataset

SEED = 20260715
IMAGE_SIZE = 256
OPTIMISED_PROMPT = (
    "You are a descriptive biomedical image assistant. This is a DAPI-like fluorescence "
    "microscopy image of nuclei. Describe only visible image content, never diagnose, "
    "infer patient information, or invent facts. Return JSON only with exactly these keys: "
    "modality, tissue_type, notable_features, image_quality. Use uncertain when evidence "
    "is insufficient. notable_features must be a short array of observable features."
)
NAIVE_PROMPT = "Describe this medical image."
NUMBERS_PROMPT = (
    "You are a biomedical image-analysis assistant. The following information is a "
    "numbers-only summary computed from a segmented microscopy image. Do not infer "
    "anything that is not present in these measurements and do not diagnose. Return "
    "JSON only with exactly these keys: n_objects, density_class, shape_regularity, "
    "quality_flag. Use uncertain when evidence is insufficient.\n\n"
)
DIRECT_KEYS = {"modality", "tissue_type", "notable_features", "image_quality"}
NUMBERS_KEYS = {"n_objects", "density_class", "shape_regularity", "quality_flag"}


def positive_int(value: str) -> int:
    epochs = int(value)
    if epochs < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return epochs


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def read_gray(path: Path) -> np.ndarray:
    image = io.imread(path)
    if image.ndim == 3:
        image = image[..., 2]  # blue DAPI channel
    image = image.astype(np.float32)
    if image.max() > 1:
        image /= 255.0
    if image.shape != (IMAGE_SIZE, IMAGE_SIZE):
        image = resize(image, (IMAGE_SIZE, IMAGE_SIZE), anti_aliasing=True)
    return image.astype(np.float32)


class NucleiDataset(Dataset):
    def __init__(self, root: Path, split: str):
        self.images = sorted((root / split / "images").glob("*.png"))
        self.masks = root / split / "masks"

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        image_path = self.images[index]
        image = read_gray(image_path)
        mask = io.imread(self.masks / image_path.name) > 0
        return (torch.from_numpy(image[None]), torch.from_numpy(mask.astype(np.float32)[None]), image_path.stem)


class SmallUNet(nn.Module):
    """Lightweight U-Net for nuclei segmentation on 256×256 images.
    
    Architecture:
    - Encoder: Two conv blocks with max pooling (halves spatial dims each)
    - Bottleneck: Deepest feature extraction
    - Decoder: Two upsampling blocks with skip connections
    - Output: Binary segmentation map
    """
    def __init__(self) -> None:
        super().__init__()
        # Encoder (downsampling)
        self.enc1 = self.block(1, 16)      # 256x256 → 256x256
        self.enc2 = self.block(16, 32)     # 128x128 → 128x128
        self.bottleneck = self.block(32, 64)  # 64x64 → 64x64
        
        # Pooling
        self.pool = nn.MaxPool2d(2)
        
        # Decoder (upsampling) with corrected output padding for proper dimensions
        self.up2 = nn.ConvTranspose2d(64, 32, 2, 2, padding=0, output_padding=0)  # 64x64 → 128x128
        self.dec2 = self.block(64, 32)     # 64+32 channels after skip connection
        self.up1 = nn.ConvTranspose2d(32, 16, 2, 2, padding=0, output_padding=0)  # 128x128 → 256x256
        self.dec1 = self.block(32, 16)     # 16+16 channels after skip connection
        
        # Output layer
        self.out = nn.Conv2d(16, 1, 1)

    @staticmethod
    def block(input_channels: int, output_channels: int) -> nn.Sequential:
        """Convolutional block: Conv2d + ReLU + Conv2d + ReLU."""
        return nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with skip connections.
        
        Args:
            x: Input tensor (B, 1, 256, 256)
            
        Returns:
            Segmentation logits (B, 1, 256, 256)
        """
        # Encoder
        first = self.enc1(x)                 # (B, 16, 256, 256)
        second = self.enc2(self.pool(first)) # (B, 32, 128, 128)
        bridge = self.bottleneck(self.pool(second))  # (B, 64, 64, 64)
        
        # Decoder with skip connections
        decoded = self.up2(bridge)           # (B, 32, 128, 128)
        decoded = self.dec2(torch.cat([decoded, second], dim=1))  # Concatenate skip
        decoded = self.up1(decoded)          # (B, 16, 256, 256)
        decoded = self.dec1(torch.cat([decoded, first], dim=1))   # Concatenate skip
        
        return self.out(decoded)             # (B, 1, 256, 256)


def dice_score(prediction: torch.Tensor, target: torch.Tensor) -> float:
    prediction = prediction.reshape(-1).float()
    target = target.reshape(-1).float()
    return float((2 * (prediction * target).sum() + 1e-6) / (prediction.sum() + target.sum() + 1e-6))


def iou_score(prediction: torch.Tensor, target: torch.Tensor) -> float:
    intersection = (prediction & target).sum().float()
    union = (prediction | target).sum().float()
    return float((intersection + 1e-6) / (union + 1e-6))


def segmentation_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    predicted_tensor = torch.from_numpy(prediction.astype(bool))
    target_tensor = torch.from_numpy(target.astype(bool))
    return {
        "dice": dice_score(predicted_tensor, target_tensor),
        "iou": iou_score(predicted_tensor, target_tensor),
    }


def otsu_mask(image: np.ndarray) -> np.ndarray:
    mask = image > threshold_otsu(image)
    mask = binary_opening(mask, disk(1))
    mask = binary_closing(mask, disk(2))
    return remove_small_objects(mask, min_size=20)


def features_from_mask(image: np.ndarray, mask: np.ndarray) -> pd.DataFrame:
    cleaned = remove_small_objects(mask.astype(bool), min_size=20)
    objects = label(cleaned)
    table = regionprops_table(
        objects, intensity_image=image,
        properties=("label", "area", "eccentricity", "solidity", "mean_intensity", "centroid"),
    )
    return pd.DataFrame(table)


def feature_summary(features: pd.DataFrame, image_id: str) -> dict[str, Any]:
    if features.empty:
        return {"image_id": image_id, "n_objects": 0, "mean_area": 0.0,
        "density_class": "sparse", "shape_regularity": "uncertain",
        "quality_flag": "low_signal"}
    count = len(features)
    density = "sparse" if count < 15 else "dense" if count > 45 else "moderate"
    regularity = "regular" if features.solidity.mean() >= 0.75 else "irregular"
    return {"image_id": image_id, "n_objects": count,
            "mean_area": round(float(features.area.mean()), 3),
        "density_class": density, "shape_regularity": regularity,
            "quality_flag": "review" if features.solidity.mean() < 0.75 else "acceptable"}


def numbers_narrative(summary: dict[str, Any]) -> str:
    return (f"The segmented image contains {summary['n_objects']} connected objects with "
            f"mean area {summary['mean_area']} pixels. The measured density is "
            f"{summary['density_class']}, the average shape is "
            f"{summary['shape_regularity']}, and the automated quality flag is "
            f"{summary['quality_flag']}. These are image-derived measurements, not a diagnosis.")


def numbers_description(summary: dict[str, Any], use_ollama: bool, model: str) -> dict[str, Any]:
    """Ask the LLM to interpret measurements without providing the image."""
    measurements = json.dumps({
        key: summary[key]
        for key in ("n_objects", "mean_area", "density_class", "shape_regularity", "quality_flag")
    })
    text = ollama_chat(NUMBERS_PROMPT + measurements, None, model) if use_ollama else None
    parsed = parse_json(text or "")
    if parsed is None or not NUMBERS_KEYS.issubset(parsed):
        parsed = {
            "n_objects": summary["n_objects"],
            "density_class": summary["density_class"],
            "shape_regularity": summary["shape_regularity"],
            "quality_flag": summary["quality_flag"],
            "source": "fallback",
        }
    else:
        parsed["source"] = "ollama"
    return parsed


def parse_json(text: str) -> dict[str, Any] | None:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                return None
    return None


def ollama_chat(
    prompt: str,
    image_path: Path | None,
    model: str,
    temperature: float = 0.0,
) -> str | None:
    payload: dict[str, Any] = {"model": model, "prompt": prompt, "stream": False,
                               "options": {"temperature": temperature}}
    if image_path is not None:
        payload["images"] = [base64.b64encode(image_path.read_bytes()).decode("ascii")]
    try:
        response = requests.post("http://localhost:11434/api/generate", json=payload, timeout=180)
        response.raise_for_status()
        return response.json().get("response", "")
    except (requests.RequestException, ValueError) as error:
        return None


def direct_descriptions(image_path: Path, output_dir: Path, use_ollama: bool, model: str) -> None:
    """Task 1: Generate direct VLM descriptions using optimised and naive prompts.
    
    The assignment compares prompt engineering against a naive baseline by sending a
    sample image to the model three times with a mixed prompt schedule:
    - optimised prompt: structured JSON response with constrained biomedical wording
    - naive prompt: simple baseline prompt for comparison
    
    Args:
        image_path: Path to sample image for VLM description
        output_dir: Directory to save results
        use_ollama: Whether to use actual Ollama (True) or fallback (False)
        model: Ollama model name (e.g., "llama3.2-vision")
    """
    records = {
        "image_id": image_path.stem,
        "naive_prompt": NAIVE_PROMPT,
        "optimised_prompt": OPTIMISED_PROMPT,
        "runs": [],
    }
    prompt_schedule = [
        ("optimised", OPTIMISED_PROMPT),
        ("naive", NAIVE_PROMPT),
        ("optimised", OPTIMISED_PROMPT),
    ]

    for prompt_type, prompt in prompt_schedule:
        text = ollama_chat(prompt, image_path, model, temperature=0.7) if use_ollama else None
        parsed = parse_json(text or "")

        if parsed is None or not DIRECT_KEYS.issubset(parsed):
            parsed = {
                "modality": "fluorescence microscopy",
                "tissue_type": "uncertain",
                "notable_features": ["bright blue objects on dark background"],
                "image_quality": "uncertain",
                "source": "fallback",
            }
        else:
            parsed["source"] = "ollama"

        parsed["prompt_type"] = prompt_type
        records["runs"].append(parsed)

    output_path = output_dir / "direct_vlm_example.json"
    output_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"  → Saved {len(records['runs'])} VLM descriptions to {output_path.name}")


def train_model(root: Path, epochs: int, output_dir: Path) -> tuple[SmallUNet, list[dict[str, float]]]:
    """Train U-Net on the nuclei dataset.
    
    Args:
        root: Path to dataset root (containing train/ and val/ directories)
        epochs: Number of training epochs
        output_dir: Directory to save model weights and training history
        
    Returns:
        Tuple of (trained model, training history list with epoch metrics)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    
    # Create data loaders for training and validation splits (batch_size=2 for memory efficiency)
    train_loader = DataLoader(NucleiDataset(root, "train"), batch_size=2, shuffle=True, num_workers=0)
    validation_loader = DataLoader(NucleiDataset(root, "val"), batch_size=2, shuffle=False, num_workers=0)
    print(f"[INFO] Loaded {len(train_loader)} training batches, {len(validation_loader)} validation batches")
    
    # Initialize model, optimizer, and loss function
    model = SmallUNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_function = nn.BCEWithLogitsLoss()
    history: list[dict[str, float]] = []
    best_dice = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    
    # Train for specified number of epochs
    for epoch in range(epochs):
        # Training phase
        model.train()
        losses = []
        for images, masks, _ in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            loss = loss_function(model(images), masks)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        
        # Validation phase
        model.eval()
        val_dice, val_iou = [], []
        with torch.no_grad():
            for images, masks, _ in validation_loader:
                images, masks = images.to(device), masks.to(device)
                prediction = torch.sigmoid(model(images)) > 0.5
                val_dice.append(dice_score(prediction, masks > 0.5))
                val_iou.append(iou_score(prediction, masks > 0.5))
        
        # Record and print epoch metrics
        epoch_record = {"epoch": epoch + 1, "loss": float(np.mean(losses)),
                        "val_dice": float(np.mean(val_dice)), "val_iou": float(np.mean(val_iou))}
        history.append(epoch_record)
        if epoch_record["val_dice"] > best_dice:
            best_dice = epoch_record["val_dice"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        print(f"[Epoch {epoch + 1}/{epochs}] {epoch_record}")
    
    # Save trained model and training history
    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), output_dir / "unet.pt")
    pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)
    print(f"[INFO] Model saved to {output_dir / 'unet.pt'}")
    return model, history


def save_figures(root: Path, model: SmallUNet, history: list[dict[str, float]], output_dir: Path) -> None:
    """Task 3: Create and save validation visualizations.
    
    Generates three publication-quality figures:
    1. Validation panel: Input image, ground-truth mask, and U-Net prediction side-by-side
    2. Training curves: BCE loss and validation Dice over epochs
    3. EDA: Sample image intensity histogram (exploratory data analysis)
    
    Args:
        root: Path to dataset root
        model: Trained U-Net model (must be on the correct device)
        history: Training history from train_model()
        output_dir: Directory to save PNG figures
    """
    figure, axes = plt.subplots(3, 3, figsize=(11, 11), squeeze=False)
    figure.suptitle("Validation examples: image, reference mask, and prediction", fontsize=16)
    sample = NucleiDataset(root, "val")
    device = next(model.parameters()).device
    model.eval()
    
    # Validation panel: each row contains input, ground truth, and prediction.
    for row in range(min(3, len(sample))):
        image, mask, image_id = sample[row]
        with torch.no_grad():
            prediction = (torch.sigmoid(model(image[None].to(device)))[0, 0].cpu().numpy() > 0.5)
        for column, (shown, title) in enumerate(zip(
            (image[0], mask[0], prediction),
            (f"Input {image_id}", "Ground truth", "U-Net prediction"),
        )):
            axes[row, column].imshow(shown, cmap="gray", vmin=0, vmax=1)
            if row == 0:
                axes[row, column].set_title(title, fontsize=12, pad=8)
            axes[row, column].axis("off")
        axes[row, 0].set_ylabel(image_id, fontsize=11, labelpad=12)
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(output_dir / "validation_panel.png", dpi=180)
    plt.close(figure)
    print(f"  -> Saved validation panel to {output_dir / 'validation_panel.png'}")

    # Training curves: use separate panels because loss and Dice have different scales.
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
    frame = pd.DataFrame(history)
    axes[0].plot(frame.epoch, frame.loss, color="tab:blue", marker="o", linewidth=2)
    axes[0].set(xlabel="Epoch", ylabel="BCE loss", title="Optimisation loss")
    axes[1].plot(frame.epoch, frame.val_dice, color="tab:orange", marker="s", linewidth=2)
    axes[1].set(xlabel="Epoch", ylabel="Dice score", title="Validation performance", ylim=(0, 1.05))
    for axis in axes:
        axis.set_xticks(frame.epoch)
        axis.grid(True, alpha=0.3)
    figure.suptitle("U-Net training history", fontsize=15)
    figure.tight_layout()
    figure.savefig(output_dir / "training_curves.png", dpi=180)
    plt.close(figure)
    print(f"  -> Saved training curves to {output_dir / 'training_curves.png'}")

    # EDA: sample image and intensity histogram
    image = read_gray(sorted((root / "train" / "images").glob("*.png"))[0])
    figure, axes = plt.subplots(1, 2, figsize=(8, 3))
    axes[0].imshow(image, cmap="gray")
    axes[0].axis("off")
    axes[0].set_title("Sample image")
    threshold = threshold_otsu(image)
    axes[1].hist(image.ravel(), bins=32, color="steelblue", edgecolor="black")
    axes[1].axvline(threshold, color="crimson", linestyle="--", linewidth=2,
                    label=f"Otsu threshold = {threshold:.3f}")
    axes[1].set_title("Intensity histogram")
    axes[1].set_xlabel("Pixel intensity")
    axes[1].set_ylabel("Frequency")
    axes[1].legend(frameon=False)
    figure.suptitle("Exploratory data analysis: train_000", fontsize=14)
    figure.tight_layout()
    figure.savefig(output_dir / "eda.png", dpi=180)
    plt.close(figure)
    print(f"  -> Saved EDA to {output_dir / 'eda.png'}")

    # Classical baseline: Otsu mask versus the validation ground truth.
    baseline_dice, baseline_iou = [], []
    for index in range(len(sample)):
        image, mask, _ = sample[index]
        otsu_prediction = torch.from_numpy(otsu_mask(image[0].numpy()))
        baseline_dice.append(dice_score(otsu_prediction, mask[0] > 0.5))
        baseline_iou.append(iou_score(otsu_prediction, mask[0] > 0.5))
    best_record = max(history, key=lambda record: record["val_dice"]) if history else None
    metrics = {
        "unet_best_val_epoch": best_record["epoch"] if best_record else None,
        "unet_best_val_dice": best_record["val_dice"] if best_record else None,
        "unet_best_val_iou": best_record["val_iou"] if best_record else None,
        "otsu_mean_val_dice": float(np.mean(baseline_dice)),
        "otsu_mean_val_iou": float(np.mean(baseline_iou)),
    }
    (output_dir / "evaluation_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"  → Saved evaluation metrics to evaluation_metrics.json")

    # Compare the learned model with the classical Otsu baseline.
    metric_names = ["Dice", "IoU"]
    unet_scores = [metrics["unet_best_val_dice"], metrics["unet_best_val_iou"]]
    otsu_scores = [metrics["otsu_mean_val_dice"], metrics["otsu_mean_val_iou"]]
    positions = np.arange(len(metric_names))
    figure, axis = plt.subplots(figsize=(6, 4))
    width = 0.36
    axis.bar(positions - width / 2, otsu_scores, width, label="Otsu", color="tab:gray")
    axis.bar(positions + width / 2, unet_scores, width, label="U-Net", color="tab:orange")
    axis.set_xticks(positions, metric_names)
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("Score")
    axis.set_title("Validation segmentation comparison")
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_dir / "metric_comparison.png", dpi=180)
    plt.close(figure)
    print(f"  → Saved metric comparison to {output_dir / 'metric_comparison.png'}")


def run_hybrid(root: Path, model: SmallUNet, output_dir: Path, use_ollama: bool, model_name: str) -> None:
    """Task 4: Run full hybrid pipeline on unseen test images.
    
    For each test image:
    1. Apply trained U-Net to generate segmentation mask
    2. Extract classical image features (area, eccentricity, etc.) with regionprops
    3. Generate structured JSON record with density classification and quality flag
    4. Create natural-language narrative from the JSON (numbers-first description)
    
    Outputs:
    - test_records.csv: Aggregated table of all test records (for analysis)
    - test_records.json: Full JSON records with narratives (for auditing)
    
    Args:
        root: Path to dataset root containing test/images/
        model: Trained U-Net model (must be on the correct device)
        output_dir: Directory to save CSV and JSON outputs
    """
    device = next(model.parameters()).device
    records = []
    test_dice, test_iou = [], []
    model.eval()
    
    test_images = sorted((root / "test" / "images").glob("*.png"))
    print(f"  Processing {len(test_images)} test images...")
    
    for path in test_images:
        # Read image and generate U-Net mask
        image = read_gray(path)
        with torch.no_grad():
            predicted = torch.sigmoid(model(torch.from_numpy(image[None, None]).to(device)))[0, 0].cpu().numpy() > 0.5

        mask_path = root / "test" / "masks" / path.name
        if mask_path.exists():
            ground_truth = io.imread(mask_path) > 0
            metrics = segmentation_metrics(predicted, ground_truth)
            summary_metrics = {f"test_{key}": value for key, value in metrics.items()}
            test_dice.append(metrics["dice"])
            test_iou.append(metrics["iou"])
        else:
            summary_metrics = {}
        
        # Task 2: Extract features from mask and generate summary
        summary = feature_summary(features_from_mask(image, predicted), path.stem)
        summary.update(summary_metrics)
        summary["numbers_llm_record"] = numbers_description(summary, use_ollama, model_name)
        
        # Generate natural-language narrative
        summary["narrative"] = numbers_narrative(summary)
        records.append(summary)
    
    # Save aggregated records as CSV and JSON
    csv_path = output_dir / "test_records.csv"
    json_path = output_dir / "test_records.json"
    pd.DataFrame(records).to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    test_metrics = {
        "test_images": len(records),
        "mean_test_dice": float(np.mean(test_dice)) if test_dice else None,
        "mean_test_iou": float(np.mean(test_iou)) if test_iou else None,
    }
    (output_dir / "test_evaluation_metrics.json").write_text(
        json.dumps(test_metrics, indent=2), encoding="utf-8"
    )
    print(f"  → Saved {len(records)} test records to test_records.csv and test_records.json")
    print(f"  → Saved test metrics to {output_dir / 'test_evaluation_metrics.json'}")


def run_classical_example(root: Path, output_dir: Path, use_ollama: bool, model: str) -> None:
    """Save the Otsu feature table and numbers-first interpretation for one image."""
    image_path = sorted((root / "train" / "images").glob("*.png"))[0]
    image = read_gray(image_path)
    mask = otsu_mask(image)
    features = features_from_mask(image, mask)
    summary = feature_summary(features, image_path.stem)
    summary["narrative"] = numbers_narrative(summary)
    summary["llm_record"] = numbers_description(summary, use_ollama, model)
    features.to_csv(output_dir / "classical_features_example.csv", index=False)
    (output_dir / "numbers_first_example.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print("  → Saved classical_features_example.csv and numbers_first_example.json")


def run_robustness(root: Path, model: SmallUNet, output_dir: Path) -> None:
    """Trace segmentation and measurements on the supplied corrupted images."""
    device = next(model.parameters()).device
    records = []
    model.eval()
    for path in sorted((root / "test_corrupted" / "images").glob("*.png")):
        image = read_gray(path)
        with torch.no_grad():
            predicted = torch.sigmoid(
                model(torch.from_numpy(image[None, None]).to(device))
            )[0, 0].cpu().numpy() > 0.5
        summary = feature_summary(features_from_mask(image, predicted), path.stem)
        summary["narrative"] = numbers_narrative(summary)
        records.append(summary)
    (output_dir / "robustness_records.json").write_text(
        json.dumps(records, indent=2), encoding="utf-8"
    )
    if records:
        frame = pd.DataFrame(records).sort_values("image_id")
        figure, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].bar(frame["image_id"], frame["n_objects"], color="tab:blue")
        axes[0].set_title("Detected objects after corruption")
        axes[0].set_ylabel("Connected objects")
        axes[1].bar(frame["image_id"], frame["mean_area"], color="tab:red")
        axes[1].set_title("Mean detected object area")
        axes[1].set_ylabel("Pixels")
        for axis in axes:
            axis.tick_params(axis="x", rotation=60, labelsize=8)
            axis.grid(axis="y", alpha=0.3)
        figure.suptitle("Robustness trace on corrupted images")
        figure.tight_layout()
        figure.savefig(output_dir / "robustness_summary.png", dpi=180)
        plt.close(figure)
        print(f"  → Saved robustness summary to {output_dir / 'robustness_summary.png'}")
    print(f"  → Saved {len(records)} corrupted-image records to robustness_records.json")


def main() -> None:
    """Main entry point: execute full biomedical image analysis pipeline.
    
    Workflow:
    1. Direct VLM descriptions (Task 1): Compare optimised and naive prompts
    2. Classical features (Task 2): Otsu, regionprops, and numbers-only interpretation
    3. Train and evaluate U-Net (Task 3): Save metrics and required figures
    4. Hybrid pipeline (Task 4): Segment test images and generate auditable records
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path(__file__).parent, 
                        help="Path to dataset root directory (default: script directory)")
    parser.add_argument("--output", type=Path, default=Path("outputs"),
                        help="Directory to save outputs (default: ./outputs)")
    parser.add_argument("--epochs", type=positive_int, default=8,
                        help="Number of training epochs (default: 8)")
    parser.add_argument("--ollama", action="store_true",
                        help="Use local Ollama for VLM descriptions (default: use fallback)")
    parser.add_argument("--model", default="llama3.2-vision",
                        help="Ollama model name for VLM (default: llama3.2-vision)")
    args = parser.parse_args()
    
    # Initialize random seeds for reproducibility
    seed_everything()
    print(f"[INFO] Initialized with seed {SEED}")
    
    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Output directory: {args.output.resolve()}")
    
    # Task 1: Direct VLM descriptions with optimized and naive prompts
    print("\n[TASK 1] Generating direct VLM descriptions...")
    direct_descriptions(sorted((args.data / "train" / "images").glob("*.png"))[0], 
                       args.output, args.ollama, args.model)
    
    # Task 2: Classical segmentation, features, and numbers-first interpretation.
    print("\n[TASK 2] Extracting classical features and numbers-first description...")
    run_classical_example(args.data, args.output, args.ollama, args.model)

    # Task 3: Train U-Net and generate visualizations.
    print("\n[TASK 3] Training U-Net segmentation model...")
    model, history = train_model(args.data, args.epochs, args.output)
    
    print("\n[TASK 3] Saving training visualizations...")
    save_figures(args.data, model, history, args.output)
    
    # Task 4: Run full hybrid pipeline on test set.
    print("\n[TASK 4] Running hybrid pipeline on test set...")
    run_hybrid(args.data, model, args.output, args.ollama, args.model)

    print("\n[EXTENSION] Running corrupted-image robustness trace...")
    run_robustness(args.data, model, args.output)
    
    print(f"\n✓ Completed pipeline. All outputs saved to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
