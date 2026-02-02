import os
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim.lr_scheduler as lr_scheduler
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
import glob
import random
import math

# --- FIX FOR OMP ERROR ---
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# ==========================================
# 1. UNIVERSAL DATASET CLASS
# ==========================================
class UniversalDataset(Dataset):
    def __init__(self, data_pairs, target_list, transform=None, img_size=(512, 512), line_thickness=6):
        self.data_pairs = data_pairs
        self.target_list = target_list
        self.transform = transform
        self.img_size = img_size
        self.line_thickness = line_thickness

    def __len__(self):
        return len(self.data_pairs)

    def _create_mask_from_json(self, json_path, img_shape):
        # multi_channel_mask: (C, H, W)
        multi_channel_mask = np.zeros((len(self.target_list), img_shape[0], img_shape[1]), dtype=np.uint8)

        try:
            with open(json_path, "r") as f:
                data = json.load(f)

            for shape in data.get("shapes", []):
                label = shape.get("label", "").strip().lower()
                sType = shape.get("shape_type", "").lower()
                points = np.array(shape.get("points", []), dtype=np.float32)

                for i, target_info in enumerate(self.target_list):
                    target_label = target_info["label"].strip().lower()
                    target_shape_type = target_info["shape"].strip().lower()

                    # exact match preferred; keep `in` for flexibility
                    if target_label == label or target_label in label:
                        current_mask = np.zeros(img_shape, dtype=np.uint8)

                        if target_shape_type == "polygon" and sType == "polygon":
                            if len(points) >= 3:
                                pts = points.astype(np.int32).reshape((-1, 1, 2))
                                cv2.fillPoly(current_mask, [pts], color=1)

                        elif target_shape_type == "rectangle" and sType == "rectangle":
                            if len(points) >= 2:
                                pt1 = tuple(points[0].astype(int))
                                pt2 = tuple(points[1].astype(int))
                                cv2.rectangle(current_mask, pt1, pt2, color=1, thickness=-1)

                        elif target_shape_type == "circle" and sType == "circle":
                            if len(points) >= 2:
                                center = tuple(points[0].astype(int))
                                edge = points[1]
                                radius = int(math.sqrt((edge[0] - center[0]) ** 2 + (edge[1] - center[1]) ** 2))
                                cv2.circle(current_mask, center, radius, color=1, thickness=-1)

                        multi_channel_mask[i, :, :] = np.maximum(multi_channel_mask[i, :, :], current_mask)

        except Exception as e:
            print(f"Dataset Warning: Error processing {json_path}: {e}")

        return multi_channel_mask

    def __getitem__(self, idx):
        img_path, json_path = self.data_pairs[idx]
        image = cv2.imread(img_path)

        if image is None:
            raise RuntimeError(f"Failed to load image at {img_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, _ = image.shape

        mask = self._create_mask_from_json(json_path, (h, w))  # (C, H, W)

        if self.transform:
            augmented = self.transform(image=image, mask=mask.transpose(1, 2, 0))  # (H, W, C)
            image = augmented["image"]
            mask = augmented["mask"].permute(2, 0, 1)  # (C, H, W)

        return image, mask.float()


# ==========================================
# 2. AUGMENTATIONS
# ==========================================
def get_transforms(phase, img_size):
    if phase == "train":
        return A.Compose([
            A.Resize(img_size[0], img_size[1]),
            A.CLAHE(p=0.5, clip_limit=2.0, tile_grid_size=(8, 8)),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Affine(scale=(0.9, 1.1), rotate=(-15, 15), translate_percent=(0.05, 0.05), p=0.7),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.05, p=0.4),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(img_size[0], img_size[1]),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])


# ==========================================
# 3. PER-CLASS IoU + WEIGHTED LOSS (focus limbus)
# ==========================================
def compute_per_class_iou_and_acc(outputs, masks, threshold=0.5):
    probs = torch.sigmoid(outputs)
    preds = (probs > threshold).float()

    C = masks.shape[1]
    iou_list = []
    acc_list = []

    for c in range(C):
        pred_c = preds[:, c, :, :]
        mask_c = masks[:, c, :, :]

        inter = (pred_c * mask_c).sum()
        union = pred_c.sum() + mask_c.sum() - inter
        iou_c = (inter + 1e-7) / (union + 1e-7)
        iou_list.append(iou_c)

        acc_c = (pred_c == mask_c).float().mean()
        acc_list.append(acc_c)

    mean_iou = torch.stack(iou_list).mean()
    mean_acc = torch.stack(acc_list).mean()
    return iou_list, acc_list, mean_iou, mean_acc


def weighted_multilabel_loss(outputs, masks, dice_loss_fn, bce_loss_fn, class_weights):
    bce_raw = torch.nn.functional.binary_cross_entropy_with_logits(outputs, masks, reduction="none")  # (B,C,H,W)
    w = torch.tensor(class_weights, device=outputs.device).view(1, -1, 1, 1)  # (1,C,1,1)
    bce = (bce_raw * w).mean()

    probs = torch.sigmoid(outputs)
    eps = 1e-7
    dice_per_c = []
    for c in range(masks.shape[1]):
        p = probs[:, c, :, :]
        t = masks[:, c, :, :]
        inter = (p * t).sum()
        den = p.sum() + t.sum()
        dice = (2 * inter + eps) / (den + eps)
        dice_per_c.append(1 - dice)  # loss

    dice_per_c = torch.stack(dice_per_c)  # (C,)
    dice = (dice_per_c * torch.tensor(class_weights, device=outputs.device)).mean()

    return dice + 0.5 * bce


# ==========================================
# 4. MAIN TRAINING ROUTINE
# ==========================================
def train_model(config, progress_callback=None):
    data_path = config["data_path"]
    img_size = config["img_size"]
    target_list = config["target_list"]
    assert len(target_list) == 2, "This code expects 2 classes: crop + limbus"

    labels = [t["label"].strip().lower() for t in target_list]
    if "limbus" not in labels:
        raise ValueError("target_list must include label 'limbus'")
    limbus_idx = labels.index("limbus")
    crop_idx = 1 - limbus_idx

    class_weights = [1.0, 1.0]
    class_weights[crop_idx] = config.get("crop_weight", 0.5)
    class_weights[limbus_idx] = config.get("limbus_weight", 1.5)

    if progress_callback:
        progress_callback("Scanning dataset for Image-JSON pairs...", 0, 0, 0, 0)

    img_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif"]
    image_paths = []
    for ext in img_extensions:
        image_paths.extend(glob.glob(os.path.join(data_path, "**", ext), recursive=True))
    image_paths = sorted(list(set(image_paths)))

    full_dataset_pairs = []
    for img_path in image_paths:
        base = os.path.splitext(img_path)[0]
        json_path = base + ".json"
        if os.path.exists(json_path):
            full_dataset_pairs.append((img_path, json_path))

    if not full_dataset_pairs:
        return {"status": "error", "message": f"No Image-JSON pairs found in: {data_path}"}

    random.seed(42)
    random.shuffle(full_dataset_pairs)
    train_size = int(0.8 * len(full_dataset_pairs))
    train_pairs = full_dataset_pairs[:train_size]
    val_pairs = full_dataset_pairs[train_size:]

    if train_size == 0:
        return {"status": "error", "message": "Dataset too small after pairing."}

    train_transform = get_transforms("train", img_size)
    val_transform = get_transforms("val", img_size)

    train_dataset = UniversalDataset(train_pairs, target_list, transform=train_transform, img_size=img_size)
    val_dataset = UniversalDataset(val_pairs, target_list, transform=val_transform, img_size=img_size)

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=0, pin_memory=True)

    encoder_name = config.get("encoder_name", "timm-efficientnet-b0")
    model = smp.UnetPlusPlus(
        encoder_name=encoder_name,
        encoder_weights="imagenet",
        in_channels=3,
        classes=2,
        activation=None,
    )

    device = config.get("device", "cpu")
    model.to(device)

    criterion_dice = smp.losses.DiceLoss(mode="multilabel", from_logits=True)
    criterion_bce = torch.nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=8, verbose=True)

    epochs = config["epochs"]
    best_limbus_iou = 0.0

    print("\n================ TRAINING START ================")
    print(f"Data: {data_path}")
    print(f"Targets: {target_list} | crop_idx={crop_idx} limbus_idx={limbus_idx}")
    print(f"Class weights (crop, limbus): {class_weights}")
    print(f"Train pairs: {len(train_pairs)} | Val pairs: {len(val_pairs)}")
    print(f"Encoder: {encoder_name} | Device: {device}")
    print("================================================\n")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = weighted_multilabel_loss(
                outputs=outputs,
                masks=masks,
                dice_loss_fn=criterion_dice,
                bce_loss_fn=criterion_bce,
                class_weights=class_weights
            )
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        avg_train_loss = train_loss / max(1, len(train_loader))

        model.eval()
        val_mean_iou = 0.0
        val_limbus_iou = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                outputs = model(images)
                iou_list, _, mean_iou, _ = compute_per_class_iou_and_acc(outputs, masks, threshold=0.5)
                val_mean_iou += mean_iou.item()
                val_limbus_iou += iou_list[limbus_idx].item()
        
        n = max(1, len(val_loader))
        val_mean_iou /= n
        val_limbus_iou /= n

        scheduler.step(val_limbus_iou)
        msg = f"Epoch {epoch+1}/{epochs} | Loss: {avg_train_loss:.4f} | mIoU: {val_mean_iou:.4f} | limbus IoU: {val_limbus_iou:.4f}"
        print(msg)

        if progress_callback:
            progress_callback(msg, (epoch + 1) / epochs, val_limbus_iou, avg_train_loss, val_mean_iou)

        if val_limbus_iou > best_limbus_iou:
            best_limbus_iou = val_limbus_iou
            checkpoint = {"state_dict": model.state_dict(), "config": config}
            torch.save(checkpoint, config["model_name"])

    return {"status": "success", "best_limbus_iou": best_limbus_iou, "message": "Training complete."}


if __name__ == "__main__":
    config = {
        "data_path": r"C:\Users\satyam.tripathi\Downloads\Limbus_Data\Train",
        "img_size": (512, 512),
        "batch_size": 4,
        "epochs": 60,
        "learning_rate": 1e-4,
        "encoder_name": "timm-efficientnet-b0",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "model_name": "model_limbus_crop_unetpp_weighted.pth",
        "target_list": [
            {"label": "crop", "shape": "rectangle"},
            {"label": "limbus", "shape": "polygon"},
        ],
        "crop_weight": 1,
        "limbus_weight": 2.5,
    }
    result = train_model(config)
    print("\nResult:", result)
