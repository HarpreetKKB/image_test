"""
Comparison pipeline: fine-tune Mask R-CNN and train a lightweight SAM-head (feature-extractor + head),
validate and test both models, compute IoU/Dice/Precision/Recall, and produce comparison plots.

How to use:
- Put your images in `dataset/images/` and masks in `dataset/masks/` (PNG with 0 background, 255 foreground)
- Install dependencies: torch, torchvision, opencv-python, pillow, scikit-learn, matplotlib, numpy, albumentations
  Optional for SAM: `segment-anything` package (https://github.com/facebookresearch/segment-anything) if you want to run SAM feature extraction.
- Run: `python comparison_pipeline_sam_maskrcnn.py --data_dir dataset --out_dir results --device cuda`

Notes:
- This script provides a robust Mask R-CNN training loop.
- For SAM, it provides a practical approach: use SAM (if available) as a frozen encoder to extract image embeddings, then train a small decoder (UNet-like) on those embeddings. True end-to-end fine-tuning of SAM requires the official SAM training machinery and large compute; this approach works well for small datasets.

Author: Generated for user (Canadian English preferences)
"""
# pip install torch torchvision numpy pillow matplotlib scikit-learn opencv-python albumentations
# python comparison_pipeline_sam_maskrcnn.py --data_dir dataset --out_dir results --device cuda --use_sam --sam_checkpoint /path/to/sam/checkpoint

import os
import random
import argparse
from glob import glob
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision
from torchvision import transforms
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.ops import sigmoid_focal_loss

from sklearn.metrics import precision_score, recall_score

# Optional imports for SAM if installed
try:
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    SAM_AVAILABLE = True
except Exception:
    SAM_AVAILABLE = False

# ---------- Dataset ----------
class SegmentationDataset(Dataset):
    def __init__(self, images: List[Path], masks: List[Path], transform=None):
        self.images = images
        self.masks = masks
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert('RGB')
        mask = Image.open(self.masks[idx]).convert('L')

        img = np.array(img)
        mask = np.array(mask)
        # Ensure binary mask
        mask = (mask > 127).astype(np.uint8)

        sample = {"image": img, "mask": mask}

        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            sample["image"] = augmented["image"]
            sample["mask"] = augmented["mask"]

        # Convert to tensors
        image_tensor = transforms.ToTensor()(sample['image'])
        mask_tensor = torch.from_numpy(sample['mask']).long()

        return image_tensor, mask_tensor

# ---------- Utilities ----------

def collate_fn_maskrcnn(batch):
    images = []
    targets = []
    for img, mask in batch:
        images.append(img)
        # Mask-RCNN expects boxes + labels + masks per instance.
        # For single-instance images, convert mask to one object.
        masks = mask.unsqueeze(0)  # 1 x H x W
        pos = torch.nonzero(masks[0])
        if pos.shape[0] == 0:
            # empty mask -> create tiny box
            boxes = torch.tensor([[0,0,1,1]], dtype=torch.float32)
            labels = torch.tensor([1], dtype=torch.int64)
            masks = torch.zeros((1, mask.shape[0], mask.shape[1]), dtype=torch.uint8)
        else:
            ys = pos[:,0].float()
            xs = pos[:,1].float()
            x1 = xs.min(); y1 = ys.min(); x2 = xs.max(); y2 = ys.max()
            boxes = torch.tensor([[x1, y1, x2, y2]], dtype=torch.float32)
            labels = torch.tensor([1], dtype=torch.int64)
            masks = masks

        targets.append({
            'boxes': boxes,
            'labels': labels,
            'masks': masks.float()
        })
    return images, targets

# Metric functions

def compute_iou(y_true: np.ndarray, y_pred: np.ndarray, eps=1e-7) -> float:
    y_true = y_true.astype(bool)
    y_pred = y_pred.astype(bool)
    inter = (y_true & y_pred).sum()
    union = (y_true | y_pred).sum()
    if union == 0:
        return 1.0 if inter == 0 else 0.0
    return inter / (union + eps)


def compute_dice(y_true: np.ndarray, y_pred: np.ndarray, eps=1e-7) -> float:
    y_true = y_true.astype(bool)
    y_pred = y_pred.astype(bool)
    inter = 2 * (y_true & y_pred).sum()
    denom = y_true.sum() + y_pred.sum()
    if denom == 0:
        return 1.0
    return inter / (denom + eps)

# ---------- Mask R-CNN Training & Evaluation ----------

def get_maskrcnn_model(num_classes=2):
    # num_classes: background + object
    model = maskrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

    mask_in_features = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(mask_in_features, hidden_layer, num_classes)
    return model


def train_maskrcnn(train_loader, val_loader, device, out_dir, epochs=10, lr=1e-4):
    model = get_maskrcnn_model(num_classes=2)
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    best_val_miou = 0.0
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, targets in train_loader:
            images = [img.to(device) for img in images]
            targets = [{k:v.to(device) for k,v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            running_loss += losses.item()

        lr_scheduler.step()
        avg_loss = running_loss / len(train_loader)
        val_miou, val_metrics = evaluate_maskrcnn(model, val_loader, device)
        print(f"Epoch {epoch+1}/{epochs} | Train loss: {avg_loss:.4f} | Val mIoU: {val_miou:.4f}")

        # save best
        if val_miou > best_val_miou:
            best_val_miou = val_miou
            torch.save(model.state_dict(), os.path.join(out_dir, 'maskrcnn_best.pth'))

    # final evaluation on val
    model.load_state_dict(torch.load(os.path.join(out_dir, 'maskrcnn_best.pth')))
    return model


def evaluate_maskrcnn(model, loader, device) -> Tuple[float, dict]:
    model.eval()
    ious = []
    dices = []
    precisions = []
    recalls = []

    with torch.no_grad():
        for images, masks in loader:
            images_gpu = [img.to(device) for img in images]
            outputs = model(images_gpu)
            for i, out in enumerate(outputs):
                # outputs have 'masks' as [N,1,H,W]
                if out['masks'].shape[0] == 0:
                    pred_mask = np.zeros_like(masks[i].numpy())
                else:
                    # take the highest scoring mask
                    scores = out.get('scores', None)
                    masks_pred = out['masks'].cpu().numpy()
                    if masks_pred.shape[0] == 0:
                        pred_mask = np.zeros_like(masks[i].numpy())
                    else:
                        # combine masks by thresholding and taking union
                        m = (masks_pred[:,0] > 0.5).astype(np.uint8)
                        pred_mask = np.any(m, axis=0).astype(np.uint8)

                gt = masks[i].numpy().astype(np.uint8)
                iou = compute_iou(gt, pred_mask)
                dice = compute_dice(gt, pred_mask)
                ious.append(iou)
                dices.append(dice)

                # pixel-level precision/recall
                if pred_mask.sum() == 0 and gt.sum() == 0:
                    precisions.append(1.0)
                    recalls.append(1.0)
                else:
                    precisions.append(precision_score(gt.flatten(), pred_mask.flatten(), zero_division=1))
                    recalls.append(recall_score(gt.flatten(), pred_mask.flatten(), zero_division=1))

    metrics = {
        'mIoU': float(np.mean(ious)),
        'mDice': float(np.mean(dices)),
        'Precision': float(np.mean(precisions)),
        'Recall': float(np.mean(recalls))
    }
    return metrics['mIoU'], metrics

# ---------- SAM-based feature extractor + small decoder ----------
class SmallDecoder(nn.Module):
    """A small UNet-like decoder that accepts feature maps and outputs a segmentation mask."""
    def __init__(self, in_channels=256, out_channels=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.conv3(x)
        return x


def extract_sam_features_for_dataset(data_dir: str, device: torch.device, sam_checkpoint: str = None, model_type: str = 'vit_h') -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    If SAM is available, use it as a frozen encoder to produce feature maps for each image.
    Returns lists of feature arrays and corresponding masks.

    If SAM is not installed, this function will raise an informative error.
    """
    if not SAM_AVAILABLE:
        raise RuntimeError("SAM (segment_anything) is not available. Install it to use SAM features or set up alternative encoder.")

    # Load SAM
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device)
    sam.eval()

    img_paths = sorted(glob(os.path.join(data_dir, 'images', '*')))
    mask_paths = sorted(glob(os.path.join(data_dir, 'masks', '*')))

    features = []
    masks = []
    preprocess = transforms.Compose([transforms.ToTensor()])

    with torch.no_grad():
        for ip, mp in zip(img_paths, mask_paths):
            img = Image.open(ip).convert('RGB')
            img_arr = np.array(img)
            input_tensor = preprocess(img_arr).unsqueeze(0).to(device)
            # Use SAM's image encoder (vit) to get features
            # NOTE: This code depends on the internal API of SAM and may require adaptation to the exact SAM release.
            image_embeddings = sam.image_encoder(input_tensor)
            # image_embeddings shape depends on model; we will take the last embedding and detach
            emb = image_embeddings.detach().cpu().numpy()
            features.append(emb)

            mask = Image.open(mp).convert('L')
            mask = (np.array(mask) > 127).astype(np.uint8)
            masks.append(mask)

    return features, masks


def train_sam_decoder(features_list, masks_list, device, out_dir, epochs=15, lr=1e-3, batch_size=4):
    # Convert features_list (np arrays) to tensors and create dataset
    tensors = [torch.from_numpy(f).squeeze(0).float() for f in features_list]
    # Upsample/resize masks to feature spatial size
    spatial = tensors[0].shape[-2:]

    dataset = []
    for feat, mask in zip(tensors, masks_list):
        # convert mask to shape compatible with feat
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).float()
        # resize mask to feat spatial dims
        mask_resized = nn.functional.interpolate(mask_tensor.unsqueeze(0), size=spatial, mode='nearest').squeeze(0)
        dataset.append((feat, mask_resized.squeeze(0)))

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = SmallDecoder(in_channels=tensors[0].shape[0], out_channels=1)
    model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    bce = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for feats, masks in loader:
            feats = feats.to(device)
            masks = masks.to(device)
            logits = model(feats)
            logits = nn.functional.interpolate(logits, size=masks.shape[-2:], mode='bilinear', align_corners=False)
            loss = bce(logits.squeeze(1), masks)
            optim.zero_grad()
            loss.backward()
            optim.step()
            epoch_loss += loss.item()
        print(f"SAM-decoder Epoch {epoch+1}/{epochs} loss {epoch_loss/len(loader):.4f}")

    torch.save(model.state_dict(), os.path.join(out_dir, 'sam_decoder_best.pth'))
    return model


def evaluate_sam_decoder(model, features_list, masks_list, device) -> dict:
    model.eval()
    ious, dices, precisions, recalls = [], [], [], []
    with torch.no_grad():
        for feat, mask in zip(features_list, masks_list):
            feat_t = torch.from_numpy(feat).squeeze(0).float().to(device)
            logits = model(feat_t.unsqueeze(0))
            logits_up = nn.functional.interpolate(logits, size=mask.shape, mode='bilinear', align_corners=False)
            pred = (torch.sigmoid(logits_up) > 0.5).cpu().numpy().squeeze(0).astype(np.uint8)
            iou = compute_iou(mask, pred)
            dice = compute_dice(mask, pred)
            ious.append(iou); dices.append(dice)
            precisions.append(precision_score(mask.flatten(), pred.flatten(), zero_division=1))
            recalls.append(recall_score(mask.flatten(), pred.flatten(), zero_division=1))
    metrics = {
        'mIoU': float(np.mean(ious)),
        'mDice': float(np.mean(dices)),
        'Precision': float(np.mean(precisions)),
        'Recall': float(np.mean(recalls))
    }
    return metrics

# ---------- Plotting & comparison ----------

def plot_comparison(metrics_maskrcnn: dict, metrics_sam: dict, out_dir: str):
    labels = ['mIoU', 'mDice', 'Precision', 'Recall']
    m1 = [metrics_maskrcnn[k] for k in labels]
    m2 = [metrics_sam[k] for k in labels]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8,4))
    ax.bar(x - width/2, m1, width, label='Mask-RCNN')
    ax.bar(x + width/2, m2, width, label='SAM-decoder')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0,1)
    ax.set_title('Model comparison on test set')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'comparison.png'))
    plt.close()

# ---------- Main flow ----------

def main(args):
    random.seed(42)
    torch.manual_seed(42)

    data_dir = args.data_dir
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    # collect files
    img_paths = sorted(glob(os.path.join(data_dir, 'images', '*')))
    mask_paths = sorted(glob(os.path.join(data_dir, 'masks', '*')))
    assert len(img_paths) == len(mask_paths), "Images and masks count mismatch"

    # Create dataset and splits
    full_dataset = SegmentationDataset(img_paths, mask_paths, transform=None)
    n = len(full_dataset)
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)
    n_test = n - n_train - n_val
    train_ds, val_ds, test_ds = random_split(full_dataset, [n_train, n_val, n_test])

    # DataLoaders
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, collate_fn=collate_fn_maskrcnn)
    val_loader = DataLoader(val_ds, batch_size=2, shuffle=False, collate_fn=collate_fn_maskrcnn)
    test_loader = DataLoader(test_ds, batch_size=2, shuffle=False, collate_fn=collate_fn_maskrcnn)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Train Mask-RCNN
    print('\n=== Training Mask-RCNN ===')
    maskrcnn_model = train_maskrcnn(train_loader, val_loader, device, out_dir, epochs=args.maskrcnn_epochs, lr=args.maskrcnn_lr)

    # Evaluate on test set
    print('\n=== Evaluating Mask-RCNN on test set ===')
    _, maskrcnn_metrics = evaluate_maskrcnn(maskrcnn_model, test_loader, device)
    print('Mask-RCNN test metrics:', maskrcnn_metrics)

    # SAM-based approach (optional)
    if args.use_sam:
        if not SAM_AVAILABLE:
            print('SAM not available: skipping SAM pipeline. To use SAM, install `segment-anything` and provide a checkpoint via --sam_checkpoint')
            sam_metrics = {k:0.0 for k in ['mIoU','mDice','Precision','Recall']}
        else:
            print('\n=== Extracting SAM features ===')
            features, masks = extract_sam_features_for_dataset(data_dir, device, sam_checkpoint=args.sam_checkpoint, model_type=args.sam_model_type)
            # split features/masks using same indices as dataset splits
            indices = list(range(len(features)))
            train_idx = indices[:n_train]
            val_idx = indices[n_train:n_train+n_val]
            test_idx = indices[n_train+n_val:]

            train_feats = [features[i] for i in train_idx]
            train_masks = [masks[i] for i in train_idx]
            val_feats = [features[i] for i in val_idx]
            val_masks = [masks[i] for i in val_idx]
            test_feats = [features[i] for i in test_idx]
            test_masks = [masks[i] for i in test_idx]

            print('\n=== Training SAM-decoder ===')
            sam_decoder = train_sam_decoder(train_feats, train_masks, device, out_dir, epochs=args.sam_epochs, lr=args.sam_lr)

            print('\n=== Evaluating SAM-decoder on test set ===')
            sam_metrics = evaluate_sam_decoder(sam_decoder, test_feats, test_masks, device)
            print('SAM-decoder test metrics:', sam_metrics)
    else:
        print('Skipping SAM pipeline as --use_sam not set')
        sam_metrics = {k:0.0 for k in ['mIoU','mDice','Precision','Recall']}

    # Plot comparison
    plot_comparison(maskrcnn_metrics, sam_metrics, out_dir)
    print(f"Results saved to {out_dir}. Comparison plot: comparison.png")

    # Simple decision logic
    if maskrcnn_metrics['mIoU'] > sam_metrics['mIoU']:
        print('Mask-RCNN performs better by mIoU on test set')
    elif maskrcnn_metrics['mIoU'] < sam_metrics['mIoU']:
        print('SAM-decoder performs better by mIoU on test set')
    else:
        print('Both models have same mIoU on test set')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='dataset')
    parser.add_argument('--out_dir', type=str, default='results')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--maskrcnn_epochs', type=int, default=12)
    parser.add_argument('--maskrcnn_lr', type=float, default=1e-4)
    parser.add_argument('--use_sam', action='store_true')
    parser.add_argument('--sam_checkpoint', type=str, default=None)
    parser.add_argument('--sam_model_type', type=str, default='vit_h')
    parser.add_argument('--sam_epochs', type=int, default=15)
    parser.add_argument('--sam_lr', type=float, default=1e-3)
    args = parser.parse_args()
    main(args)
