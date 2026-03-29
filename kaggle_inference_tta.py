"""
INFERENCE & TEST TIME AUGMENTATION (TTA)
Run after training to test the model
"""

import os
import torch
import cv2
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm

# ============================================================================
# SETUP
# ============================================================================

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
checkpoint_path = '/kaggle/working/checkpoints/best_model.pt'
dataset_path = '/kaggle/input/datasets/tanushreddy11/offroad-segmentation-training-dataset/Offroad_Segmentation_Training_Dataset'

print(f"✓ Device: {DEVICE}")
print(f"✓ Model: {checkpoint_path}\n")

# ============================================================================
# LOAD MODEL
# ============================================================================

from torchvision.models.segmentation import fcn_resnet50

model = fcn_resnet50(weights=None, num_classes=256).to(DEVICE)
checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"✓ Model loaded")
print(f"  Epoch: {checkpoint.get('epoch', '?')}")
print(f"  Loss: {checkpoint.get('loss', '?'):.4f}")
print(f"  IoU: {checkpoint.get('iou', '?'):.4f}\n")

# ============================================================================
# INFERENCE FUNCTION
# ============================================================================

def inference(image_path, use_tta=False):
    """Inference with optional TTA"""
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize
    image = cv2.resize(image, (320, 320))
    
    # Normalize
    image = image.astype(np.float32) / 255.0
    image = (image - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    
    # Convert to tensor
    image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    
    if not use_tta:
        with torch.no_grad():
            output = model(image_tensor)['out']
        pred = torch.argmax(output, dim=1).squeeze().cpu().numpy()
        return pred
    
    # Test Time Augmentation (TTA)
    predictions = []
    
    with torch.no_grad():
        # Original
        output = model(image_tensor)['out']
        predictions.append(torch.softmax(output, dim=1))
        
        # Horizontal flip
        flipped = torch.flip(image_tensor, dims=[3])
        output = model(flipped)['out']
        output = torch.flip(output, dims=[3])
        predictions.append(torch.softmax(output, dim=1))
        
        # Vertical flip
        flipped = torch.flip(image_tensor, dims=[2])
        output = model(flipped)['out']
        output = torch.flip(output, dims=[2])
        predictions.append(torch.softmax(output, dim=1))
        
        # Both flips
        flipped = torch.flip(image_tensor, dims=[2, 3])
        output = model(flipped)['out']
        output = torch.flip(output, dims=[2, 3])
        predictions.append(torch.softmax(output, dim=1))
    
    # Average predictions
    avg_pred = torch.mean(torch.stack(predictions), dim=0)
    pred = torch.argmax(avg_pred, dim=1).squeeze().cpu().numpy()
    
    return pred

# ============================================================================
# INFERENCE ON VAL SET
# ============================================================================

print("="*80)
print("INFERENCE ON VALIDATION SET")
print("="*80 + "\n")

val_image_dir = f'{dataset_path}/val/Color_Images'
val_mask_dir = f'{dataset_path}/val/Segmentation'

image_paths = sorted(Path(val_image_dir).glob('*.png'))[:10]  # First 10 for demo

print(f"Testing {len(image_paths)} validation images...\n")

for img_path in tqdm(image_paths):
    mask_path = Path(val_mask_dir) / img_path.name
    
    # Inference
    pred = inference(str(img_path), use_tta=False)
    
    # Load ground truth
    gt = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    gt = cv2.resize(gt, (320, 320))
    
    # Save visualization
    os.makedirs('/kaggle/working/predictions', exist_ok=True)
    
    # Create side-by-side comparison
    pred_colored = cv2.applyColorMap((pred * 255 // 255).astype(np.uint8), cv2.COLORMAP_JET)
    gt_colored = cv2.applyColorMap((gt * 255 // 255).astype(np.uint8), cv2.COLORMAP_JET)
    
    comparison = np.hstack([gt_colored, pred_colored])
    
    output_path = f'/kaggle/working/predictions/{img_path.stem}_comparison.png'
    cv2.imwrite(output_path, comparison)

print(f"✓ Predictions saved to /kaggle/working/predictions/")

# ============================================================================
# TTA TEST
# ============================================================================

print("\n" + "="*80)
print("TEST TIME AUGMENTATION (TTA)")
print("="*80 + "\n")

sample_path = image_paths[0]
print(f"Testing TTA on: {sample_path.name}")

pred_no_tta = inference(str(sample_path), use_tta=False)
pred_tta = inference(str(sample_path), use_tta=True)

print(f"✓ Inference completed")
print(f"  No TTA - Unique classes: {len(set(pred_no_tta.flatten()))}")
print(f"  With TTA - Unique classes: {len(set(pred_tta.flatten()))}")

print("\n" + "="*80)
print("✅ INFERENCE COMPLETE!")
print("="*80)
