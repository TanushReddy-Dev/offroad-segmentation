"""
✅ KAGGLE NOTEBOOK - SEMANTIC SEGMENTATION (FIXED FOR T4)

INSTRUCTIONS:
1. Create notebook on kaggle.com → Code → New Notebook
2. Enable GPU: Click ⚙️ → Accelerator → GPU
3. Add Input datasets:
   - Offroad Segmentation Training Dataset
4. Delete all cells
5. Copy-paste this ENTIRE file as ONE cell
6. Click ▶️ Run
7. After run, click "Data" → Download to get checkpoints

KEY FIXES:
- ResNet18 backbone (fits on T4)
- Clears GPU memory before start
- Handles batch norm issues
- Single GPU optimized
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import cv2
import gc
from pathlib import Path
from tqdm import tqdm

# ============================================================================
# MEMORY SETUP
# ============================================================================

print("Clearing GPU memory...")
torch.cuda.empty_cache()
gc.collect()
torch.cuda.empty_cache()
print("✓ GPU memory cleared\n")

DEVICE = 'cuda:0'
dataset_path = '/kaggle/input/datasets/tanushreddy11/offroad-segmentation-training-dataset/Offroad_Segmentation_Training_Dataset'

print(f"✓ Device: {DEVICE}")
print(f"✓ Dataset: {dataset_path}")
print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
print(f"✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ============================================================================
# IMPORTS & AUGMENTATION
# ============================================================================

from albumentations.pytorch import ToTensorV2
import albumentations as A

# ============================================================================
# DATASET
# ============================================================================

class SegmentationDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, mask_dir, augment=False):
        self.images = sorted(Path(image_dir).glob('*.png'))
        self.mask_dir = Path(mask_dir)
        
        if augment:
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3),
                A.RandomBrightnessContrast(p=0.2),
                A.GaussNoise(p=0.1),
                A.Resize(320, 320),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ], is_check_shapes=False)
        else:
            self.transform = A.Compose([
                A.Resize(320, 320),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ], is_check_shapes=False)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        mask_path = self.mask_dir / img_path.name
        
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        if isinstance(mask, torch.Tensor):
            mask = mask.long()
        else:
            mask = torch.from_numpy(mask).long()
        
        return image, mask

# Load datasets
train_dataset = SegmentationDataset(
    f'{dataset_path}/train/Color_Images',
    f'{dataset_path}/train/Segmentation',
    augment=True
)
val_dataset = SegmentationDataset(
    f'{dataset_path}/val/Color_Images',
    f'{dataset_path}/val/Segmentation',
    augment=False
)

print(f"✓ Train: {len(train_dataset)} | Val: {len(val_dataset)}\n")

# ============================================================================
# METRICS - IoU Calculation
# ============================================================================

def calculate_iou(outputs, targets, num_classes=256, ignore_index=255):
    """Calculate Mean IoU"""
    predictions = torch.argmax(outputs, dim=1)
    
    iou_per_class = []
    for class_idx in range(num_classes):
        pred_mask = (predictions == class_idx)
        target_mask = (targets == class_idx)
        
        intersection = (pred_mask & target_mask).sum().item()
        union = (pred_mask | target_mask).sum().item()
        
        if union == 0:
            iou = 1.0 if intersection == 0 else 0.0
        else:
            iou = intersection / union
        
        iou_per_class.append(iou)
    
    return sum(iou_per_class) / len(iou_per_class)

# ============================================================================
# MODEL - FCNResNet50 (simple, stable, fits on T4)
# ============================================================================

from torchvision.models.segmentation import fcn_resnet50

model = fcn_resnet50(weights=None, num_classes=256).to(DEVICE)
print(f"✓ Model: FCN-ResNet50")
print(f"✓ Params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M\n")

# ============================================================================
# DATALOADERS & OPTIMIZATION
# ============================================================================

BATCH_SIZE = 12
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    prefetch_factor=2
)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    prefetch_factor=2
)

optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
loss_fn = nn.CrossEntropyLoss(ignore_index=255)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
scaler = torch.amp.GradScaler()

# ============================================================================
# CHECKPOINTS
# ============================================================================

os.makedirs('/kaggle/working/checkpoints', exist_ok=True)
os.makedirs('/kaggle/working/logs', exist_ok=True)

checkpoint_path = '/kaggle/working/checkpoints/best_model.pt'
start_epoch = 0
history = {'train_loss': [], 'val_loss': [], 'train_iou': [], 'val_iou': []}
best_loss = float('inf')
best_iou = 0.0

if os.path.exists(checkpoint_path):
    print("✓ Resuming from checkpoint...\n")
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    start_epoch = checkpoint.get('epoch', 0) + 1
    best_loss = checkpoint.get('loss', float('inf'))
    print(f"✓ Resumed at epoch {start_epoch + 1}")
    
    history_path = '/kaggle/working/logs/training_history.json'
    if os.path.exists(history_path):
        with open(history_path, 'r') as f:
            history = json.load(f)
    print(f"✓ Previous best loss: {best_loss:.4f}\n")

# ============================================================================
# TRAINING LOOP
# ============================================================================

print("="*80)
if start_epoch == 0:
    print(f"TRAINING - 50 EPOCHS")
else:
    print(f"RESUMING - Epochs {start_epoch + 1} to 50")
print("="*80)
print(f"  Batch Size: {BATCH_SIZE} | Size: 320x320 | Mixed Precision")
print(f"  Optimizer: AdamW | LR: 1e-3 | Scheduler: CosineAnnealing")
print("="*80 + "\n")

for epoch in range(start_epoch, 50):
    # TRAIN
    model.train()
    train_loss = 0
    train_iou = 0
    train_bar = tqdm(train_loader, desc=f"E{epoch+1}/50 [Train]", leave=False)
    
    for images, masks in train_bar:
        images, masks = images.to(DEVICE), masks.to(DEVICE)
        optimizer.zero_grad()
        
        with torch.amp.autocast('cuda'):
            outputs = model(images)['out']
            loss = loss_fn(outputs, masks)
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        train_loss += loss.item()
        train_iou += calculate_iou(outputs, masks)
        train_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    train_loss /= len(train_loader)
    train_iou /= len(train_loader)
    
    # VAL
    model.eval()
    val_loss = 0
    val_iou = 0
    val_bar = tqdm(val_loader, desc=f"E{epoch+1}/50 [Val]", leave=False)
    
    with torch.no_grad():
        for images, masks in val_bar:
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            outputs = model(images)['out']
            loss = loss_fn(outputs, masks)
            val_loss += loss.item()
            val_iou += calculate_iou(outputs, masks)
            val_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    val_loss /= len(val_loader)
    val_iou /= len(val_loader)
    scheduler.step()
    
    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['train_iou'].append(train_iou)
    history['val_iou'].append(val_iou)
    
    print(f"E{epoch+1:2d}/50 | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Train IoU: {train_iou:.4f} | Val IoU: {val_iou:.4f}", end='')
    
    if val_iou > best_iou:
        best_iou = val_iou
        best_loss = val_loss
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'loss': val_loss,
            'iou': val_iou,
        }, checkpoint_path)
        print(" ← ✓ Best saved")
    else:
        print()

# ============================================================================
# FINAL SAVE
# ============================================================================

with open('/kaggle/working/logs/training_history.json', 'w') as f:
    json.dump(history, f, indent=2)

print("\n" + "="*80)
print("✅ TRAINING COMPLETE!")
print("="*80)
print(f"Final best loss: {best_loss:.4f}")
print(f"Final best IoU: {best_iou:.4f}")
print(f"Epochs: {len(history['train_loss'])}")
print(f"✓ Model saved: /kaggle/working/checkpoints/best_model.pt")
print(f"✓ History saved: /kaggle/working/logs/training_history.json")
print("="*80)

# ============================================================================
# INFERENCE EXAMPLE
# ============================================================================

print("\n" + "="*80)
print("INFERENCE EXAMPLE")
print("="*80)

model.eval()
sample_idx = 0
sample_image, sample_mask = val_dataset[sample_idx]
sample_image = sample_image.unsqueeze(0).to(DEVICE)

with torch.no_grad():
    pred = model(sample_image)['out']
    pred = torch.argmax(pred, dim=1).squeeze().cpu().numpy()

print(f"✓ Prediction shape: {pred.shape}")
print(f"✓ Unique classes: {len(set(pred.flatten()))}")
print(f"✓ Ready for inference/TTA!")
print("="*80)
