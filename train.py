"""
train.py - Training Script for Semantic Segmentation

Usage:
    python train.py --config config.yaml
    python train.py --epochs 50 --batch-size 12 --lr 1e-3
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
import argparse
import yaml

import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision.models.segmentation import fcn_resnet50

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    def __init__(self, config_dict=None):
        # Defaults
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.dataset_path = './dataset'
        self.output_dir = './outputs'
        self.batch_size = 12
        self.epochs = 50
        self.lr = 1e-3
        self.weight_decay = 1e-4
        self.image_size = 320
        self.num_classes = 256
        self.num_workers = 4
        
        if config_dict:
            self.__dict__.update(config_dict)
    
    def to_dict(self):
        return self.__dict__

# ============================================================================
# DATASET
# ============================================================================

class SegmentationDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, mask_dir, augment=False, image_size=320):
        self.images = sorted(Path(image_dir).glob('*.png'))
        self.mask_dir = Path(mask_dir)
        self.image_size = image_size
        
        if augment:
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3),
                A.RandomBrightnessContrast(p=0.2),
                A.GaussNoise(p=0.1),
                A.Resize(image_size, image_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ], is_check_shapes=False)
        else:
            self.transform = A.Compose([
                A.Resize(image_size, image_size),
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

# ============================================================================
# METRICS
# ============================================================================

def calculate_iou(outputs, targets, num_classes=256):
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
# TRAINING
# ============================================================================

def train(config):
    print("Clearing GPU memory...")
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.empty_cache()
    
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(f'{config.output_dir}/checkpoints', exist_ok=True)
    os.makedirs(f'{config.output_dir}/logs', exist_ok=True)
    
    print(f"\n✓ Device: {config.device}")
    print(f"✓ Dataset: {config.dataset_path}")
    print(f"✓ Output: {config.output_dir}\n")
    
    # Dataset
    train_dataset = SegmentationDataset(
        f'{config.dataset_path}/train/Color_Images',
        f'{config.dataset_path}/train/Segmentation',
        augment=True,
        image_size=config.image_size
    )
    val_dataset = SegmentationDataset(
        f'{config.dataset_path}/val/Color_Images',
        f'{config.dataset_path}/val/Segmentation',
        augment=False,
        image_size=config.image_size
    )
    
    print(f"✓ Train: {len(train_dataset)} | Val: {len(val_dataset)}\n")
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        prefetch_factor=2
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        prefetch_factor=2
    )
    
    # Model
    model = fcn_resnet50(weights=None, num_classes=config.num_classes).to(config.device)
    print(f"✓ Model: FCN-ResNet50")
    print(f"✓ Params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M\n")
    
    # Optimization
    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    loss_fn = nn.CrossEntropyLoss(ignore_index=255)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    scaler = torch.amp.GradScaler()
    
    # Checkpoint
    checkpoint_path = f'{config.output_dir}/checkpoints/best_model.pt'
    start_epoch = 0
    history = {'train_loss': [], 'val_loss': [], 'train_iou': [], 'val_iou': []}
    best_iou = 0.0
    best_loss = float('inf')
    
    if os.path.exists(checkpoint_path):
        print("✓ Resuming from checkpoint...\n")
        checkpoint = torch.load(checkpoint_path, map_location=config.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_iou = checkpoint.get('iou', 0.0)
        print(f"✓ Resumed at epoch {start_epoch + 1}\n")
    
    # Training
    print("="*80)
    print(f"TRAINING - {config.epochs} EPOCHS")
    print("="*80 + "\n")
    
    for epoch in range(start_epoch, config.epochs):
        # Train
        model.train()
        train_loss = 0
        train_iou = 0
        train_bar = tqdm(train_loader, desc=f"E{epoch+1}/{config.epochs} [Train]", leave=False)
        
        for images, masks in train_bar:
            images, masks = images.to(config.device), masks.to(config.device)
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
        
        # Val
        model.eval()
        val_loss = 0
        val_iou = 0
        val_bar = tqdm(val_loader, desc=f"E{epoch+1}/{config.epochs} [Val]", leave=False)
        
        with torch.no_grad():
            for images, masks in val_bar:
                images, masks = images.to(config.device), masks.to(config.device)
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
        
        print(f"E{epoch+1:2d}/{config.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Train IoU: {train_iou:.4f} | Val IoU: {val_iou:.4f}", end='')
        
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
    
    # Save history
    with open(f'{config.output_dir}/logs/training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print("\n" + "="*80)
    print("✅ TRAINING COMPLETE!")
    print("="*80)
    print(f"Best validation IoU: {best_iou:.4f}")
    print(f"Best validation Loss: {best_loss:.4f}")
    print(f"Epochs: {len(history['train_loss'])}")
    print(f"✓ Model saved: {checkpoint_path}")
    print("="*80)

# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train semantic segmentation model')
    parser.add_argument('--config', type=str, help='Path to config file (YAML)')
    parser.add_argument('--dataset-path', type=str, default='./dataset', help='Dataset path')
    parser.add_argument('--output-dir', type=str, default='./outputs', help='Output directory')
    parser.add_argument('--batch-size', type=int, default=12, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--image-size', type=int, default=320, help='Image size')
    
    args = parser.parse_args()
    
    # Load config
    if args.config:
        with open(args.config, 'r') as f:
            config_dict = yaml.safe_load(f)
        config = Config(config_dict)
    else:
        config = Config({
            'dataset_path': args.dataset_path,
            'output_dir': args.output_dir,
            'batch_size': args.batch_size,
            'epochs': args.epochs,
            'lr': args.lr,
            'image_size': args.image_size,
        })
    
    train(config)
