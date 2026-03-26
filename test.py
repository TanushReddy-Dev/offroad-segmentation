"""
test.py - Testing/Inference Script for Semantic Segmentation

Usage:
    python test.py --model best_model.pt --image test.png
    python test.py --model best_model.pt --dataset-path ./dataset --split val
"""

import os
import torch
import cv2
import numpy as np
from pathlib import Path
import argparse
import json
from tqdm import tqdm
from torchvision.models.segmentation import fcn_resnet50

# ============================================================================
# CONFIGURATION
# ============================================================================

class TestConfig:
    def __init__(self):
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.model_path = './outputs/checkpoints/best_model.pt'
        self.num_classes = 256
        self.image_size = 320
        self.use_tta = False

# ============================================================================
# METRICS
# ============================================================================

def calculate_iou(pred, target, num_classes=256):
    """Calculate Mean IoU"""
    iou_per_class = []
    for class_idx in range(num_classes):
        pred_mask = (pred == class_idx)
        target_mask = (target == class_idx)
        
        intersection = (pred_mask & target_mask).sum()
        union = (pred_mask | target_mask).sum()
        
        if union == 0:
            iou = 1.0 if intersection == 0 else 0.0
        else:
            iou = intersection.item() / union.item()
        
        iou_per_class.append(iou)
    
    return sum(iou_per_class) / len(iou_per_class)

def calculate_metrics(predictions, targets):
    """Calculate multiple metrics"""
    predictions = predictions.cpu().numpy()
    targets = targets.cpu().numpy()
    
    metrics = {}
    iou_scores = []
    
    for i in range(predictions.shape[0]):
        pred = predictions[i]
        target = targets[i]
        iou = calculate_iou(torch.from_numpy(pred), torch.from_numpy(target))
        iou_scores.append(iou)
    
    metrics['mean_iou'] = np.mean(iou_scores)
    metrics['std_iou'] = np.std(iou_scores)
    metrics['max_iou'] = np.max(iou_scores)
    metrics['min_iou'] = np.min(iou_scores)
    
    return metrics

# ============================================================================
# INFERENCE
# ============================================================================

def load_model(model_path, num_classes=256, device='cuda:0'):
    """Load trained model"""
    model = fcn_resnet50(weights=None, num_classes=num_classes).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✓ Model loaded: {model_path}")
    print(f"  Epoch: {checkpoint.get('epoch', '?')}")
    print(f"  IoU: {checkpoint.get('iou', '?'):.4f}")
    print(f"  Loss: {checkpoint.get('loss', '?'):.4f}\n")
    
    return model

def inference_single(model, image_path, image_size=320, use_tta=False, device='cuda:0'):
    """Run inference on single image"""
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize
    image = cv2.resize(image, (image_size, image_size))
    
    # Normalize
    image = image.astype(np.float32) / 255.0
    image = (image - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    
    # Convert to tensor
    image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(device)
    
    if not use_tta:
        with torch.no_grad():
            output = model(image_tensor)['out']
        pred = torch.argmax(output, dim=1).squeeze().cpu().numpy()
        return pred
    
    # Test Time Augmentation (4 predictions)
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
    
    # Average
    avg_pred = torch.mean(torch.stack(predictions), dim=0)
    pred = torch.argmax(avg_pred, dim=1).squeeze().cpu().numpy()
    
    return pred

def test_dataset(model, dataset_path, split='val', image_size=320, output_dir='./outputs/predictions', device='cuda:0'):
    """Test on entire dataset"""
    image_dir = f'{dataset_path}/{split}/Color_Images'
    mask_dir = f'{dataset_path}/{split}/Segmentation'
    
    os.makedirs(output_dir, exist_ok=True)
    
    image_paths = sorted(Path(image_dir).glob('*.png'))
    
    print(f"Testing on {split} set ({len(image_paths)} images)...\n")
    
    iou_scores = []
    results = []
    
    for img_path in tqdm(image_paths):
        mask_path = Path(mask_dir) / img_path.name
        
        # Inference
        pred = inference_single(model, str(img_path), image_size, device=device)
        
        # Load ground truth
        gt = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        gt = cv2.resize(gt, (image_size, image_size))
        
        # Calculate IoU
        iou = calculate_iou(torch.from_numpy(pred), torch.from_numpy(gt))
        iou_scores.append(iou)
        
        results.append({
            'image': img_path.name,
            'iou': iou
        })
        
        # Save visualization (optional)
        if len(iou_scores) <= 5:  # Save first 5
            pred_colored = cv2.applyColorMap((pred * 255 // 256).astype(np.uint8), cv2.COLORMAP_JET)
            gt_colored = cv2.applyColorMap((gt * 255 // 256).astype(np.uint8), cv2.COLORMAP_JET)
            comparison = np.hstack([gt_colored, pred_colored])
            cv2.imwrite(f'{output_dir}/{img_path.stem}_comparison.png', comparison)
    
    # Print results
    print(f"\n{'='*80}")
    print("TEST RESULTS")
    print(f"{'='*80}\n")
    
    mean_iou = np.mean(iou_scores)
    std_iou = np.std(iou_scores)
    max_iou = np.max(iou_scores)
    min_iou = np.min(iou_scores)
    
    print(f"✓ Dataset: {split}")
    print(f"✓ Images: {len(image_paths)}")
    print(f"✓ Mean IoU: {mean_iou:.4f}")
    print(f"✓ Std IoU: {std_iou:.4f}")
    print(f"✓ Max IoU: {max_iou:.4f}")
    print(f"✓ Min IoU: {min_iou:.4f}\n")
    
    # Save results to JSON
    results_file = f'{output_dir}/{split}_results.json'
    with open(results_file, 'w') as f:
        json.dump({
            'dataset': split,
            'num_images': len(image_paths),
            'mean_iou': mean_iou,
            'std_iou': std_iou,
            'max_iou': max_iou,
            'min_iou': min_iou,
            'per_image_results': results
        }, f, indent=2)
    
    print(f"✓ Results saved: {results_file}")
    print(f"{'='*80}\n")
    
    return mean_iou

# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test semantic segmentation model')
    parser.add_argument('--model', type=str, default='./outputs/checkpoints/best_model.pt', help='Model path')
    parser.add_argument('--dataset-path', type=str, default='./dataset', help='Dataset path')
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val'], help='Dataset split')
    parser.add_argument('--image', type=str, help='Single image path')
    parser.add_argument('--output-dir', type=str, default='./outputs/predictions', help='Output directory')
    parser.add_argument('--tta', action='store_true', help='Use Test Time Augmentation')
    
    args = parser.parse_args()
    config = TestConfig()
    config.use_tta = args.tta
    
    # Load model
    model = load_model(args.model, device=config.device)
    
    # Test single image or dataset
    if args.image:
        print(f"Testing single image: {args.image}\n")
        pred = inference_single(model, args.image, device=config.device, use_tta=args.tta)
        print(f"✓ Prediction shape: {pred.shape}")
        print(f"✓ Unique classes: {len(set(pred.flatten()))}")
    else:
        # Test on full dataset
        mean_iou = test_dataset(model, args.dataset_path, split=args.split, output_dir=args.output_dir, device=config.device)
