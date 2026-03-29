"""
SUBMIT MODEL TO GOOGLE DRIVE
Run this AFTER training completes on Kaggle
"""

import os
import torch
from google.colab import auth
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

# ============================================================================
# AUTHENTICATE WITH GOOGLE DRIVE
# ============================================================================

print("Authenticating with Google Drive...")
auth.authenticate_user()
drive_service = build('drive', 'v3')
print("✓ Authenticated!\n")

# ============================================================================
# LOAD MODEL INFO
# ============================================================================

checkpoint_path = '/kaggle/working/checkpoints/best_model.pt'
history_path = '/kaggle/working/logs/training_history.json'

print("="*80)
print("MODEL CHECKPOINT")
print("="*80)

if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    print(f"✓ Model found: {checkpoint_path}")
    print(f"  Epoch: {checkpoint.get('epoch', '?')}")
    print(f"  Loss: {checkpoint.get('loss', '?'):.4f}")
    print(f"  IoU: {checkpoint.get('iou', '?'):.4f}")
else:
    print("❌ Model not found!")
    exit()

# Load training history
import json
if os.path.exists(history_path):
    with open(history_path, 'r') as f:
        history = json.load(f)
    print(f"\n✓ Training history found")
    print(f"  Epochs completed: {len(history['train_loss'])}")
    print(f"  Final train loss: {history['train_loss'][-1]:.4f}")
    print(f"  Final val loss: {history['val_loss'][-1]:.4f}")
    print(f"  Final train IoU: {history['train_iou'][-1]:.4f}")
    print(f"  Final val IoU: {history['val_iou'][-1]:.4f}")

# ============================================================================
# UPLOAD TO GOOGLE DRIVE
# ============================================================================

print("\n" + "="*80)
print("UPLOADING TO GOOGLE DRIVE")
print("="*80 + "\n")

# Create folder in Google Drive
folder_name = 'Offroad_Segmentation_Results'
query = f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
results = drive_service.files().list(q=query, spaces='drive', pageSize=1).execute()
folders = results.get('files', [])

if folders:
    folder_id = folders[0]['id']
    print(f"✓ Found existing folder: {folder_name} (ID: {folder_id})")
else:
    print(f"Creating folder: {folder_name}...")
    file_metadata = {'name': folder_name, 'mimeType': 'application/vnd.google-apps.folder'}
    folder = drive_service.files().create(body=file_metadata, fields='id').execute()
    folder_id = folder.get('id')
    print(f"✓ Created folder (ID: {folder_id})")

# Upload checkpoint
print(f"\nUploading model checkpoint...")
file_metadata = {
    'name': 'best_model.pt',
    'parents': [folder_id]
}
media = MediaFileUpload(checkpoint_path)
file = drive_service.files().create(
    body=file_metadata,
    media_body=media,
    fields='id,webViewLink'
).execute()
print(f"✓ Uploaded: {file.get('webViewLink')}")

# Upload history
print(f"Uploading training history...")
file_metadata = {
    'name': 'training_history.json',
    'parents': [folder_id]
}
media = MediaFileUpload(history_path)
file = drive_service.files().create(
    body=file_metadata,
    media_body=media,
    fields='id,webViewLink'
).execute()
print(f"✓ Uploaded: {file.get('webViewLink')}")

print("\n" + "="*80)
print("✅ SUBMISSION COMPLETE!")
print("="*80)
print(f"\n📁 Google Drive Folder: {folder_name}")
print(f"   Files uploaded:")
print(f"   • best_model.pt (trained model)")
print(f"   • training_history.json (metrics)")
print("\n" + "="*80)
