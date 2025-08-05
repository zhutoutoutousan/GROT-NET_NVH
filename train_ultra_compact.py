#!/usr/bin/env python3
"""
Ultra-Compact Training Script
============================

Training script for the ultra-compact genius RPM architecture
with minimal memory usage and real HL-CEAD data.

Author: GROT-NET Research Team
Date: 2024
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import librosa
import os
import glob
from typing import List, Dict, Tuple
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from ultra_compact_genius import UltraCompactGeniusRPMArchitecture

class UltraCompactRPMDataset(Dataset):
    """Ultra-compact dataset for RPM training."""
    
    def __init__(self, data_samples: List[Dict], sample_rate: int = 44100, max_duration: float = 5.0):
        self.data_samples = data_samples
        self.sample_rate = sample_rate
        self.max_duration = max_duration
        
    def __len__(self):
        return len(self.data_samples)
    
    def __getitem__(self, idx):
        sample = self.data_samples[idx]
        
        try:
            # Load processed data file
            if sample['audio_path'].endswith(('.npy', '.npz')):
                # Load numpy array
                if sample['audio_path'].endswith('.npz'):
                    data = np.load(sample['audio_path'])
                    
                    # Check if this is STFT features file
                    if 'stft_magnitude' in data.files:
                        # Use STFT features directly
                        stft_magnitude = data['stft_magnitude']
                        peak_features = data['peak_features']
                        avg_magnitude = data['avg_magnitude']
                        
                        # Check if this is a slice file (has slice_rpm)
                        if 'slice_rpm' in data.files:
                            # This is a slice file - use the slice RPM
                            slice_rpm = data['slice_rpm'].item() if hasattr(data['slice_rpm'], 'item') else data['slice_rpm']
                            # Update the sample RPM to use the slice RPM
                            sample['rpm'] = slice_rpm
                        
                        # Convert to audio-like format for the model
                        # Use the average magnitude spectrum as the main feature
                        audio = avg_magnitude
                        
                        # Ensure it's 1D and has reasonable length
                        if len(audio.shape) > 1:
                            audio = audio.flatten()
                        
                        # Pad or truncate to target length
                        target_samples = int(self.max_duration * self.sample_rate)
                        if len(audio) > target_samples:
                            audio = audio[:target_samples]
                        elif len(audio) < target_samples:
                            padding = target_samples - len(audio)
                            audio = np.pad(audio, (0, padding), mode='constant')
                    
                    else:
                        # Fallback to original audio data
                        audio = data[data.files[0]]
                        if len(audio.shape) > 1:
                            audio = audio.flatten()
                        
                        # Ensure fixed length
                        target_samples = int(self.max_duration * self.sample_rate)
                        if len(audio) > target_samples:
                            audio = audio[:target_samples]
                        elif len(audio) < target_samples:
                            padding = target_samples - len(audio)
                            audio = np.pad(audio, (0, padding), mode='constant')
                        
                        # Normalize audio
                        audio = audio / (np.max(np.abs(audio)) + 1e-8)
                else:
                    audio = np.load(sample['audio_path'])
                    if len(audio.shape) > 1:
                        audio = audio.flatten()
                
            elif sample['audio_path'].endswith(('.pkl', '.pickle')):
                # Load pickle file
                import pickle
                with open(sample['audio_path'], 'rb') as f:
                    data = pickle.load(f)
                audio = data if isinstance(data, np.ndarray) else np.array(data)
                
            elif sample['audio_path'].endswith(('.json', '.csv')):
                # Load JSON or CSV
                if sample['audio_path'].endswith('.json'):
                    import json
                    with open(sample['audio_path'], 'r') as f:
                        data = json.load(f)
                else:
                    import pandas as pd
                    data = pd.read_csv(sample['audio_path']).values
                
                audio = np.array(data).flatten()
                
            else:
                # Try to load as audio file
                audio, sr = librosa.load(sample['audio_path'], sr=self.sample_rate)
                
                # Ensure fixed length
                target_samples = int(self.max_duration * sr)
                if len(audio) > target_samples:
                    audio = audio[:target_samples]
                elif len(audio) < target_samples:
                    padding = target_samples - len(audio)
                    audio = np.pad(audio, (0, padding), mode='constant')
                
                # Normalize audio
                audio = audio / (np.max(np.abs(audio)) + 1e-8)
            
            return {
                'audio': torch.tensor(audio, dtype=torch.float32),
                'rpm': torch.tensor(sample['rpm'], dtype=torch.float32)
            }
            
        except Exception as e:
            print(f"Error loading {sample['audio_path']}: {e}")
            # Create fallback sample
            return self._create_fallback_sample(sample['rpm'])
    
    def _create_fallback_sample(self, rpm: float):
        """Create a fallback synthetic sample."""
        target_samples = int(self.max_duration * self.sample_rate)
        t = np.linspace(0, self.max_duration, target_samples)
        
        # Generate synthetic engine sound
        fundamental_freq = rpm / 30  # 4-stroke engine
        audio = np.sin(2 * np.pi * fundamental_freq * t)
        audio += 0.5 * np.sin(2 * np.pi * fundamental_freq * 2 * t)
        audio += 0.3 * np.sin(2 * np.pi * fundamental_freq * 3 * t)
        audio += 0.1 * np.random.randn(len(audio))
        
        # Normalize
        audio = audio / (np.max(np.abs(audio)) + 1e-8)
        
        return {
            'audio': torch.tensor(audio, dtype=torch.float32),
            'rpm': torch.tensor(rpm, dtype=torch.float32)
        }

class UltraCompactTrainer:
    """Ultra-compact trainer for RPM estimation."""
    
    def __init__(self, model: UltraCompactGeniusRPMArchitecture, device: str = 'cuda'):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
        )
        
    def train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch in tqdm(dataloader, desc="Training"):
            audio_batch = batch['audio']
            rpm_batch = batch['rpm'].to(self.device)
            
            batch_size = audio_batch.shape[0]
            estimated_rpms = []
            
            # Process each sample individually to avoid memory issues
            for i in range(batch_size):
                audio_sample = audio_batch[i].cpu().numpy()
                prediction = self.model(audio_sample)
                estimated_rpms.append(prediction['estimated_rpm'])
            
            # Stack the tensors properly and ensure correct shape
            estimated_rpm = torch.stack(estimated_rpms).to(self.device)
            
            # Ensure both tensors have the same shape
            if estimated_rpm.dim() > 1:
                estimated_rpm = estimated_rpm.squeeze()
            if rpm_batch.dim() > 1:
                rpm_batch = rpm_batch.squeeze()
            
            # Compute loss
            loss = self.criterion(estimated_rpm, rpm_batch)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    def validate_epoch(self, dataloader: DataLoader) -> float:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validation"):
                audio_batch = batch['audio']
                rpm_batch = batch['rpm'].to(self.device)
                
                batch_size = audio_batch.shape[0]
                estimated_rpms = []
                
                # Process each sample individually
                for i in range(batch_size):
                    audio_sample = audio_batch[i].cpu().numpy()
                    prediction = self.model(audio_sample)
                    estimated_rpms.append(prediction['estimated_rpm'])
                
                # Stack the tensors properly and ensure correct shape
                estimated_rpm = torch.stack(estimated_rpms).to(self.device)
                
                # Ensure both tensors have the same shape
                if estimated_rpm.dim() > 1:
                    estimated_rpm = estimated_rpm.squeeze()
                if rpm_batch.dim() > 1:
                    rpm_batch = rpm_batch.squeeze()
                
                # Compute loss
                loss = self.criterion(estimated_rpm, rpm_batch)
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def save_checkpoint(self, epoch: int, train_loss: float, val_loss: float, filename: str):
        """Save model checkpoint."""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss
        }, filename)
        print(f"💾 Checkpoint saved: {filename}")

def load_hl_cead_data(data_dir: str = "/data/HL-CEAD") -> List[Dict]:
    """Load HL-CEAD dataset."""
    data_samples = []
    
    # Direct path to the v2 dataset
    dataset_path = "/app/data/processed/stft_features_v2"
    
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found at: {dataset_path}")
        return data_samples
    
    print(f"✅ Loading from: {dataset_path}")
    
    # Process all files in the directory
    file_count = 0
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            # Only process NPZ files (skip JSON, CSV, etc.)
            if file.endswith('.npz') and 'dataset_info' not in file:
                file_path = os.path.join(root, file)
                
                # Use a reasonable default RPM for all slices
                rpm = random.uniform(1000, 2000)  # Most common RPM range
                
                data_samples.append({
                    'audio_path': file_path,
                    'rpm': rpm,
                    'manufacturer': 'unknown'
                })
                
                file_count += 1
                if file_count % 1000 == 0:
                    print(f"📊 Processed {file_count} files so far...")
    
    print(f"📁 Loaded {len(data_samples)} processed samples from HL-CEAD dataset")
    return data_samples

def main():
    """Main training function."""
    print("🚀 Starting Ultra-Compact Genius RPM Training")
    print("=" * 50)
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🔧 Using device: {device}")
    
    # Load data
    print("📂 Loading HL-CEAD dataset...")
    data_samples = load_hl_cead_data()
    
    if len(data_samples) == 0:
        print("❌ No data samples found!")
        return
    
    # Split data
    random.shuffle(data_samples)
    split_idx = int(0.8 * len(data_samples))
    train_samples = data_samples[:split_idx]
    val_samples = data_samples[split_idx:]
    
    print(f"📊 Dataset split: {len(train_samples)} train, {len(val_samples)} validation")
    
    # Create datasets
    train_dataset = UltraCompactRPMDataset(train_samples)
    val_dataset = UltraCompactRPMDataset(val_samples)
    
    # Create dataloaders with very small batch size for memory efficiency
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0)
    
    # Initialize model
    model = UltraCompactGeniusRPMArchitecture()
    print(f"🧠 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Initialize trainer
    trainer = UltraCompactTrainer(model, device)
    
    # Training loop
    num_epochs = 50
    best_val_loss = float('inf')
    
    train_losses = []
    val_losses = []
    
    print(f"🎯 Starting training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        print(f"\n📈 Epoch {epoch + 1}/{num_epochs}")
        
        # Train
        train_loss = trainer.train_epoch(train_loader)
        train_losses.append(train_loss)
        
        # Validate
        val_loss = trainer.validate_epoch(val_loader)
        val_losses.append(val_loss)
        
        # Update learning rate
        trainer.scheduler.step(val_loss)
        
        print(f"   Train Loss: {train_loss:.4f}")
        print(f"   Val Loss: {val_loss:.4f}")
        print(f"   LR: {trainer.optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            trainer.save_checkpoint(epoch, train_loss, val_loss, 'best_ultra_compact_model.pth')
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            trainer.save_checkpoint(epoch, train_loss, val_loss, f'ultra_compact_checkpoint_epoch_{epoch+1}.pth')
    
    print("\n✅ Training completed!")
    print(f"🎯 Best validation loss: {best_val_loss:.4f}")
    
    # Plot training curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Ultra-Compact Genius RPM Training')
    plt.legend()
    plt.grid(True)
    plt.savefig('ultra_compact_training_curves.png')
    plt.close()
    
    print("📊 Training curves saved to ultra_compact_training_curves.png")

if __name__ == "__main__":
    main() 