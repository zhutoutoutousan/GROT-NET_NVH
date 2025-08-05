#!/usr/bin/env python3
"""
Ultra-Compact Genius RPM Estimation Architecture
===============================================

Ultra-memory-efficient deep learning architecture for RPM estimation:
- Simple CNN for frequency domain analysis
- Compact LSTM for temporal processing
- Direct fusion without complex dimensions
- Minimal memory footprint

Author: GROT-NET Research Team
Date: 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import librosa
from typing import Dict, List, Tuple, Optional
from scipy.signal import find_peaks

class UltraCompactFrequencyCNN(nn.Module):
    """Ultra-compact CNN for frequency domain analysis."""
    
    def __init__(self, input_size: int = 512, output_size: int = 32):
        super(UltraCompactFrequencyCNN, self).__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        
        # Simple CNN layers
        self.conv1 = nn.Conv1d(1, 16, kernel_size=7, stride=2, padding=3)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1)
        
        # Adaptive pooling to get fixed size
        self.adaptive_pool = nn.AdaptiveAvgPool1d(output_size)
        
        # Final projection
        self.output_proj = nn.Sequential(
            nn.Linear(64 * output_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, output_size)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process frequency domain information.
        
        Args:
            x: (batch_size, freq_bins)
            
        Returns:
            Frequency features: (batch_size, output_size)
        """
        # Add channel dimension
        x = x.unsqueeze(1)  # (batch_size, 1, freq_bins)
        
        # Apply CNN layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Adaptive pooling
        x = self.adaptive_pool(x)  # (batch_size, 64, output_size)
        
        # Flatten and project
        x = x.flatten(1)  # (batch_size, 64 * output_size)
        x = self.output_proj(x)
        
        return x

class UltraCompactTemporalLSTM(nn.Module):
    """Ultra-compact LSTM for temporal processing."""
    
    def __init__(self, input_dim: int = 13, hidden_dim: int = 32, output_size: int = 32):
        super(UltraCompactTemporalLSTM, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.output_size = output_size
        
        # Simple LSTM
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=1, batch_first=True)
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, output_size),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process temporal information.
        
        Args:
            x: (batch_size, time_steps, input_dim)
            
        Returns:
            Temporal features: (batch_size, output_size)
        """
        # Apply LSTM
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Use last hidden state
        last_hidden = hidden[-1]  # (batch_size, hidden_dim)
        
        # Project to output size
        output = self.output_proj(last_hidden)
        
        return output

class UltraCompactFusion(nn.Module):
    """Ultra-compact fusion network."""
    
    def __init__(self, freq_size: int = 32, temp_size: int = 32, fusion_size: int = 64):
        super(UltraCompactFusion, self).__init__()
        
        self.fusion_size = fusion_size
        
        # Simple fusion network
        self.fusion_net = nn.Sequential(
            nn.Linear(freq_size + temp_size, fusion_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(fusion_size, fusion_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(fusion_size // 2, 1)  # RPM output
        )
        
    def forward(self, freq_features: torch.Tensor, temp_features: torch.Tensor) -> torch.Tensor:
        """
        Fuse frequency and temporal features.
        
        Args:
            freq_features: (batch_size, freq_size)
            temp_features: (batch_size, temp_size)
            
        Returns:
            Fused features: (batch_size, 1)
        """
        # Concatenate features
        combined = torch.cat([freq_features, temp_features], dim=1)
        
        # Apply fusion network
        output = self.fusion_net(combined)
        
        return output

class UltraCompactGeniusRPMArchitecture(nn.Module):
    """Ultra-compact genius RPM estimation architecture."""
    
    def __init__(self, 
                 sample_rate: int = 44100,
                 n_fft: int = 512,  # Very small FFT
                 hop_length: int = 256,
                 freq_size: int = 32,
                 temp_size: int = 32,
                 fusion_size: int = 64):
        super(UltraCompactGeniusRPMArchitecture, self).__init__()
        
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.freq_bins = librosa.fft_frequencies(sr=sample_rate, n_fft=n_fft)
        
        # Ultra-compact Frequency CNN
        self.freq_cnn = UltraCompactFrequencyCNN(
            input_size=n_fft // 2 + 1,  # Number of frequency bins
            output_size=freq_size
        )
        
        # Ultra-compact Temporal LSTM
        self.temp_lstm = UltraCompactTemporalLSTM(
            input_dim=13,  # MFCC features
            hidden_dim=32,
            output_size=temp_size
        )
        
        # Ultra-compact Fusion
        self.fusion = UltraCompactFusion(
            freq_size=freq_size,
            temp_size=temp_size,
            fusion_size=fusion_size
        )
        
    def extract_features(self, audio: np.ndarray) -> Dict[str, torch.Tensor]:
        """Extract features from audio."""
        # Get device from model parameters
        device = next(self.parameters()).device
        
        # Ensure proper length
        if len(audio) < self.sample_rate * 2:
            audio = np.tile(audio, int(np.ceil(self.sample_rate * 2 / len(audio))))
        
        # STFT with very small n_fft
        stft = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length)
        magnitude = np.abs(stft)
        
        # MFCC for temporal features
        mfcc = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=13)
        
        # Average magnitude spectrum across time
        avg_magnitude = np.mean(magnitude, axis=1)
        
        return {
            'magnitude_spectrum': torch.tensor(avg_magnitude, dtype=torch.float32).to(device),
            'mfcc': torch.tensor(mfcc.T, dtype=torch.float32).to(device)
        }
    
    def forward(self, audio: np.ndarray) -> Dict[str, torch.Tensor]:
        """
        Ultra-compact RPM estimation pipeline.
        """
        # Extract features
        features = self.extract_features(audio)
        
        # Frequency domain processing
        freq_features = self.freq_cnn(features['magnitude_spectrum'].unsqueeze(0))
        
        # Temporal processing
        temp_features = self.temp_lstm(features['mfcc'].unsqueeze(0))
        
        # Fusion
        estimated_rpm = self.fusion(freq_features, temp_features)
        
        return {
            'estimated_rpm': estimated_rpm,
            'freq_features': freq_features,
            'temp_features': temp_features
        }

def test_ultra_compact_architecture():
    """Test the ultra-compact genius RPM estimation architecture."""
    print("🧠 Testing Ultra-Compact Genius RPM Architecture")
    print("=" * 60)
    
    # Create test audio
    sample_rate = 44100
    duration = 5.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Test RPMs
    test_rpms = [800, 1500, 2500]
    
    # Initialize model
    model = UltraCompactGeniusRPMArchitecture()
    
    # Print model size
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 Model Statistics:")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Model size: {total_params * 4 / 1024 / 1024:.2f} MB")
    
    for rpm in test_rpms:
        print(f"\n🔍 Testing RPM: {rpm}")
        
        # Generate audio with harmonics
        fundamental_freq = rpm / 30  # 4-stroke engine
        audio = np.sin(2 * np.pi * fundamental_freq * t)  # Fundamental
        audio += 0.5 * np.sin(2 * np.pi * fundamental_freq * 2 * t)  # 2nd harmonic
        audio += 0.3 * np.sin(2 * np.pi * fundamental_freq * 3 * t)  # 3rd harmonic
        audio += 0.2 * np.sin(2 * np.pi * fundamental_freq * 4 * t)  # 4th harmonic
        audio += 0.1 * np.random.randn(len(audio))  # Noise
        
        # Forward pass
        with torch.no_grad():
            output = model(audio)
        
        estimated_rpm = output['estimated_rpm'].item()
        error = abs(estimated_rpm - rpm)
        error_percentage = (error / rpm) * 100
        
        print(f"True RPM: {rpm}")
        print(f"Estimated RPM: {estimated_rpm:.1f}")
        print(f"Error: {error:.1f} RPM ({error_percentage:.1f}%)")
        
        # Print feature dimensions
        print(f"Frequency features: {output['freq_features'].shape}")
        print(f"Temporal features: {output['temp_features'].shape}")
    
    print("\n✅ Ultra-compact architecture test completed!")
    print("\n🎯 Ultra-Memory Optimizations:")
    print("1. Simple CNN instead of transformer")
    print("2. Single LSTM layer")
    print("3. Very small FFT (512 instead of 2048)")
    print("4. Fixed output dimensions (32)")
    print("5. Minimal fusion network")
    print("6. No complex attention mechanisms")

if __name__ == "__main__":
    test_ultra_compact_architecture() 