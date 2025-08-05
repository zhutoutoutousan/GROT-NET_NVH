#!/usr/bin/env python3
"""
Genius RPM Estimation Architecture
=================================

Advanced deep learning architecture for RPM estimation:
- Frequency Domain Transformer (FDT)
- Multi-Scale LSTM with Attention
- Harmonic Order Detection Network
- Cross-Modal Fusion Architecture

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
import matplotlib.pyplot as plt

class FrequencyDomainTransformer(nn.Module):
    """Transformer specifically designed for frequency domain analysis."""
    
    def __init__(self, freq_dim: int = 1024, d_model: int = 256, n_heads: int = 8, 
                 n_layers: int = 4, dropout: float = 0.1):
        super(FrequencyDomainTransformer, self).__init__()
        
        self.freq_dim = freq_dim
        self.d_model = d_model
        
        # Frequency embedding
        self.freq_embedding = nn.Linear(1, d_model)  # Embed frequency values
        
        # Positional encoding for frequency bins
        self.pos_encoding = nn.Parameter(torch.randn(1, freq_dim, d_model))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Frequency attention
        self.freq_attention = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, 1)
        )
        
    def to(self, device):
        """Override to ensure pos_encoding is moved to device."""
        super().to(device)
        if hasattr(self, 'pos_encoding'):
            self.pos_encoding = self.pos_encoding.to(device)
        return self
        
    def forward(self, magnitude_spectrum: torch.Tensor) -> torch.Tensor:
        """
        Process frequency domain information.
        
        Args:
            magnitude_spectrum: (batch_size, freq_bins)
            
        Returns:
            Frequency features: (batch_size, d_model)
        """
        batch_size = magnitude_spectrum.shape[0]
        
        # Create frequency embeddings
        freq_values = torch.linspace(0, 1, self.freq_dim, device=magnitude_spectrum.device).unsqueeze(0).repeat(batch_size, 1)
        freq_embeddings = self.freq_embedding(freq_values.unsqueeze(-1))
        
        # Add positional encoding
        freq_embeddings = freq_embeddings + self.pos_encoding
        
        # Weight by magnitude spectrum
        weighted_embeddings = freq_embeddings * magnitude_spectrum.unsqueeze(-1)
        
        # Apply transformer
        transformed = self.transformer(weighted_embeddings)
        
        # Apply frequency attention
        attended, _ = self.freq_attention(transformed, transformed, transformed)
        
        # Global average pooling
        pooled = torch.mean(attended, dim=1)
        
        # Output projection
        output = self.output_proj(pooled)
        
        return output

class HarmonicOrderDetector(nn.Module):
    """Detect engine order harmonics from frequency peaks."""
    
    def __init__(self, max_orders: int = 8, hidden_dim: int = 128):
        super(HarmonicOrderDetector, self).__init__()
        
        self.max_orders = max_orders
        
        # Peak detection network
        self.peak_detector = nn.Sequential(
            nn.Linear(2, hidden_dim),  # (frequency, magnitude)
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, max_orders)  # Order probabilities
        )
        
        # Harmonic relationship network
        self.harmonic_net = nn.Sequential(
            nn.Linear(max_orders * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1)  # Fundamental frequency
        )
        
    def forward(self, peak_frequencies: torch.Tensor, peak_magnitudes: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Detect harmonic orders and estimate fundamental frequency.
        
        Args:
            peak_frequencies: (batch_size, num_peaks)
            peak_magnitudes: (batch_size, num_peaks)
            
        Returns:
            Dictionary with harmonic analysis
        """
        batch_size = peak_frequencies.shape[0]
        
        # Combine frequency and magnitude
        peak_features = torch.stack([peak_frequencies, peak_magnitudes], dim=-1)
        
        # Detect order probabilities for each peak
        order_probs = self.peak_detector(peak_features)  # (batch_size, num_peaks, max_orders)
        
        # Find most likely fundamental frequency
        fundamental_candidates = []
        
        for b in range(batch_size):
            for i in range(peak_frequencies.shape[1]):
                if peak_magnitudes[b, i] > 0:  # Valid peak
                    freq = peak_frequencies[b, i]
                    # Check if this frequency has harmonics
                    harmonics_found = 0
                    for j in range(peak_frequencies.shape[1]):
                        if i != j and peak_magnitudes[b, j] > 0:
                            other_freq = peak_frequencies[b, j]
                            # Check for harmonic relationships
                            for order in range(2, self.max_orders + 1):
                                if abs(other_freq - freq * order) < 2.0:  # 2 Hz tolerance
                                    harmonics_found += 1
                                    break
                    
                    if harmonics_found > 0:
                        fundamental_candidates.append((freq, peak_magnitudes[b, i], harmonics_found))
            
            # Select best fundamental frequency
            if fundamental_candidates:
                fundamental_candidates.sort(key=lambda x: (x[2], x[1]), reverse=True)
                fundamental_freq = fundamental_candidates[0][0]
            else:
                # Fallback to strongest peak
                max_idx = torch.argmax(peak_magnitudes[b])
                fundamental_freq = peak_frequencies[b, max_idx]
            
            fundamental_candidates = []  # Reset for next batch
        
        return {
            'order_probs': order_probs,
            'fundamental_freq': fundamental_freq
        }

class MultiScaleLSTM(nn.Module):
    """Multi-scale LSTM for temporal analysis."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 3):
        super(MultiScaleLSTM, self).__init__()
        
        self.hidden_dim = hidden_dim
        
        # Multiple LSTM layers with different scales
        self.lstm_scales = nn.ModuleList([
            nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2),
            nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2),
            nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        ])
        
        # Scale-specific attention
        self.scale_attention = nn.ModuleList([
            nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
            for _ in range(3)
        ])
        
        # Cross-scale fusion
        self.fusion_net = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Multi-scale temporal processing.
        
        Args:
            x: (batch_size, seq_len, input_dim)
            
        Returns:
            Multi-scale features: (batch_size, hidden_dim // 2)
        """
        batch_size = x.shape[0]
        
        # Process with different scales
        scale_outputs = []
        
        for i, (lstm, attention) in enumerate(zip(self.lstm_scales, self.scale_attention)):
            # Apply LSTM
            lstm_out, _ = lstm(x)
            
            # Apply attention
            attended, _ = attention(lstm_out, lstm_out, lstm_out)
            
            # Global average pooling
            pooled = torch.mean(attended, dim=1)
            scale_outputs.append(pooled)
        
        # Concatenate scale outputs
        concatenated = torch.cat(scale_outputs, dim=1)
        
        # Fusion
        fused = self.fusion_net(concatenated)
        
        return fused

class CrossModalFusion(nn.Module):
    """Fuse frequency domain and temporal information."""
    
    def __init__(self, freq_dim: int = 256, temp_dim: int = 64, fusion_dim: int = 128):
        super(CrossModalFusion, self).__init__()
        
        self.freq_dim = freq_dim
        self.temp_dim = temp_dim
        self.fusion_dim = fusion_dim
        
        # Cross-attention between frequency and temporal
        self.cross_attention = nn.MultiheadAttention(
            freq_dim, num_heads=8, batch_first=True
        )
        
        # Fusion network
        self.fusion_net = nn.Sequential(
            nn.Linear(freq_dim + temp_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(fusion_dim // 2, fusion_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(fusion_dim // 4, 1)  # RPM output
        )
        
    def forward(self, freq_features: torch.Tensor, temp_features: torch.Tensor) -> torch.Tensor:
        """
        Fuse frequency and temporal features.
        
        Args:
            freq_features: (batch_size, freq_dim)
            temp_features: (batch_size, temp_dim)
            
        Returns:
            Fused features: (batch_size, 1)
        """
        # Expand temporal features to match frequency dimension
        temp_expanded = temp_features.unsqueeze(1).expand(-1, freq_features.shape[1], -1)
        
        # Cross-attention
        fused, _ = self.cross_attention(freq_features.unsqueeze(1), temp_expanded, temp_expanded)
        fused = fused.squeeze(1)
        
        # Concatenate and fuse
        combined = torch.cat([freq_features, temp_features], dim=1)
        output = self.fusion_net(combined)
        
        return output

class GeniusRPMArchitecture(nn.Module):
    """Genius RPM estimation architecture combining all components."""
    
    def __init__(self, 
                 sample_rate: int = 44100,
                 n_fft: int = 2048,
                 hop_length: int = 512,
                 freq_dim: int = 256,
                 temp_dim: int = 64,
                 fusion_dim: int = 128):
        super(GeniusRPMArchitecture, self).__init__()
        
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.freq_bins = librosa.fft_frequencies(sr=sample_rate, n_fft=n_fft)
        
        # Frequency Domain Transformer
        self.freq_transformer = FrequencyDomainTransformer(
            freq_dim=len(self.freq_bins),
            d_model=freq_dim,
            n_heads=8,
            n_layers=4,
            dropout=0.1
        )
        
        # Harmonic Order Detector
        self.harmonic_detector = HarmonicOrderDetector(
            max_orders=8,
            hidden_dim=128
        )
        
        # Multi-Scale LSTM
        self.temporal_lstm = MultiScaleLSTM(
            input_dim=13,  # MFCC features
            hidden_dim=128,
            num_layers=3
        )
        
        # Cross-Modal Fusion
        self.fusion = CrossModalFusion(
            freq_dim=freq_dim,
            temp_dim=temp_dim,
            fusion_dim=fusion_dim
        )
        
        # Final RPM estimator
        self.rpm_estimator = nn.Sequential(
            nn.Linear(fusion_dim // 4, fusion_dim // 8),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(fusion_dim // 8, 1)
        )
        
    def extract_features(self, audio: np.ndarray) -> Dict[str, torch.Tensor]:
        """Extract all features from audio."""
        # Get device from model parameters
        device = next(self.parameters()).device
        
        # Ensure proper length
        if len(audio) < self.sample_rate * 2:
            audio = np.tile(audio, int(np.ceil(self.sample_rate * 2 / len(audio))))
        
        # STFT
        stft = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length)
        magnitude = np.abs(stft)
        
        # MFCC for temporal features
        mfcc = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=13)
        
        # Find frequency peaks
        avg_magnitude = np.mean(magnitude, axis=1)
        peaks, _ = find_peaks(avg_magnitude, height=np.max(avg_magnitude) * 0.1)
        peak_frequencies = self.freq_bins[peaks]
        peak_magnitudes = avg_magnitude[peaks]
        
        # Pad peaks to fixed size
        max_peaks = 20
        if len(peak_frequencies) > max_peaks:
            peak_frequencies = peak_frequencies[:max_peaks]
            peak_magnitudes = peak_magnitudes[:max_peaks]
        else:
            # Pad with zeros
            padding_size = max_peaks - len(peak_frequencies)
            peak_frequencies = np.pad(peak_frequencies, (0, padding_size), 'constant')
            peak_magnitudes = np.pad(peak_magnitudes, (0, padding_size), 'constant')
        
        return {
            'magnitude_spectrum': torch.tensor(magnitude.T, dtype=torch.float32).to(device),
            'peak_frequencies': torch.tensor(peak_frequencies, dtype=torch.float32).to(device),
            'peak_magnitudes': torch.tensor(peak_magnitudes, dtype=torch.float32).to(device),
            'mfcc': torch.tensor(mfcc.T, dtype=torch.float32).to(device)
        }
    
    def forward(self, audio: np.ndarray) -> Dict[str, torch.Tensor]:
        """
        Complete RPM estimation pipeline.
        
        Args:
            audio: Audio signal
            
        Returns:
            Dictionary with RPM estimation and intermediate features
        """
        # Extract features
        features = self.extract_features(audio)
        
        # Frequency domain processing
        freq_features = self.freq_transformer(features['magnitude_spectrum'])
        
        # Harmonic analysis
        harmonic_analysis = self.harmonic_detector(
            features['peak_frequencies'].unsqueeze(0),
            features['peak_magnitudes'].unsqueeze(0)
        )
        
        # Temporal processing
        temp_features = self.temporal_lstm(features['mfcc'])
        
        # Cross-modal fusion
        fused_features = self.fusion(freq_features, temp_features)
        
        # Final RPM estimation
        estimated_rpm = self.rpm_estimator(fused_features)
        
        return {
            'estimated_rpm': estimated_rpm,
            'freq_features': freq_features,
            'temp_features': temp_features,
            'fused_features': fused_features,
            'harmonic_analysis': harmonic_analysis,
            'raw_features': features
        }

def test_genius_architecture():
    """Test the genius RPM estimation architecture."""
    print("🧠 Testing Genius RPM Architecture")
    print("=" * 50)
    
    # Create test audio
    sample_rate = 44100
    duration = 5.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Test RPMs
    test_rpms = [800, 1500, 2500]
    
    # Initialize model
    model = GeniusRPMArchitecture()
    
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
        print(f"Fused features: {output['fused_features'].shape}")
    
    print("\n✅ Genius architecture test completed!")
    print("\n🎯 Architecture Components:")
    print("1. Frequency Domain Transformer - Processes STFT magnitude")
    print("2. Harmonic Order Detector - Identifies engine orders")
    print("3. Multi-Scale LSTM - Temporal analysis with attention")
    print("4. Cross-Modal Fusion - Combines frequency and temporal")
    print("5. Final RPM Estimator - Outputs RPM prediction")

if __name__ == "__main__":
    test_genius_architecture() 