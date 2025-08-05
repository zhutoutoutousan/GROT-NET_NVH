#!/usr/bin/env python3
"""
Simple STFT Slices Preprocessing
================================

Simplified script to process HL-CEAD audio files and create STFT-based features.
"""

import os
import numpy as np
import librosa
import random
from typing import List, Dict, Tuple
import json

def extract_stft_features(audio: np.ndarray, sr: int = 44100) -> Dict[str, np.ndarray]:
    """Extract STFT features from audio."""
    # STFT parameters
    n_fft = 512
    hop_length = 256
    win_length = 512
    
    # Compute STFT
    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    magnitude_spectrum = np.abs(stft)
    
    # Average magnitude spectrum
    avg_magnitude = np.mean(magnitude_spectrum, axis=1)
    
    # Find frequency peaks (top 5)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    peak_indices = np.argsort(avg_magnitude)[-5:][::-1]
    peak_freqs = freqs[peak_indices]
    peak_mags = avg_magnitude[peak_indices]
    
    # Combine peak features
    peak_features = np.concatenate([peak_freqs, peak_mags])
    
    return {
        'stft_magnitude': magnitude_spectrum,
        'avg_magnitude': avg_magnitude,
        'peak_features': peak_features
    }

def create_audio_slices(audio: np.ndarray, sr: int = 44100, slice_duration: float = 1.0) -> List[Tuple[np.ndarray, float]]:
    """Create 1-second slices from audio."""
    slice_samples = int(slice_duration * sr)
    slices = []
    
    for i in range(0, len(audio) - slice_samples + 1, slice_samples):
        slice_audio = audio[i:i + slice_samples]
        slice_time = i / sr
        slices.append((slice_audio, slice_time))
    
    return slices

def process_audio_file(file_path: str, rpm: float, manufacturer: str, car_model: str) -> List[Dict]:
    """Process a single audio file into slices."""
    try:
        # Load audio
        audio, sr = librosa.load(file_path, sr=44100)
        
        # Create slices
        slices = create_audio_slices(audio, sr)
        
        results = []
        for i, (slice_audio, slice_time) in enumerate(slices):
            # Extract features
            features = extract_stft_features(slice_audio, sr)
            
            # Interpolate RPM for this slice (simple linear interpolation)
            # Assuming RPM changes linearly over time
            total_duration = len(audio) / sr
            if total_duration > 0:
                # Use the original RPM for now (could be enhanced with interpolation)
                slice_rpm = rpm
            else:
                slice_rpm = rpm
            
            # Create result
            result = {
                'stft_magnitude': features['stft_magnitude'],
                'avg_magnitude': features['avg_magnitude'],
                'peak_features': features['peak_features'],
                'slice_rpm': slice_rpm,
                'manufacturer': manufacturer,
                'car_model': car_model,
                'original_rpm': rpm,
                'slice_time': slice_time
            }
            
            results.append(result)
        
        return results
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return []

def main():
    """Main processing function."""
    print("🚀 Starting Simple STFT Slices Preprocessing")
    
    # Base directory
    base_dir = "/app/datasets/hl_cead/extracted/MLVRG-Car-Engine-Audio"
    output_dir = "/app/data/processed/stft_features_v2"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    all_results = []
    file_count = 0
    
    # Process recursively
    for manufacturer in os.listdir(base_dir):
        manufacturer_path = os.path.join(base_dir, manufacturer)
        if not os.path.isdir(manufacturer_path):
            continue
            
        print(f"📁 Processing manufacturer: {manufacturer}")
        
        for car_model in os.listdir(manufacturer_path):
            car_model_path = os.path.join(manufacturer_path, car_model)
            if not os.path.isdir(car_model_path):
                continue
                
            print(f"  🚗 Processing car model: {car_model}")
            
            for rpm_dir in os.listdir(car_model_path):
                rpm_path = os.path.join(car_model_path, rpm_dir)
                if not os.path.isdir(rpm_path):
                    continue
                    
                # Extract RPM from directory name
                try:
                    rpm = float(rpm_dir)
                except ValueError:
                    print(f"    ⚠️ Invalid RPM directory: {rpm_dir}")
                    continue
                    
                print(f"    ⚡ Processing RPM: {rpm}")
                
                # Process all audio files in this RPM directory
                for audio_file in os.listdir(rpm_path):
                    if audio_file.endswith(('.wav', '.mp3', '.flac')):
                        audio_path = os.path.join(rpm_path, audio_file)
                        
                        # Process the audio file
                        results = process_audio_file(audio_path, rpm, manufacturer, car_model)
                        
                        # Save each slice as a separate NPZ file
                        for i, result in enumerate(results):
                            # Create filename
                            filename = f"{file_count:04d}_slice_{i:03d}_stft_features.npz"
                            output_path = os.path.join(output_dir, filename)
                            
                            # Save to NPZ
                            np.savez_compressed(
                                output_path,
                                stft_magnitude=result['stft_magnitude'],
                                avg_magnitude=result['avg_magnitude'],
                                peak_features=result['peak_features'],
                                slice_rpm=result['slice_rpm'],
                                manufacturer=result['manufacturer'],
                                car_model=result['car_model'],
                                original_rpm=result['original_rpm'],
                                slice_time=result['slice_time']
                            )
                            
                            all_results.append({
                                'file_path': output_path,
                                'rpm': result['slice_rpm'],
                                'manufacturer': result['manufacturer'],
                                'car_model': result['car_model']
                            })
                        
                        file_count += 1
                        if file_count % 10 == 0:
                            print(f"      📊 Processed {file_count} files so far...")
    
    # Save dataset info
    dataset_info = {
        'total_files': len(all_results),
        'manufacturers': list(set(r['manufacturer'] for r in all_results)),
        'rpm_range': {
            'min': min(r['rpm'] for r in all_results),
            'max': max(r['rpm'] for r in all_results)
        }
    }
    
    with open(os.path.join(output_dir, 'dataset_info.json'), 'w') as f:
        json.dump(dataset_info, f, indent=2)
    
    print(f"\n✅ Processing completed!")
    print(f"📁 Total files created: {len(all_results)}")
    print(f"📊 RPM range: {dataset_info['rpm_range']['min']} - {dataset_info['rpm_range']['max']}")
    print(f"🏭 Manufacturers: {', '.join(dataset_info['manufacturers'])}")

if __name__ == "__main__":
    main() 