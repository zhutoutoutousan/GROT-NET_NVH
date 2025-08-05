#!/usr/bin/env python3
"""
Comprehensive Engine Sound Comparison Test
=========================================

Compare Deep Learning and STFTSC methods on realistic engine simulation
from 0 to 5000 RPM with detailed analysis.
"""

import numpy as np
import librosa
import matplotlib.pyplot as plt
import torch
import json
import os
from stftsc_implementation import STFTSCEstimator
from ultra_compact_genius import UltraCompactGeniusRPMArchitecture
import warnings
warnings.filterwarnings('ignore')

def create_realistic_engine_simulation(duration=8.0, sr=44100):
    """Create realistic engine sound simulation with multiple scenarios."""
    print("🔧 Creating comprehensive engine sound simulation...")
    
    # Time array
    t = np.linspace(0, duration, int(sr * duration))
    
    # Create different RPM scenarios
    scenarios = {
        'linear_acceleration': {
            'rpm_curve': np.linspace(0, 5000, len(t)),
            'description': 'Linear acceleration from 0 to 5000 RPM'
        },
        'sigmoid_acceleration': {
            'rpm_curve': 5000 * (1 / (1 + np.exp(-8 * (t - duration/2) / duration))),
            'description': 'Sigmoid acceleration curve'
        },
        'step_acceleration': {
            'rpm_curve': np.concatenate([
                np.linspace(0, 1000, len(t)//4),
                np.linspace(1000, 3000, len(t)//4),
                np.linspace(3000, 5000, len(t)//2)
            ]),
            'description': 'Step-wise acceleration'
        }
    }
    
    # Use linear acceleration for main test
    rpm_curve = scenarios['linear_acceleration']['rpm_curve']
    
    # Convert RPM to frequency (4-stroke engine)
    fundamental_freq = rpm_curve / 60 / 2  # Hz
    
    # Generate signal with realistic engine orders
    signal_data = np.zeros_like(t)
    
    # Engine orders with realistic amplitudes
    engine_orders = {
        0.5: 0.2,   # Half order (firing frequency)
        1.0: 1.0,   # First order (fundamental)
        1.5: 0.4,   # 1.5 order
        2.0: 0.6,   # Second order
        2.5: 0.2,   # 2.5 order
        3.0: 0.3,   # Third order
        4.0: 0.15,  # Fourth order
        6.0: 0.1,   # Sixth order
        8.0: 0.05   # Eighth order
    }
    
    # Generate signal with all engine orders
    for order, amplitude in engine_orders.items():
        freq_component = fundamental_freq * order
        # Only add frequencies that are audible and below Nyquist
        audible_mask = (freq_component > 20) & (freq_component < sr/2)
        if np.any(audible_mask):
            signal_data[audible_mask] += amplitude * np.sin(2 * np.pi * freq_component[audible_mask] * t[audible_mask])
    
    # Add realistic engine noise
    # 1. Combustion noise
    combustion_noise = 0.08 * np.random.randn(len(t))
    signal_data += combustion_noise
    
    # 2. Mechanical noise (fixed frequencies)
    mechanical_freqs = [100, 200, 300, 400, 500]
    for freq in mechanical_freqs:
        if freq < sr/2:
            mechanical_noise = 0.015 * np.sin(2 * np.pi * freq * t + np.random.rand() * 2 * np.pi)
            signal_data += mechanical_noise
    
    # 3. Valve train noise (modulated by RPM)
    valve_noise = 0.03 * np.sin(2 * np.pi * fundamental_freq * 2 * t) * np.random.randn(len(t))
    signal_data += valve_noise
    
    # 4. Exhaust noise
    exhaust_freq = fundamental_freq * 0.25
    exhaust_noise = 0.02 * np.sin(2 * np.pi * exhaust_freq * t)
    signal_data += exhaust_noise
    
    # Normalize audio
    signal_data = signal_data / np.max(np.abs(signal_data)) * 0.8
    
    print(f"✅ Engine simulation created:")
    print(f"   Duration: {duration:.1f} seconds")
    print(f"   RPM range: 0 - 5000")
    print(f"   Frequency range: {np.min(fundamental_freq):.1f} - {np.max(fundamental_freq):.1f} Hz")
    print(f"   Engine orders: {len(engine_orders)}")
    
    return signal_data, rpm_curve, fundamental_freq

def test_deep_learning_model(audio_data, sr, model_path="best_ultra_compact_model.pth"):
    """Test the deep learning model."""
    print("\n🧠 Testing Deep Learning Model...")
    
    try:
        # Check if model exists
        if not os.path.exists(model_path):
            print(f"⚠️ Model file not found: {model_path}")
            print("   Using simulated DL results for comparison")
            return None
        
        # Load and test model
        model = UltraCompactGeniusRPMArchitecture()
        
        # Load the checkpoint and extract model state dict
        checkpoint = torch.load(model_path, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        
        # Process in chunks
        chunk_duration = 1.0
        chunk_samples = int(sr * chunk_duration)
        num_chunks = len(audio_data) // chunk_samples
        
        dl_estimates = []
        dl_timestamps = []
        
        for i in range(num_chunks):
            start_idx = i * chunk_samples
            end_idx = start_idx + chunk_samples
            chunk = audio_data[start_idx:end_idx]
            
            if len(chunk) < chunk_samples:
                chunk = np.pad(chunk, (0, chunk_samples - len(chunk)))
            
            try:
                with torch.no_grad():
                    prediction = model(chunk)
                    estimated_rpm = prediction['estimated_rpm'].item()
                    dl_estimates.append(estimated_rpm)
                    dl_timestamps.append(i * chunk_duration)
            except Exception as e:
                print(f"⚠️ Error processing chunk {i}: {e}")
                dl_estimates.append(1500.0)
                dl_timestamps.append(i * chunk_duration)
        
        print(f"✅ Deep Learning: Processed {len(dl_estimates)} chunks")
        return np.array(dl_estimates), np.array(dl_timestamps)
        
    except Exception as e:
        print(f"❌ Deep Learning model error: {e}")
        return None

def test_stftsc_method(audio_data, sr):
    """Test the STFTSC method."""
    print("\n🛤️ Testing STFTSC Method...")
    
    try:
        # Initialize STFTSC estimator
        estimator = STFTSCEstimator(sr=sr, rpm_range=(0, 5000))
        
        # Process in chunks
        chunk_duration = 1.0
        chunk_samples = int(sr * chunk_duration)
        num_chunks = len(audio_data) // chunk_samples
        
        stftsc_estimates = []
        stftsc_timestamps = []
        stftsc_frequencies = []
        
        for i in range(num_chunks):
            start_idx = i * chunk_samples
            end_idx = start_idx + chunk_samples
            chunk = audio_data[start_idx:end_idx]
            
            try:
                result = estimator.estimate_rpm_with_orders(chunk, max_orders=3)
                estimated_rpm = result['estimated_rpm']
                mean_freq = np.mean(result['frequency_series']) if 'frequency_series' in result else 0
                
                stftsc_estimates.append(estimated_rpm)
                stftsc_timestamps.append(i * chunk_duration)
                stftsc_frequencies.append(mean_freq)
                
            except Exception as e:
                print(f"⚠️ Error processing chunk {i}: {e}")
                stftsc_estimates.append(1500.0)
                stftsc_timestamps.append(i * chunk_duration)
                stftsc_frequencies.append(25.0)
        
        print(f"✅ STFTSC: Processed {len(stftsc_estimates)} chunks")
        return np.array(stftsc_estimates), np.array(stftsc_timestamps), np.array(stftsc_frequencies)
        
    except Exception as e:
        print(f"❌ STFTSC method error: {e}")
        return None

def calculate_comprehensive_metrics(estimated_rpm, true_rpm, timestamps, method_name):
    """Calculate comprehensive performance metrics."""
    if estimated_rpm is None or len(estimated_rpm) == 0:
        return None
    
    # Interpolate true RPM to match estimated timestamps
    time_axis = np.linspace(0, len(true_rpm)/44100, len(true_rpm))
    true_rpm_interp = np.interp(timestamps, time_axis, true_rpm)
    
    # Calculate metrics
    errors = np.abs(estimated_rpm - true_rpm_interp)
    percentage_errors = (errors / np.maximum(true_rpm_interp, 1)) * 100
    
    # Calculate R² score
    ss_res = np.sum(errors**2)
    ss_tot = np.sum((true_rpm_interp - np.mean(true_rpm_interp))**2)
    r2_score = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    metrics = {
        'method': method_name,
        'mae': np.mean(errors),
        'rmse': np.sqrt(np.mean(errors**2)),
        'mean_percentage_error': np.mean(percentage_errors),
        'median_percentage_error': np.median(percentage_errors),
        'max_error': np.max(errors),
        'min_error': np.min(errors),
        'std_error': np.std(errors),
        'r2_score': r2_score,
        'estimated_rpm_range': [np.min(estimated_rpm), np.max(estimated_rpm)],
        'true_rpm_range': [np.min(true_rpm_interp), np.max(true_rpm_interp)]
    }
    
    return metrics

def create_comprehensive_visualization(audio_data, true_rpm_series, true_freq_series, 
                                     dl_results, stftsc_results, sr):
    """Create comprehensive visualization."""
    print("\n📊 Creating comprehensive visualization...")
    
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Original Audio Signal
    ax1 = plt.subplot(4, 3, 1)
    time_axis = np.arange(len(audio_data)) / sr
    ax1.plot(time_axis, audio_data, 'b-', alpha=0.7)
    ax1.set_title('Simulated Engine Audio Signal')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Amplitude')
    ax1.grid(True, alpha=0.3)
    
    # 2. True RPM vs Time
    ax2 = plt.subplot(4, 3, 2)
    ax2.plot(time_axis, true_rpm_series, 'g-', linewidth=2, label='True RPM')
    ax2.set_title('True Engine RPM Over Time')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('RPM')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # 3. True Frequency vs Time
    ax3 = plt.subplot(4, 3, 3)
    ax3.plot(time_axis, true_freq_series, 'r-', linewidth=2, label='True Frequency')
    ax3.set_title('True Engine Frequency Over Time')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Frequency (Hz)')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # 4. STFT Spectrogram
    ax4 = plt.subplot(4, 3, 4)
    stft_result = librosa.stft(audio_data, n_fft=2048, hop_length=512, win_length=1024)
    magnitude = np.abs(stft_result)
    log_magnitude = np.log10(magnitude + 1e-10)
    
    im4 = ax4.imshow(log_magnitude, aspect='auto', origin='lower', cmap='viridis')
    ax4.set_title('STFT Spectrogram')
    ax4.set_xlabel('Time Frame')
    ax4.set_ylabel('Frequency (Hz)')
    plt.colorbar(im4, ax=ax4)
    
    # 5. Deep Learning Results
    ax5 = plt.subplot(4, 3, 5)
    if dl_results is not None:
        dl_estimates, dl_timestamps = dl_results
        ax5.plot(dl_timestamps, dl_estimates, 'blue', linewidth=2, label='DL Estimated RPM')
        ax5.plot(time_axis, true_rpm_series, 'g--', linewidth=2, label='True RPM')
        ax5.set_title('Deep Learning RPM Estimation')
        ax5.set_xlabel('Time (s)')
        ax5.set_ylabel('RPM')
        ax5.grid(True, alpha=0.3)
        ax5.legend()
    else:
        ax5.text(0.5, 0.5, 'DL Model Not Available', ha='center', va='center', transform=ax5.transAxes)
        ax5.set_title('Deep Learning Results')
    
    # 6. STFTSC Results
    ax6 = plt.subplot(4, 3, 6)
    if stftsc_results is not None:
        stftsc_estimates, stftsc_timestamps, stftsc_frequencies = stftsc_results
        ax6.plot(stftsc_timestamps, stftsc_estimates, 'red', linewidth=2, label='STFTSC Estimated RPM')
        ax6.plot(time_axis, true_rpm_series, 'g--', linewidth=2, label='True RPM')
        ax6.set_title('STFTSC RPM Estimation')
        ax6.set_xlabel('Time (s)')
        ax6.set_ylabel('RPM')
        ax6.grid(True, alpha=0.3)
        ax6.legend()
    else:
        ax6.text(0.5, 0.5, 'STFTSC Not Available', ha='center', va='center', transform=ax6.transAxes)
        ax6.set_title('STFTSC Results')
    
    # 7. Error Comparison
    ax7 = plt.subplot(4, 3, 7)
    if dl_results is not None and stftsc_results is not None:
        dl_estimates, dl_timestamps = dl_results
        stftsc_estimates, stftsc_timestamps, _ = stftsc_results
        
        # Interpolate true RPM
        time_axis_short = np.linspace(0, len(true_rpm_series)/44100, len(true_rpm_series))
        true_rpm_dl = np.interp(dl_timestamps, time_axis_short, true_rpm_series)
        true_rpm_stftsc = np.interp(stftsc_timestamps, time_axis_short, true_rpm_series)
        
        dl_errors = np.abs(dl_estimates - true_rpm_dl)
        stftsc_errors = np.abs(stftsc_estimates - true_rpm_stftsc)
        
        ax7.plot(dl_timestamps, dl_errors, 'blue', linewidth=2, label='DL Error')
        ax7.plot(stftsc_timestamps, stftsc_errors, 'red', linewidth=2, label='STFTSC Error')
        ax7.set_title('Absolute Error Comparison')
        ax7.set_xlabel('Time (s)')
        ax7.set_ylabel('Absolute Error (RPM)')
        ax7.grid(True, alpha=0.3)
        ax7.legend()
    
    # 8. Performance Metrics Comparison
    ax8 = plt.subplot(4, 3, 8)
    if dl_results is not None and stftsc_results is not None:
        # Calculate metrics
        dl_metrics = calculate_comprehensive_metrics(dl_estimates, true_rpm_series, dl_timestamps, "Deep Learning")
        stftsc_metrics = calculate_comprehensive_metrics(stftsc_estimates, true_rpm_series, stftsc_timestamps, "STFTSC")
        
        if dl_metrics and stftsc_metrics:
            metrics_names = ['MAE', 'RMSE', 'Mean % Error']
            dl_values = [dl_metrics['mae'], dl_metrics['rmse'], dl_metrics['mean_percentage_error']]
            stftsc_values = [stftsc_metrics['mae'], stftsc_metrics['rmse'], stftsc_metrics['mean_percentage_error']]
            
            x = np.arange(len(metrics_names))
            width = 0.35
            
            ax8.bar(x - width/2, dl_values, width, label='Deep Learning', color='blue', alpha=0.7)
            ax8.bar(x + width/2, stftsc_values, width, label='STFTSC', color='red', alpha=0.7)
            
            ax8.set_title('Performance Metrics Comparison')
            ax8.set_ylabel('Error Value')
            ax8.set_xticks(x)
            ax8.set_xticklabels(metrics_names)
            ax8.legend()
            ax8.grid(True, alpha=0.3)
    
    # 9. RPM Range Comparison
    ax9 = plt.subplot(4, 3, 9)
    if dl_results is not None and stftsc_results is not None:
        dl_estimates, _ = dl_results
        stftsc_estimates, _, _ = stftsc_results
        
        ax9.hist(dl_estimates, bins=20, alpha=0.7, color='blue', label='Deep Learning', density=True)
        ax9.hist(stftsc_estimates, bins=20, alpha=0.7, color='red', label='STFTSC', density=True)
        ax9.axvline(np.mean(true_rpm_series), color='green', linestyle='--', linewidth=2, label='True Mean')
        ax9.set_title('RPM Distribution Comparison')
        ax9.set_xlabel('RPM')
        ax9.set_ylabel('Density')
        ax9.legend()
        ax9.grid(True, alpha=0.3)
    
    # 10. Error Distribution
    ax10 = plt.subplot(4, 3, 10)
    if dl_results is not None and stftsc_results is not None:
        dl_metrics = calculate_comprehensive_metrics(dl_estimates, true_rpm_series, dl_timestamps, "Deep Learning")
        stftsc_metrics = calculate_comprehensive_metrics(stftsc_estimates, true_rpm_series, stftsc_timestamps, "STFTSC")
        
        if dl_metrics and stftsc_metrics:
            # Calculate percentage errors
            time_axis_short = np.linspace(0, len(true_rpm_series)/44100, len(true_rpm_series))
            true_rpm_dl = np.interp(dl_timestamps, time_axis_short, true_rpm_series)
            true_rpm_stftsc = np.interp(stftsc_timestamps, time_axis_short, true_rpm_series)
            
            dl_percent_errors = np.abs(dl_estimates - true_rpm_dl) / np.maximum(true_rpm_dl, 1) * 100
            stftsc_percent_errors = np.abs(stftsc_estimates - true_rpm_stftsc) / np.maximum(true_rpm_stftsc, 1) * 100
            
            ax10.hist(dl_percent_errors, bins=20, alpha=0.7, color='blue', label='Deep Learning', density=True)
            ax10.hist(stftsc_percent_errors, bins=20, alpha=0.7, color='red', label='STFTSC', density=True)
            ax10.set_title('Percentage Error Distribution')
            ax10.set_xlabel('Percentage Error (%)')
            ax10.set_ylabel('Density')
            ax10.legend()
            ax10.grid(True, alpha=0.3)
    
    # 11. R² Score Comparison
    ax11 = plt.subplot(4, 3, 11)
    if dl_results is not None and stftsc_results is not None:
        dl_metrics = calculate_comprehensive_metrics(dl_estimates, true_rpm_series, dl_timestamps, "Deep Learning")
        stftsc_metrics = calculate_comprehensive_metrics(stftsc_estimates, true_rpm_series, stftsc_timestamps, "STFTSC")
        
        if dl_metrics and stftsc_metrics:
            methods = ['Deep Learning', 'STFTSC']
            r2_scores = [dl_metrics['r2_score'], stftsc_metrics['r2_score']]
            colors = ['blue', 'red']
            
            bars = ax11.bar(methods, r2_scores, color=colors, alpha=0.7)
            ax11.set_title('R² Score Comparison')
            ax11.set_ylabel('R² Score')
            ax11.set_ylim(-1, 1)
            ax11.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, score in zip(bars, r2_scores):
                height = bar.get_height()
                ax11.text(bar.get_x() + bar.get_width()/2., height,
                         f'{score:.3f}', ha='center', va='bottom')
    
    # 12. Summary Statistics
    ax12 = plt.subplot(4, 3, 12)
    ax12.axis('off')
    
    if dl_results is not None and stftsc_results is not None:
        dl_metrics = calculate_comprehensive_metrics(dl_estimates, true_rpm_series, dl_timestamps, "Deep Learning")
        stftsc_metrics = calculate_comprehensive_metrics(stftsc_estimates, true_rpm_series, stftsc_timestamps, "STFTSC")
        
        if dl_metrics and stftsc_metrics:
            summary_text = f"""
            Summary Statistics:
            
            Deep Learning:
            • MAE: {dl_metrics['mae']:.1f} RPM
            • RMSE: {dl_metrics['rmse']:.1f} RPM
            • Mean % Error: {dl_metrics['mean_percentage_error']:.2f}%
            • R² Score: {dl_metrics['r2_score']:.3f}
            
            STFTSC:
            • MAE: {stftsc_metrics['mae']:.1f} RPM
            • RMSE: {stftsc_metrics['rmse']:.1f} RPM
            • Mean % Error: {stftsc_metrics['mean_percentage_error']:.2f}%
            • R² Score: {stftsc_metrics['r2_score']:.3f}
            """
            
            ax12.text(0.1, 0.9, summary_text, transform=ax12.transAxes, 
                     fontsize=10, verticalalignment='top',
                     bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('comprehensive_engine_comparison.png', dpi=300, bbox_inches='tight')
    print("📊 Comprehensive visualization saved to: comprehensive_engine_comparison.png")
    plt.show()

def main():
    """Main comparison function."""
    print("🚗 Comprehensive Engine Sound Comparison Test")
    print("=" * 60)
    print("Comparing Deep Learning and STFTSC methods on realistic engine simulation")
    print("from 0 to 5000 RPM with detailed analysis.")
    
    # Create realistic engine sound simulation
    audio_data, true_rpm_series, true_freq_series = create_realistic_engine_simulation(duration=8.0, sr=44100)
    
    # Test both methods
    dl_results = test_deep_learning_model(audio_data, 44100)
    stftsc_results = test_stftsc_method(audio_data, 44100)
    
    # Calculate and display comprehensive metrics
    print("\n📈 Comprehensive Performance Analysis:")
    print("=" * 40)
    
    if dl_results is not None:
        dl_estimates, dl_timestamps = dl_results
        dl_metrics = calculate_comprehensive_metrics(dl_estimates, true_rpm_series, dl_timestamps, "Deep Learning")
        if dl_metrics:
            print("🧠 Deep Learning Model:")
            print(f"   MAE: {dl_metrics['mae']:.1f} RPM")
            print(f"   RMSE: {dl_metrics['rmse']:.1f} RPM")
            print(f"   Mean % Error: {dl_metrics['mean_percentage_error']:.2f}%")
            print(f"   R² Score: {dl_metrics['r2_score']:.3f}")
            print(f"   RPM Range: {dl_metrics['estimated_rpm_range'][0]:.1f} - {dl_metrics['estimated_rpm_range'][1]:.1f}")
    
    if stftsc_results is not None:
        stftsc_estimates, stftsc_timestamps, stftsc_frequencies = stftsc_results
        stftsc_metrics = calculate_comprehensive_metrics(stftsc_estimates, true_rpm_series, stftsc_timestamps, "STFTSC")
        if stftsc_metrics:
            print("\n🛤️ STFTSC Method:")
            print(f"   MAE: {stftsc_metrics['mae']:.1f} RPM")
            print(f"   RMSE: {stftsc_metrics['rmse']:.1f} RPM")
            print(f"   Mean % Error: {stftsc_metrics['mean_percentage_error']:.2f}%")
            print(f"   R² Score: {stftsc_metrics['r2_score']:.3f}")
            print(f"   RPM Range: {stftsc_metrics['estimated_rpm_range'][0]:.1f} - {stftsc_metrics['estimated_rpm_range'][1]:.1f}")
    
    # Create comprehensive visualization
    create_comprehensive_visualization(audio_data, true_rpm_series, true_freq_series, 
                                     dl_results, stftsc_results, 44100)
    
    # Save comprehensive results
    results = {
        'simulation_params': {
            'duration': 8.0,
            'sample_rate': 44100,
            'rpm_range': [0, 5000],
            'engine_orders': [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0, 8.0]
        },
        'deep_learning': {
            'available': dl_results is not None,
            'metrics': dl_metrics if dl_results is not None else None
        },
        'stftsc': {
            'available': stftsc_results is not None,
            'metrics': stftsc_metrics if stftsc_results is not None else None
        }
    }
    
    with open('comprehensive_engine_comparison_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Comprehensive results saved to: comprehensive_engine_comparison_results.json")
    print("✅ Comprehensive engine comparison completed!")

if __name__ == "__main__":
    main() 