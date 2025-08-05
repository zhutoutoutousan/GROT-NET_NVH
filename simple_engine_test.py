#!/usr/bin/env python3
"""
Simple Engine Sound Simulation Test
==================================

Simplified test for engine sound simulation from 0 to 5000 RPM.
"""

import numpy as np
import librosa
import matplotlib.pyplot as plt
import json
from stftsc_implementation import STFTSCEstimator
import warnings
warnings.filterwarnings('ignore')

def create_simple_engine_sound(duration=5.0, sr=44100):
    """Create a simple but realistic engine sound simulation."""
    print("🔧 Creating simple engine sound simulation...")
    
    # Time array
    t = np.linspace(0, duration, int(sr * duration))
    
    # Simple RPM progression: linear increase from 0 to 5000 RPM
    rpm_start = 0
    rpm_end = 5000
    rpm_curve = rpm_start + (rpm_end - rpm_start) * t / duration
    
    # Convert RPM to frequency (4-stroke engine)
    fundamental_freq = rpm_curve / 60 / 2  # Hz
    
    # Generate signal with main engine orders
    signal_data = np.zeros_like(t)
    
    # Simplified engine orders
    engine_orders = {
        1.0: 1.0,   # First order (fundamental)
        2.0: 0.6,   # Second order
        3.0: 0.3,   # Third order
        4.0: 0.15   # Fourth order
    }
    
    # Generate signal with engine orders
    for order, amplitude in engine_orders.items():
        freq_component = fundamental_freq * order
        # Only add frequencies that are audible and below Nyquist
        audible_mask = (freq_component > 20) & (freq_component < sr/2)
        if np.any(audible_mask):
            signal_data[audible_mask] += amplitude * np.sin(2 * np.pi * freq_component[audible_mask] * t[audible_mask])
    
    # Add some noise
    noise = 0.05 * np.random.randn(len(t))
    signal_data += noise
    
    # Normalize audio
    signal_data = signal_data / np.max(np.abs(signal_data)) * 0.8
    
    print(f"✅ Simple engine sound created:")
    print(f"   Duration: {duration:.1f} seconds")
    print(f"   RPM range: {rpm_start} - {rpm_end}")
    print(f"   Frequency range: {np.min(fundamental_freq):.1f} - {np.max(fundamental_freq):.1f} Hz")
    
    return signal_data, rpm_curve, fundamental_freq

def test_stftsc_simple(audio_data, sr):
    """Test STFTSC on the simple engine sound."""
    print("\n🛤️ Testing STFTSC Method...")
    
    try:
        # Initialize STFTSC estimator
        estimator = STFTSCEstimator(sr=sr, rpm_range=(0, 5000))
        
        # Process the entire audio at once
        result = estimator.estimate_rpm_with_orders(audio_data, max_orders=3)
        
        if result and 'estimated_rpm' in result:
            estimated_rpm = result['estimated_rpm']
            print(f"✅ STFTSC estimated RPM: {estimated_rpm:.1f}")
            
            # Calculate metrics against the middle RPM (average)
            true_rpm_avg = np.mean(np.linspace(0, 5000, len(audio_data)))
            error = abs(estimated_rpm - true_rpm_avg)
            percentage_error = (error / true_rpm_avg) * 100
            
            print(f"   True average RPM: {true_rpm_avg:.1f}")
            print(f"   Error: {error:.1f} RPM ({percentage_error:.2f}%)")
            
            return {
                'estimated_rpm': estimated_rpm,
                'error': error,
                'percentage_error': percentage_error,
                'true_avg_rpm': true_rpm_avg
            }
        else:
            print("❌ STFTSC failed to estimate RPM")
            return None
            
    except Exception as e:
        print(f"❌ STFTSC method error: {e}")
        return None

def create_visualization(audio_data, true_rpm_series, true_freq_series, stftsc_result, sr):
    """Create visualization of the test results."""
    print("\n📊 Creating visualization...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Simple Engine Sound Simulation Test', fontsize=16, fontweight='bold')
    
    # 1. Original Audio Signal
    ax1 = axes[0, 0]
    time_axis = np.arange(len(audio_data)) / sr
    ax1.plot(time_axis, audio_data, 'b-', alpha=0.7)
    ax1.set_title('Simulated Engine Audio Signal')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Amplitude')
    ax1.grid(True, alpha=0.3)
    
    # 2. True RPM vs Time
    ax2 = axes[0, 1]
    ax2.plot(time_axis, true_rpm_series, 'g-', linewidth=2, label='True RPM')
    if stftsc_result:
        ax2.axhline(y=stftsc_result['estimated_rpm'], color='red', linestyle='--', 
                    linewidth=2, label=f"STFTSC Estimate: {stftsc_result['estimated_rpm']:.1f}")
    ax2.set_title('Engine RPM Over Time')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('RPM')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # 3. True Frequency vs Time
    ax3 = axes[1, 0]
    ax3.plot(time_axis, true_freq_series, 'r-', linewidth=2, label='True Frequency')
    ax3.set_title('Engine Frequency Over Time')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Frequency (Hz)')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # 4. STFT Spectrogram
    ax4 = axes[1, 1]
    stft_result = librosa.stft(audio_data, n_fft=2048, hop_length=512, win_length=1024)
    magnitude = np.abs(stft_result)
    log_magnitude = np.log10(magnitude + 1e-10)
    
    im4 = ax4.imshow(log_magnitude, aspect='auto', origin='lower', cmap='viridis')
    ax4.set_title('STFT Spectrogram')
    ax4.set_xlabel('Time Frame')
    ax4.set_ylabel('Frequency (Hz)')
    plt.colorbar(im4, ax=ax4)
    
    plt.tight_layout()
    plt.savefig('simple_engine_test_visualization.png', dpi=300, bbox_inches='tight')
    print("📊 Visualization saved to: simple_engine_test_visualization.png")
    plt.show()

def main():
    """Main test function."""
    print("🚗 Simple Engine Sound Simulation Test")
    print("=" * 50)
    print("Testing STFTSC method on simple engine simulation from 0 to 5000 RPM.")
    
    # Create simple engine sound simulation
    audio_data, true_rpm_series, true_freq_series = create_simple_engine_sound(duration=5.0, sr=44100)
    
    # Test STFTSC method
    stftsc_result = test_stftsc_simple(audio_data, 44100)
    
    # Display results
    print("\n📈 Test Results:")
    print("=" * 20)
    
    if stftsc_result:
        print("🛤️ STFTSC Method:")
        print(f"   Estimated RPM: {stftsc_result['estimated_rpm']:.1f}")
        print(f"   True Average RPM: {stftsc_result['true_avg_rpm']:.1f}")
        print(f"   Absolute Error: {stftsc_result['error']:.1f} RPM")
        print(f"   Percentage Error: {stftsc_result['percentage_error']:.2f}%")
    else:
        print("❌ STFTSC method failed")
    
    # Create visualization
    create_visualization(audio_data, true_rpm_series, true_freq_series, stftsc_result, 44100)
    
    # Save results
    results = {
        'simulation_params': {
            'duration': 5.0,
            'sample_rate': 44100,
            'rpm_range': [0, 5000],
            'engine_orders': [1.0, 2.0, 3.0, 4.0]
        },
        'stftsc_result': stftsc_result
    }
    
    with open('simple_engine_test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: simple_engine_test_results.json")
    print("✅ Simple engine test completed!")

if __name__ == "__main__":
    main() 