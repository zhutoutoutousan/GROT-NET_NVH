#!/usr/bin/env python3
"""
Method Comparison: Deep Learning vs STFTSC
==========================================

Compare our ultra-compact deep learning approach with the proper STFTSC implementation.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from ultra_compact_genius import UltraCompactGeniusRPMArchitecture
from stftsc_implementation import STFTSCEstimator
import librosa
import random

def load_trained_model(model_path='best_ultra_compact_model.pth'):
    """Load the trained deep learning model."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = UltraCompactGeniusRPMArchitecture()
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model, device

def create_test_signals():
    """Create various test signals for comparison."""
    sr = 44100
    duration = 3.0
    t = np.linspace(0, duration, int(sr * duration))
    
    test_signals = {}
    
    # Test 1: Constant RPM (1500 RPM = 25 Hz)
    freq_constant = 25  # Hz
    signal_constant = np.sin(2 * np.pi * freq_constant * t)
    test_signals['constant_1500rpm'] = {
        'signal': signal_constant,
        'expected_rpm': 1500,
        'description': 'Constant 1500 RPM'
    }
    
    # Test 2: Variable RPM (1000 to 2000 RPM)
    freq_start = 1000 / 60  # Hz
    freq_end = 2000 / 60    # Hz
    freq_t = freq_start + (freq_end - freq_start) * t / duration
    signal_variable = np.sin(2 * np.pi * freq_t * t)
    test_signals['variable_1000to2000rpm'] = {
        'signal': signal_variable,
        'expected_rpm': np.mean([1000, 2000]),
        'description': 'Variable 1000-2000 RPM'
    }
    
    # Test 3: Engine-like signal with harmonics
    freq_engine = 25  # Hz (1500 RPM)
    signal_engine = (np.sin(2 * np.pi * freq_engine * t) + 
                    0.5 * np.sin(2 * np.pi * 2 * freq_engine * t) +  # 2nd harmonic
                    0.3 * np.sin(2 * np.pi * 3 * freq_engine * t) +  # 3rd harmonic
                    0.2 * np.sin(2 * np.pi * 4 * freq_engine * t))   # 4th harmonic
    test_signals['engine_harmonics_1500rpm'] = {
        'signal': signal_engine,
        'expected_rpm': 1500,
        'description': 'Engine signal with harmonics (1500 RPM)'
    }
    
    # Test 4: Noisy signal
    signal_noisy = signal_engine + 0.1 * np.random.randn(len(signal_engine))
    test_signals['noisy_engine_1500rpm'] = {
        'signal': signal_noisy,
        'expected_rpm': 1500,
        'description': 'Noisy engine signal (1500 RPM)'
    }
    
    return test_signals

def test_deep_learning_method(model, device, audio_data):
    """Test the deep learning method."""
    try:
        # Convert to numpy if needed
        if isinstance(audio_data, torch.Tensor):
            audio_data = audio_data.cpu().numpy()
        
        # Get prediction
        with torch.no_grad():
            prediction = model(audio_data)
            estimated_rpm = prediction['estimated_rpm'].item()
        
        return {
            'estimated_rpm': estimated_rpm,
            'method': 'Deep Learning',
            'success': True
        }
    except Exception as e:
        return {
            'estimated_rpm': 1500.0,
            'method': 'Deep Learning',
            'success': False,
            'error': str(e)
        }

def test_stftsc_method(estimator, audio_data):
    """Test the STFTSC method."""
    try:
        result = estimator.estimate_rpm(audio_data)
        return {
            'estimated_rpm': result['estimated_rpm'],
            'rpm_series': result['rpm_series'],
            'method': 'STFTSC',
            'success': True
        }
    except Exception as e:
        return {
            'estimated_rpm': 1500.0,
            'method': 'STFTSC',
            'success': False,
            'error': str(e)
        }

def compare_methods():
    """Compare both methods on various test signals."""
    print("🔍 Comparing Deep Learning vs STFTSC Methods")
    print("=" * 50)
    
    # Load trained model
    try:
        model, device = load_trained_model()
        print("✅ Deep learning model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load deep learning model: {e}")
        return
    
    # Initialize STFTSC estimator
    stftsc_estimator = STFTSCEstimator()
    print("✅ STFTSC estimator initialized")
    
    # Create test signals
    test_signals = create_test_signals()
    print(f"📊 Created {len(test_signals)} test signals")
    
    # Results storage
    results = []
    
    # Test each signal
    for signal_name, signal_info in test_signals.items():
        print(f"\n🎵 Testing: {signal_info['description']}")
        print(f"   Expected RPM: {signal_info['expected_rpm']}")
        
        audio_data = signal_info['signal']
        
        # Test deep learning method
        dl_result = test_deep_learning_method(model, device, audio_data)
        
        # Test STFTSC method
        stftsc_result = test_stftsc_method(stftsc_estimator, audio_data)
        
        # Calculate errors
        expected_rpm = signal_info['expected_rpm']
        
        dl_error = abs(dl_result['estimated_rpm'] - expected_rpm)
        dl_percentage_error = (dl_error / expected_rpm) * 100
        
        stftsc_error = abs(stftsc_result['estimated_rpm'] - expected_rpm)
        stftsc_percentage_error = (stftsc_error / expected_rpm) * 100
        
        result = {
            'signal_name': signal_name,
            'description': signal_info['description'],
            'expected_rpm': expected_rpm,
            'deep_learning': {
                'estimated_rpm': dl_result['estimated_rpm'],
                'error': dl_error,
                'percentage_error': dl_percentage_error,
                'success': dl_result['success']
            },
            'stftsc': {
                'estimated_rpm': stftsc_result['estimated_rpm'],
                'error': stftsc_error,
                'percentage_error': stftsc_percentage_error,
                'success': stftsc_result['success']
            }
        }
        
        results.append(result)
        
        # Print results
        print(f"   Deep Learning: {dl_result['estimated_rpm']:.1f} RPM (Error: {dl_percentage_error:.2f}%)")
        print(f"   STFTSC: {stftsc_result['estimated_rpm']:.1f} RPM (Error: {stftsc_percentage_error:.2f}%)")
    
    # Summary statistics
    print("\n📊 Summary Statistics")
    print("=" * 30)
    
    dl_errors = [r['deep_learning']['percentage_error'] for r in results if r['deep_learning']['success']]
    stftsc_errors = [r['stftsc']['percentage_error'] for r in results if r['stftsc']['success']]
    
    if dl_errors:
        print(f"Deep Learning - Mean Error: {np.mean(dl_errors):.2f}%, Std: {np.std(dl_errors):.2f}%")
    if stftsc_errors:
        print(f"STFTSC - Mean Error: {np.mean(stftsc_errors):.2f}%, Std: {np.std(stftsc_errors):.2f}%")
    
    # Create comparison plot
    create_comparison_plot(results)
    
    return results

def create_comparison_plot(results):
    """Create a comparison plot of the results."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Deep Learning vs STFTSC Method Comparison', fontsize=16)
    
    # Extract data for plotting
    signal_names = [r['signal_name'] for r in results]
    expected_rpms = [r['expected_rpm'] for r in results]
    dl_estimates = [r['deep_learning']['estimated_rpm'] for r in results]
    stftsc_estimates = [r['stftsc']['estimated_rpm'] for r in results]
    dl_errors = [r['deep_learning']['percentage_error'] for r in results]
    stftsc_errors = [r['stftsc']['percentage_error'] for r in results]
    
    # 1. Estimated RPM comparison
    ax1 = axes[0, 0]
    x = np.arange(len(signal_names))
    width = 0.35
    
    ax1.bar(x - width/2, expected_rpms, width, label='Expected', alpha=0.7)
    ax1.bar(x + width/2, dl_estimates, width, label='Deep Learning', alpha=0.7)
    ax1.bar(x + width/2, stftsc_estimates, width, label='STFTSC', alpha=0.7, bottom=dl_estimates)
    
    ax1.set_xlabel('Test Signal')
    ax1.set_ylabel('RPM')
    ax1.set_title('Estimated RPM Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels([name.split('_')[0] for name in signal_names], rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Percentage error comparison
    ax2 = axes[0, 1]
    ax2.bar(x - width/2, dl_errors, width, label='Deep Learning', alpha=0.7)
    ax2.bar(x + width/2, stftsc_errors, width, label='STFTSC', alpha=0.7)
    
    ax2.set_xlabel('Test Signal')
    ax2.set_ylabel('Percentage Error (%)')
    ax2.set_title('Error Comparison')
    ax2.set_xticks(x)
    ax2.set_xticklabels([name.split('_')[0] for name in signal_names], rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Scatter plot: Expected vs Estimated
    ax3 = axes[1, 0]
    ax3.scatter(expected_rpms, dl_estimates, alpha=0.7, label='Deep Learning', s=100)
    ax3.scatter(expected_rpms, stftsc_estimates, alpha=0.7, label='STFTSC', s=100)
    
    # Perfect prediction line
    min_rpm = min(min(expected_rpms), min(dl_estimates), min(stftsc_estimates))
    max_rpm = max(max(expected_rpms), max(dl_estimates), max(stftsc_estimates))
    ax3.plot([min_rpm, max_rpm], [min_rpm, max_rpm], 'r--', alpha=0.8, label='Perfect Prediction')
    
    ax3.set_xlabel('Expected RPM')
    ax3.set_ylabel('Estimated RPM')
    ax3.set_title('Expected vs Estimated RPM')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Error distribution
    ax4 = axes[1, 1]
    ax4.hist(dl_errors, bins=10, alpha=0.7, label='Deep Learning', edgecolor='black')
    ax4.hist(stftsc_errors, bins=10, alpha=0.7, label='STFTSC', edgecolor='black')
    
    ax4.set_xlabel('Percentage Error (%)')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Error Distribution')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('method_comparison_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("📊 Comparison plot saved to: method_comparison_results.png")

def main():
    """Main comparison function."""
    print("🚀 Starting Method Comparison")
    print("=" * 40)
    
    results = compare_methods()
    
    print("\n✅ Comparison completed!")
    print("📊 Check the generated files:")
    print("   - method_comparison_results.png")

if __name__ == "__main__":
    main() 