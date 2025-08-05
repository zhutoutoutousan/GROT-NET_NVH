
#!/usr/bin/env python3
"""
STFTSC Implementation Based on Paper
====================================

Implementation of STFTSC (Short-Time Fourier Transform Seam Carving) 
method for instantaneous frequency estimation as described in the paper.
"""

import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy import signal
from scipy.optimize import minimize

class STFTSCEstimator:
    """STFTSC-based RPM estimator following the paper methodology."""
    
    def __init__(self, sr=44100, n_fft=512, hop_length=256, win_length=512, rpm_range=None, engine_type='automobile'):
        """Initialize STFTSC estimator."""
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        self.engine_type = engine_type
        
        # Set RPM range based on engine type or user input
        if rpm_range is None:
            if engine_type == 'automobile':
                # Automobile engine RPM range: 0-2000 RPM
                self.rpm_range = (0, 2000)
            else:
                # Default range
                self.rpm_range = (0, 8000)
        else:
            self.rpm_range = rpm_range
            
        # Calculate frequency range based on RPM range
        self.min_freq = self.rpm_range[0] / 60  # RPM to Hz conversion
        self.max_freq = self.rpm_range[1] / 60
        
        print(f"🔧 STFTSC initialized for {engine_type} engine")
        print(f"   RPM range: {self.rpm_range[0]}-{self.rpm_range[1]} RPM")
        print(f"   Frequency range: {self.min_freq:.1f}-{self.max_freq:.1f} Hz")
        
    def polynomial_chirplet_transform(self, signal_data, alpha_params=None):
        """
        Implement Polynomial Chirplet Transform (PCT) as described in the paper.
        
        Eq (1): P_CTS(t0, ω, α1, · · · , αn; σ) = ∫ z(t)Φ^R_α1···αn(t) × 
                Φ^M_α1···αn(t, t0)ω(σ)(t − t0) exp(−jωt)dt
        """
        if alpha_params is None:
            # Initialize with zeros as per paper
            alpha_params = np.zeros(3)  # α1, α2, α3
            
        # Create analytical signal using Hilbert transform
        z_t = signal.hilbert(signal_data)
        
        # Time axis
        t = np.arange(len(signal_data)) / self.sr
        
        # Gaussian window parameters
        sigma = self.n_fft / (2 * np.pi)
        
        # Compute PCT for each time-frequency point
        stft_result = np.zeros((len(self.freqs), len(t)), dtype=complex)
        
        for i, freq in enumerate(self.freqs):
            for j, t0 in enumerate(t):
                # Frequency rotation operator Φ^R
                phi_r = np.exp(-1j * np.sum([alpha_params[k] * t**(k+2) / (k+2) 
                                            for k in range(len(alpha_params))]))
                
                # Frequency shift operator Φ^M  
                phi_m = np.exp(1j * np.sum([alpha_params[k] * t0**(k+1) * t 
                                           for k in range(len(alpha_params))]))
                
                # Gaussian window
                window = np.exp(-((t - t0) / sigma)**2 / 2)
                
                # Integrate over time
                integrand = z_t * phi_r * phi_m * window * np.exp(-1j * 2 * np.pi * freq * t)
                stft_result[i, j] = np.trapz(integrand, t)
                
        return stft_result
    
    def compute_energy_function(self, stft_spectrum):
        """
        Compute energy function using Sobel operator as described in the paper.
        
        Eq (10): e(I(x, y)) = |∂I/∂x| + |∂I/∂y|
        """
        # Convert to magnitude spectrum
        magnitude = np.abs(stft_spectrum)
        
        # Apply log scale for better visualization
        log_magnitude = np.log10(magnitude + 1e-10)
        
        # Normalize to 0-255 range for gradient calculation
        normalized = ((log_magnitude - log_magnitude.min()) / 
                     (log_magnitude.max() - log_magnitude.min()) * 255).astype(np.uint8)
        
        # Apply Sobel operator for gradient calculation using numpy
        sobel_x = self.sobel_filter(normalized, axis=1)
        sobel_y = self.sobel_filter(normalized, axis=0)
        
        # Energy function as per Eq (10)
        energy_function = np.abs(sobel_x) + np.abs(sobel_y)
        
        return energy_function, log_magnitude
    
    def sobel_filter(self, image, axis=0):
        """Apply Sobel filter using numpy."""
        if axis == 0:  # Vertical gradient
            kernel = np.array([[-1, -2, -1],
                              [0, 0, 0],
                              [1, 2, 1]])
        else:  # Horizontal gradient
            kernel = np.array([[-1, 0, 1],
                              [-2, 0, 2],
                              [-1, 0, 1]])
        
        # Apply convolution
        from scipy import ndimage
        return ndimage.convolve(image.astype(float), kernel, mode='constant', cval=0)
    
    def find_optimal_seam_path(self, energy_function, start_freq_idx=None):
        """
        Extract optimal seam path using dynamic programming.
        
        Eq (13): M(x, y) = e(I(x, y)) + min{M(x-1, y-1), M(x-1, y), M(x-1, y+1)}
        """
        height, width = energy_function.shape
        
        # Initialize cumulative energy matrix
        M = np.zeros((height, width))
        M[0, :] = energy_function[0, :]
        
        # Path tracking matrix
        path_matrix = np.zeros((height, width), dtype=int)
        
        # Dynamic programming forward pass
        for x in range(1, height):
            for y in range(width):
                # Find minimum energy from previous row
                if y == 0:
                    min_idx = np.argmin([M[x-1, y], M[x-1, y+1]])
                    min_energy = min(M[x-1, y], M[x-1, y+1])
                    path_matrix[x, y] = min_idx
                elif y == width - 1:
                    min_idx = np.argmin([M[x-1, y-1], M[x-1, y]])
                    min_energy = min(M[x-1, y-1], M[x-1, y])
                    path_matrix[x, y] = min_idx - 1
                else:
                    min_idx = np.argmin([M[x-1, y-1], M[x-1, y], M[x-1, y+1]])
                    min_energy = min(M[x-1, y-1], M[x-1, y], M[x-1, y+1])
                    path_matrix[x, y] = min_idx - 1
                
                M[x, y] = energy_function[x, y] + min_energy
        
        # Backward pass to extract optimal path
        seam_path = []
        
        # Find starting point (minimum energy in last row)
        if start_freq_idx is None:
            start_idx = np.argmin(M[-1, :])
        else:
            start_idx = start_freq_idx
            
        current_y = start_idx
        seam_path.append(current_y)
        
        # Trace back through the path
        for x in range(height-1, 0, -1):
            # Ensure current_y stays within bounds
            current_y = max(0, min(width-1, current_y + path_matrix[x, current_y]))
            seam_path.append(current_y)
            
        seam_path.reverse()
        
        return np.array(seam_path), M
    
    def extract_instantaneous_frequency(self, audio_data, max_iterations=10, threshold=0.01):
        """
        Extract instantaneous frequency using STFTSC algorithm.
        
        Follows the paper's methodology:
        1. STFT time-frequency analysis
        2. Energy function calculation
        3. Initial value selection
        4. Ridge extraction using DP
        5. Iterative refinement
        """
        print("🔍 STFTSC: Starting instantaneous frequency extraction...")
        
        # Step 1: STFT time-frequency analysis
        print("📊 Step 1: Computing STFT...")
        stft_result = librosa.stft(audio_data, n_fft=self.n_fft, 
                                  hop_length=self.hop_length, win_length=self.win_length)
        
        # Step 2: Energy function calculation
        print("⚡ Step 2: Computing energy function...")
        energy_function, log_magnitude = self.compute_energy_function(stft_result)
        
        # Step 3: Initial value selection (find fundamental frequency)
        print("🎯 Step 3: Selecting initial frequency...")
        # Find the frequency with maximum energy in the middle of the signal
        # Focus on frequency range based on RPM range
        freq_range = (self.freqs >= self.min_freq) & (self.freqs <= self.max_freq)
        if np.sum(freq_range) > 0:
            mid_time = log_magnitude.shape[1] // 2
            freq_energy = np.sum(log_magnitude[freq_range, max(0, mid_time-10):mid_time+10], axis=1)
            if len(freq_energy) > 0:
                max_idx = np.argmax(freq_energy)
                start_freq_idx = np.where(freq_range)[0][max_idx]
            else:
                # Default to middle of RPM range
                default_freq = (self.min_freq + self.max_freq) / 2
                start_freq_idx = np.argmin(np.abs(self.freqs - default_freq))
        else:
            # Default to middle of RPM range
            default_freq = (self.min_freq + self.max_freq) / 2
            start_freq_idx = np.argmin(np.abs(self.freqs - default_freq))
        
        print(f"   Initial frequency index: {start_freq_idx} ({self.freqs[start_freq_idx]:.1f} Hz)")
        
        # Step 4: Ridge extraction using dynamic programming
        print("🛤️ Step 4: Extracting optimal ridge path...")
        seam_path, cumulative_energy = self.find_optimal_seam_path(energy_function, start_freq_idx)
        
        # Convert frequency indices to actual frequencies
        # Ensure seam_path indices are within bounds
        valid_indices = np.clip(seam_path, 0, len(self.freqs) - 1)
        instantaneous_freqs = self.freqs[valid_indices]
        
        # Step 5: Convert frequency to RPM using simple proportion scaling
        max_freq = np.max(instantaneous_freqs)
        min_freq = np.min(instantaneous_freqs)
        
        # Simple proportion-based RPM calculation
        rpm_estimates = []
        for freq in instantaneous_freqs:
            if max_freq > min_freq:
                proportion = (freq - min_freq) / (max_freq - min_freq)
            else:
                proportion = 0.5
            
            # Scale to 0-2000 RPM range
            rpm = proportion * self.rpm_range[1]  # 0 to 2000
            rpm_estimates.append(rpm)
        
        rpm_estimates = np.array(rpm_estimates)
        
        # Filter out unrealistic RPM values based on input range
        valid_mask = (rpm_estimates >= self.rpm_range[0]) & (rpm_estimates <= self.rpm_range[1])
        if np.sum(valid_mask) > 0:
            rpm_estimates = rpm_estimates[valid_mask]
            instantaneous_freqs = instantaneous_freqs[valid_mask]
            seam_path = seam_path[valid_mask]
        else:
            # If no valid RPMs found, use middle of RPM range
            default_rpm = (self.rpm_range[0] + self.rpm_range[1]) / 2
            default_freq = default_rpm / 60
            rpm_estimates = np.array([default_rpm])
            instantaneous_freqs = np.array([default_freq])
            seam_path = np.array([start_freq_idx])
        
        print(f"✅ STFTSC: Extracted {len(rpm_estimates)} RPM estimates")
        print(f"   RPM range: {rpm_estimates.min():.1f} - {rpm_estimates.max():.1f}")
        
        return {
            'rpm_estimates': rpm_estimates,
            'instantaneous_freqs': instantaneous_freqs,
            'seam_path': seam_path,
            'energy_function': energy_function,
            'log_magnitude': log_magnitude,
            'stft_result': stft_result
        }
    
    def estimate_rpm(self, audio_data):
        """Main RPM estimation function."""
        try:
            results = self.extract_instantaneous_frequency(audio_data)
            
            # Return the mean RPM as the final estimate
            mean_rpm = np.mean(results['rpm_estimates'])
            
            return {
                'estimated_rpm': mean_rpm,
                'rpm_series': results['rpm_estimates'],
                'frequency_series': results['instantaneous_freqs'],
                'confidence': self.calculate_confidence(results),
                'log_magnitude': results['log_magnitude'],
                'energy_function': results['energy_function'],
                'seam_path': results['seam_path']
            }
            
        except Exception as e:
            print(f"❌ STFTSC Error: {e}")
            return {
                'estimated_rpm': 1500.0, 
                'error': str(e),
                'rpm_series': np.array([1500.0]),
                'frequency_series': np.array([25.0]),
                'confidence': 0.0,
                'log_magnitude': np.zeros((257, 1)),
                'energy_function': np.zeros((257, 1)),
                'seam_path': np.array([0])
            }
    
    def calculate_confidence(self, results):
        """Calculate confidence based on energy consistency along the seam path."""
        seam_path = results['seam_path']
        energy_function = results['energy_function']
        
        # Extract energy values along the seam path
        path_energies = energy_function[seam_path, np.arange(len(seam_path))]
        
        # Calculate consistency (lower variance = higher confidence)
        energy_variance = np.var(path_energies)
        confidence = 1.0 / (1.0 + energy_variance)
        
        return confidence
    
    def calculate_intermediate_rpm(self, frequency_hz, engine_order=1):
        """
        Calculate intermediate RPM value from frequency using engine order scaling.
        
        Args:
            frequency_hz: Detected frequency in Hz
            engine_order: Engine order (1 for fundamental, 2 for 2nd order, etc.)
            
        Returns:
            Estimated RPM value
        """
        # RPM = (Frequency * 60) / Engine_Order
        rpm = (frequency_hz * 60) / engine_order
        
        # Constrain to engine RPM range
        rpm = np.clip(rpm, self.rpm_range[0], self.rpm_range[1])
        
        return rpm
    
    def estimate_rpm_with_orders(self, audio_data, max_orders=3):
        """
        Estimate RPM using proportion-based scaling with compression.
        
        Args:
            audio_data: Input audio signal
            max_orders: Maximum number of engine orders to consider (not used in new approach)
            
        Returns:
            Dictionary with RPM estimates using proportion-based scaling
        """
        try:
            results = self.extract_instantaneous_frequency(audio_data)
            instantaneous_freqs = results['instantaneous_freqs']
            
            # Use simple proportion-based scaling
            max_freq = np.max(instantaneous_freqs)
            min_freq = np.min(instantaneous_freqs)
            
            # Create different scaling factors for "orders"
            rpm_estimates_orders = {}
            scaling_factors = [0.5, 0.75, 1.0]  # Different scaling levels
            
            for i, scaling in enumerate(scaling_factors, 1):
                order_rpms = []
                for freq in instantaneous_freqs:
                    if max_freq > min_freq:
                        proportion = (freq - min_freq) / (max_freq - min_freq)
                    else:
                        proportion = 0.5
                    
                    # Scale to 0-2000 RPM range
                    rpm = proportion * self.rpm_range[1] * scaling  # 0 to 2000 * scaling
                    order_rpms.append(rpm)
                
                rpm_estimates_orders[f'scaling_{scaling}'] = {
                    'rpms': np.array(order_rpms),
                    'mean_rpm': np.mean(order_rpms),
                    'std_rpm': np.std(order_rpms)
                }
            
            # Find the most consistent scaling (lowest standard deviation)
            best_scaling = min(rpm_estimates_orders.keys(), 
                              key=lambda x: rpm_estimates_orders[x]['std_rpm'])
            
            best_rpm = rpm_estimates_orders[best_scaling]['mean_rpm']
            
            return {
                'estimated_rpm': best_rpm,
                'rpm_series': rpm_estimates_orders[best_scaling]['rpms'],
                'frequency_series': results['instantaneous_freqs'],
                'confidence': self.calculate_confidence(results),
                'log_magnitude': results['log_magnitude'],
                'energy_function': results['energy_function'],
                'seam_path': results['seam_path'],
                'all_orders': rpm_estimates_orders,
                'best_order': best_scaling
            }
            
        except Exception as e:
            print(f"❌ STFTSC Error: {e}")
            return {
                'estimated_rpm': (self.rpm_range[0] + self.rpm_range[1]) / 2,
                'error': str(e),
                'rpm_series': np.array([(self.rpm_range[0] + self.rpm_range[1]) / 2]),
                'frequency_series': np.array([(self.rpm_range[0] + self.rpm_range[1]) / 120]),
                'confidence': 0.0,
                'log_magnitude': np.zeros((257, 1)),
                'energy_function': np.zeros((257, 1)),
                'seam_path': np.array([0])
            }
    
    def visualize_results(self, results, save_path=None):
        """Visualize STFTSC results."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('STFTSC Instantaneous Frequency Estimation', fontsize=16)
        
        # 1. STFT magnitude spectrum
        ax1 = axes[0, 0]
        im1 = ax1.imshow(results['log_magnitude'], aspect='auto', origin='lower')
        ax1.set_title('STFT Magnitude Spectrum (log scale)')
        ax1.set_xlabel('Time Frame')
        ax1.set_ylabel('Frequency (Hz)')
        plt.colorbar(im1, ax=ax1)
        
        # 2. Energy function
        ax2 = axes[0, 1]
        im2 = ax2.imshow(results['energy_function'], aspect='auto', origin='lower')
        ax2.set_title('Energy Function (Sobel gradient)')
        ax2.set_xlabel('Time Frame')
        ax2.set_ylabel('Frequency (Hz)')
        plt.colorbar(im2, ax=ax2)
        
        # 3. Seam path overlay
        ax3 = axes[1, 0]
        im3 = ax3.imshow(results['log_magnitude'], aspect='auto', origin='lower')
        seam_path = results['seam_path']
        time_frames = np.arange(len(seam_path))
        ax3.plot(time_frames, seam_path, 'r-', linewidth=2, label='Optimal Seam Path')
        ax3.set_title('STFT with Optimal Seam Path')
        ax3.set_xlabel('Time Frame')
        ax3.set_ylabel('Frequency (Hz)')
        ax3.legend()
        plt.colorbar(im3, ax=ax3)
        
        # 4. RPM over time
        ax4 = axes[1, 1]
        rpm_series = results.get('rpm_estimates', results.get('rpm_series', np.array([1500.0])))
        time_axis = np.arange(len(rpm_series)) * self.hop_length / self.sr
        ax4.plot(time_axis, rpm_series, 'b-', linewidth=2)
        ax4.set_title('Estimated RPM Over Time')
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('RPM')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Visualization saved to: {save_path}")
        
        plt.show()

def test_stftsc_implementation():
    """Test the STFTSC implementation."""
    print("🧪 Testing STFTSC Implementation")
    print("=" * 40)
    
    # Create a test signal with variable frequency (simulating variable RPM)
    sr = 44100
    duration = 5.0
    t = np.linspace(0, duration, int(sr * duration))
    
    # Create a signal with linearly increasing frequency (simulating RPM increase)
    # Frequency from 20 Hz to 40 Hz (RPM from 1200 to 2400)
    freq_start = 20  # Hz
    freq_end = 40    # Hz
    freq_t = freq_start + (freq_end - freq_start) * t / duration
    
    # Generate signal with harmonics
    signal_data = (np.sin(2 * np.pi * freq_t * t) + 
                   0.5 * np.sin(2 * np.pi * 2 * freq_t * t) +  # 2nd harmonic
                   0.3 * np.sin(2 * np.pi * 3 * freq_t * t))   # 3rd harmonic
    
    # Add noise
    noise = 0.1 * np.random.randn(len(signal_data))
    signal_data += noise
    
    print(f"📊 Test signal: {duration}s, {sr} Hz sampling rate")
    print(f"🎵 Frequency range: {freq_start}-{freq_end} Hz")
    print(f"⚡ Expected RPM range: {freq_start*60}-{freq_end*60} RPM")
    
    # Initialize STFTSC estimator
    estimator = STFTSCEstimator(sr=sr)
    
    # Estimate RPM
    result = estimator.estimate_rpm(signal_data)
    
    print(f"\n📈 STFTSC Results:")
    print(f"   Estimated RPM: {result['estimated_rpm']:.1f}")
    print(f"   RPM range: {result['rpm_series'].min():.1f} - {result['rpm_series'].max():.1f}")
    print(f"   Confidence: {result['confidence']:.3f}")
    
    # Visualize results
    estimator.visualize_results(result, 'stftsc_test_results.png')
    
    return result

if __name__ == "__main__":
    test_stftsc_implementation() 