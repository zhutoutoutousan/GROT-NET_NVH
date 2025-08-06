# GROT-NET: Ultra-Compact Transformer for RPM Estimation

## Overview
This repository contains the implementation of GROT-NET (GAN-RNN-based Order Tracking Network) and its enhanced ultra-compact transformer architecture for automotive NVH (Noise, Vibration, and Harshness) analysis and real-time RPM estimation from engine audio.

## 🚀 Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Using the Pre-trained Model
```python
import torch
from ultra_compact_genius import UltraCompactGeniusRPMArchitecture

# Load model from Hugging Face
from huggingface_hub import hf_hub_download
model_path = hf_hub_download(repo_id="fxxkingusername/grot-net-ultra-compact-rpm-estimator", 
                            filename="best_ultra_compact_model.pth")

# Initialize and load model
model = UltraCompactGeniusRPMArchitecture()
model.load_state_dict(torch.load(model_path))
model.eval()

# Process audio (example)
# audio_features = extract_stft_features(audio_signal)
# rpm_prediction = model(audio_features)
```

### Training Your Own Model
```bash
# Preprocess data
python preprocess_simple.py

# Train the ultra-compact model
python train_ultra_compact.py

# Evaluate performance
python comprehensive_engine_comparison.py
```

## 📊 Model Performance

Our ultra-compact transformer architecture achieves state-of-the-art performance:

- **R² Score**: 0.883
- **Mean Absolute Error**: 104.09 RPM  
- **Mean Percentage Error**: 6.90%
- **Model Size**: ~3.5MB
- **Real-time Capable**: Yes

## 🏗️ Architecture

The ultra-compact transformer combines:
- **Frequency-domain attention mechanisms** for spectral pattern recognition
- **Temporal modeling** for sequence dependencies
- **Memory optimization** for embedded automotive systems
- **STFT-based feature extraction** from engine audio

## 📁 Project Structure

```
GROT-NET_NVH/
├── ultra_compact_genius.py          # Main model architecture
├── train_ultra_compact.py           # Training script
├── preprocess_simple.py             # Data preprocessing
├── comprehensive_engine_comparison.py # Evaluation and comparison
├── stftsc_implementation.py         # STFTSC baseline implementation
├── paper/                           # Research paper (LaTeX)
│   ├── TRANSFORMER_RPM_ESTIMATION.tex
│   └── NVH.bib
├── best_ultra_compact_model.pth     # Pre-trained model (3.5MB)
└── requirements.txt                 # Dependencies
```

## 🔬 Research Paper

Our complete methodology is documented in the IEEE-style paper:
- **Title**: "Ultra-Compact Transformer Architecture for Real-Time RPM Estimation from STFT Spectrograms"
- **Authors**: Tian Shao, Raad Bin Tareaf
- **Location**: `paper/TRANSFORMER_RPM_ESTIMATION.tex`

## 🤗 Hugging Face Model

The pre-trained model is available on Hugging Face:
- **Repository**: [fxxkingusername/grot-net-ultra-compact-rpm-estimator](https://huggingface.co/fxxkingusername/grot-net-ultra-compact-rpm-estimator)
- **Model Card**: Complete documentation and usage examples
- **Direct Download**: `best_ultra_compact_model.pth`

## 📈 Comparison Results

We compare our approach with traditional STFTSC method:

| Method | R² Score | MAE (RPM) | MPE (%) | Real-time |
|--------|----------|-----------|---------|-----------|
| **Our Transformer** | **0.883** | **104.09** | **6.90** | ✅ |
| STFTSC Baseline | 0.721 | 156.34 | 9.45 | ❌ |

## 🎯 Applications

- **Automotive NVH Analysis**: Engine condition monitoring
- **Real-time RPM Estimation**: Live engine performance tracking
- **Fault Diagnosis**: Engine anomaly detection
- **Research**: Audio-based mechanical analysis

## 📚 Dataset

We use the **HL-CEAD** (High-Level Car Engine Audio Database) dataset:
- Real engine audio recordings
- RPM range: 0-2000
- Multiple engine conditions and speeds
- STFT-processed frequency domain features

## 🛠️ Dependencies

Key dependencies include:
- PyTorch
- NumPy
- SciPy
- Librosa
- Matplotlib
- Hugging Face Hub

## 📄 License

MIT License - see LICENSE file for details.

## 👥 Authors

- **Tian Shao** (t.shao@student.xu-university.de)
- **Raad Bin Tareaf** (r.bintareaf@xu-university.de)

## 📖 Citation

```bibtex
@article{shao2024ultra,
  title={Ultra-Compact Transformer Architecture for Real-Time RPM Estimation from STFT Spectrograms},
  author={Shao, Tian and Tareaf, Raad Bin},
  journal={arXiv preprint},
  year={2024}
}
```

## 🔗 Links

- **GitHub Repository**: https://github.com/zhutoutoutousan/GROT-NET_NVH
- **Hugging Face Model**: https://huggingface.co/fxxkingusername/grot-net-ultra-compact-rpm-estimator
- **Research Paper**: `paper/TRANSFORMER_RPM_ESTIMATION.tex`
