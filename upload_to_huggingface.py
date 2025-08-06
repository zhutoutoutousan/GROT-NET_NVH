#!/usr/bin/env python3
"""
Script to upload the ultra-compact transformer model to Hugging Face Hub
"""

import os
from huggingface_hub import HfApi, create_repo
import torch

def upload_model_to_hf():
    """Upload the model to Hugging Face Hub"""
    
    # Model file path
    model_path = "best_ultra_compact_model.pth"
    
    if not os.path.exists(model_path):
        print(f"❌ Model file {model_path} not found!")
        return False
    
    # Model metadata
    model_name = "grot-net-ultra-compact-rpm-estimator"
    repo_name = f"fxxkingusername/{model_name}"
    
    # Model description
    model_description = """
# Ultra-Compact Transformer for RPM Estimation

This is the trained model for the ultra-compact transformer architecture designed for real-time RPM estimation from STFT spectrograms.

## Model Details
- **Architecture**: Ultra-compact transformer with frequency-domain attention
- **Input**: STFT spectrograms from engine audio
- **Output**: RPM values (0-2000 range)
- **Performance**: R² = 0.883, MAE = 104.09 RPM, MPE = 6.90%
- **Dataset**: HL-CEAD (High-Level Car Engine Audio Database)

## Usage

```python
import torch
from ultra_compact_genius import UltraCompactGeniusRPMArchitecture

# Load model
model = UltraCompactGeniusRPMArchitecture()
model.load_state_dict(torch.load('best_ultra_compact_model.pth'))
model.eval()

# Load from Hugging Face
from huggingface_hub import hf_hub_download
model_path = hf_hub_download(repo_id="zhutoutoutousan/grot-net-ultra-compact-rpm-estimator", 
                            filename="best_ultra_compact_model.pth")
model.load_state_dict(torch.load(model_path))
```

## Paper
For detailed methodology and results, see our paper: "Ultra-Compact Transformer Architecture for Real-Time RPM Estimation from STFT Spectrograms"

## Authors
- Tian Shao (t.shao@student.xu-university.de)
- Raad Bin Tareaf (r.bintareaf@xu-university.de)

## License
MIT License
"""
    
    try:
        # Initialize Hugging Face API
        api = HfApi()
        
        print(f"🚀 Uploading model to {repo_name}...")
        
        # Create repository if it doesn't exist
        try:
            create_repo(repo_name, private=False, exist_ok=True)
            print(f"✅ Repository {repo_name} created/verified")
        except Exception as e:
            print(f"⚠️ Repository creation issue: {e}")
        
        # Upload model file
        api.upload_file(
            path_or_fileobj=model_path,
            path_in_repo="best_ultra_compact_model.pth",
            repo_id=repo_name,
            commit_message="Add ultra-compact transformer model for RPM estimation"
        )
        print(f"✅ Model file uploaded successfully")
        
        # Upload README
        api.upload_file(
            path_or_fileobj=model_description.encode(),
            path_in_repo="README.md",
            repo_id=repo_name,
            commit_message="Add model documentation"
        )
        print(f"✅ README uploaded successfully")
        
        # Upload model card
        model_card = """
---
language: en
tags:
- audio
- rpm-estimation
- transformer
- automotive
- nvh
license: mit
datasets:
- hl-cead
metrics:
- r2_score
- mae
- mpe
---

# Ultra-Compact Transformer for RPM Estimation

This model implements an ultra-compact transformer architecture for real-time RPM estimation from STFT spectrograms.

## Model Information

- **Model Type**: Transformer-based regression
- **Input Format**: STFT spectrograms (frequency domain features)
- **Output Format**: RPM values (0-2000 range)
- **Model Size**: ~3.5MB
- **Framework**: PyTorch

## Performance Metrics

- **R² Score**: 0.883
- **Mean Absolute Error**: 104.09 RPM
- **Mean Percentage Error**: 6.90%

## Usage Example

```python
import torch
from ultra_compact_genius import UltraCompactGeniusRPMArchitecture

# Initialize model
model = UltraCompactGeniusRPMArchitecture()

# Load trained weights
model.load_state_dict(torch.load('best_ultra_compact_model.pth'))
model.eval()

# Process audio (example)
# audio_features = extract_stft_features(audio_signal)
# rpm_prediction = model(audio_features)
```

## Citation

```bibtex
@article{shao2024ultra,
  title={Ultra-Compact Transformer Architecture for Real-Time RPM Estimation from STFT Spectrograms},
  author={Shao, Tian and Tareaf, Raad Bin},
  journal={arXiv preprint},
  year={2024}
}
```

## License

MIT License - see LICENSE file for details.
"""
        
        api.upload_file(
            path_or_fileobj=model_card.encode(),
            path_in_repo="model-card.md",
            repo_id=repo_name,
            commit_message="Add model card"
        )
        print(f"✅ Model card uploaded successfully")
        
        print(f"\n🎉 Model successfully uploaded to: https://huggingface.co/{repo_name}")
        print(f"📁 Model file: https://huggingface.co/{repo_name}/blob/main/best_ultra_compact_model.pth")
        
        return True
        
    except Exception as e:
        print(f"❌ Error uploading model: {e}")
        return False

if __name__ == "__main__":
    success = upload_model_to_hf()
    if success:
        print("\n✅ Upload completed successfully!")
    else:
        print("\n❌ Upload failed!") 