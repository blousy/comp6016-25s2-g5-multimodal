# setup.py
"""
Setup script for Magma-8B LoRA Navigation Training
"""

import os
import subprocess
import sys
from pathlib import Path

def install_requirements():
    """Install required packages."""
    requirements = [
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "transformers>=4.35.0",
        "peft>=0.6.0",
        "datasets>=2.14.0",
        "accelerate>=0.24.0",
        "bitsandbytes>=0.41.0",
        "wandb>=0.16.0",
        "habitat-sim",
        "habitat-lab",
        "numpy>=1.24.0",
        "pillow>=9.0.0",
        "matplotlib>=3.6.0",
        "tqdm>=4.64.0",
        "gtts>=2.3.0",
        "ipython>=8.0.0"
    ]
    
    print("Installing Python requirements...")
    for req in requirements:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", req])
            print(f"✅ Installed {req}")
        except subprocess.CalledProcessError:
            print(f"❌ Failed to install {req}")

def setup_directories():
    """Create necessary directories."""
    dirs = [
        "data/scene_datasets/mp3d",
        "data/datasets/pointnav/mp3d/v1/train",
        "data/datasets/pointnav/mp3d/v1/val", 
        "data/datasets/pointnav/mp3d/v1/test",
        "habitat_configs",
        "model_cache",
        "navigation_robot_results",
        "navigation_lora_checkpoints"
    ]
    
    print("Creating directories...")
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created {dir_path}")

def check_gpu():
    """Check GPU availability."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✅ GPU available: {gpu_name} (Count: {gpu_count})")
            
            # Check memory
            memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"   GPU Memory: {memory_gb:.1f} GB")
            
            if memory_gb < 16:
                print("⚠️  Warning: Less than 16GB GPU memory. Consider reducing batch size.")
            
            return True
        else:
            print("❌ No GPU available. Training will be slow on CPU.")
            return False
    except ImportError:
        print("❌ PyTorch not installed")
        return False

def download_data_instructions():
    """Provide data download instructions."""
    instructions = """
    📥 Data Download Instructions:
    
    1. Register and agree to MP3D terms at: https://niessner.github.io/Matterport/
    
    2. Install Habitat datasets:
       python -m habitat_sim.utils.datasets_download --dataset mp3d --data-path ./data/
       python -m habitat_sim.utils.datasets_download --dataset pointnav_mp3d --data-path ./data/
    
    3. Verify data structure:
       data/
       ├── scene_datasets/mp3d/
       │   ├── *.glb (90 scene files)
       │   └── *.navmesh
       └── datasets/pointnav/mp3d/v1/
           ├── train/train.json.gz (61,424 episodes)
           ├── val/val.json.gz (6,817 episodes)
           └── test/test.json.gz (4,632 episodes)
    
    4. Test installation:
       python -c "import habitat; print('Habitat installed successfully')"
    """
    print(instructions)

def main():
    """Main setup function."""
    print("🚀 Setting up Magma-8B LoRA Navigation Training Environment")
    print("=" * 60)
    
    # Install requirements
    install_requirements()
    
    # Setup directories
    setup_directories()
    
    # Check GPU
    gpu_available = check_gpu()
    
    # Data download instructions
    download_data_instructions()
    
    print("\n✅ Setup completed!")
    
    if gpu_available:
        print("\n🚀 Ready to start training! Run:")
        print("   python training_utilities.py pipeline")
    else:
        print("\n⚠️  GPU not available. Training will be very slow.")

if __name__ == "__main__":
    main()

# ---------------------------------------------------------------------------
# requirements.txt content
# ---------------------------------------------------------------------------

REQUIREMENTS_TXT = """
# Core ML libraries
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.35.0
peft>=0.6.0
datasets>=2.14.0
accelerate>=0.24.0
bitsandbytes>=0.41.0

# Habitat ecosystem
habitat-sim
habitat-lab

# Utilities
numpy>=1.24.0
pillow>=9.0.0
matplotlib>=3.6.0
tqdm>=4.64.0
wandb>=0.16.0
gtts>=2.3.0
ipython>=8.0.0

# Optional: For Jupyter notebooks
jupyter>=1.0.0
notebook>=6.5.0
"""

# Save requirements.txt
with open("requirements.txt", "w") as f:
    f.write(REQUIREMENTS_TXT.strip())

print("📄 Created requirements.txt")