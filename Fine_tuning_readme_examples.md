# Magma-8B LoRA Training for Indoor Navigation

This repository contains a complete pipeline for fine-tuning Microsoft's Magma-8B vision-language model using LoRA (Low-Rank Adaptation) on the Habitat-MP3D dataset for indoor navigation tasks.

## 🎯 Overview

The system builds upon your existing `NavigationSupportSystem` to create a specialized navigation assistant that can:

- Generate contextual navigation instructions from visual input
- Plan step-by-step actions for reaching navigation goals  
- Provide safety warnings and hazard detection
- Adapt to indoor environments through MP3D scene training

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone and setup
git clone <your-repo>
cd magma-navigation-training

# Run setup script
python setup.py

# Or install manually
pip install -r requirements.txt
```

### 2. Data Preparation

```bash
# Download MP3D dataset (requires agreement to terms)
python -m habitat_sim.utils.datasets_download --dataset mp3d --data-path ./data/
python -m habitat_sim.utils.datasets_download --dataset pointnav_mp3d --data-path ./data/

# Validate data
python training_utilities.py validate-data
```

### 3. Training

```bash
# Quick training with defaults
python training_utilities.py train --episodes 1000 --epochs 3

# Or run full pipeline
python training_utilities.py pipeline

# Custom training with config
python magma_lora_training.py
```

### 4. Evaluation

```bash
# Evaluate trained model
python training_utilities.py evaluate --model-path ./navigation_lora_checkpoints/final_model
```

## 📁 Project Structure

```
magma-navigation-training/
├── magma_lora_training.py          # Main training script
├── training_utilities.py           # Helper utilities and CLI
├── navigation_robot_system.py      # Your original system (base)
├── setup.py                        # Environment setup
├── requirements.txt                # Python dependencies
├── habitat_configs/
│   └── pointnav_mp3d.yaml         # Habitat environment config
├── data/                           # Dataset directory
│   ├── scene_datasets/mp3d/        # 3D scene files (.glb)
│   └── datasets/pointnav/mp3d/v1/  # Navigation episodes
├── navigation_lora_checkpoints/    # Training checkpoints
└── navigation_robot_results/       # Results and logs
```

## 🔧 Configuration

### Training Configuration

Key parameters in `NavigationTrainingConfig`:

```python
config = NavigationTrainingConfig(
    # Model settings
    model_name="microsoft/Magma-8B",
    
    # LoRA settings
    lora_r=16,                    # Rank
    lora_alpha=32,                # Alpha scaling
    lora_dropout=0.1,             # Dropout
    
    # Training settings
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    
    # Dataset settings
    num_episodes=1000,            # Episodes to generate
    max_episode_steps=500,        # Steps per episode
)
```

### Hardware Requirements

**Minimum:**
- GPU: 16GB VRAM (RTX 4080/V100)
- RAM: 32GB system memory
- Storage: 100GB free space

**Recommended:**
- GPU: 24GB+ VRAM (RTX 4090/A100)
- RAM: 64GB system memory
- Storage: 200GB+ SSD

## 🎮 Usage Examples

### Basic Training

```python
from magma_lora_training import NavigationLoRATrainer, NavigationTrainingConfig

# Configure training
config = NavigationTrainingConfig(
    num_episodes=2000,
    num_train_epochs=5,
    per_device_train_batch_size=1,
    learning_rate=1e-4
)

# Train model
trainer = NavigationLoRATrainer(config)
trainer.train()
```

### Custom Data Generation

```python
from magma_lora_training import HabitatNavigationDataset

# Create dataset with custom parameters
dataset = HabitatNavigationDataset(
    config=config,
    processor=processor,
    split="train", 
    max_episodes=500
)

# Access individual samples
sample = dataset[0] 
# {'image': PIL.Image, 'conversation': [...], 'episode_id': 0, 'step': 5}
```

### Model Inference

```python
from training_utilities import NavigationInferenceTester

# Load trained model
tester = NavigationInferenceTester("./navigation_lora_checkpoints/final_model")

# Test on new images
test_scenarios = [
    {
        "id": "kitchen_nav",
        "image": kitchen_image,
        "task": "Navigate to the refrigerator"
    }
]

results = tester.test_navigation_scenarios(test_scenarios)
```

### Integration with Original System

```python
from navigation_robot_system import NavigationRobotSystem
from peft import PeftModel

# Load your trained LoRA model
base_model = AutoModelForCausalLM.from_pretrained("microsoft/Magma-8B")
lora_model = PeftModel.from_pretrained(base_model, "./navigation_lora_checkpoints/final_model")

# Replace model in your existing system
nrs = NavigationRobotSystem()
nrs.model = lora_model

# Use as before
description, guidance, plan, hazards, audio = nrs.process_task(
    image_source="webcam",
    task="Go to the kitchen and find the microwave",
    save_results=True
)
```

## 📊 Training Monitoring

The system integrates with Weights & Biases for training monitoring:

```python
# Enable W&B logging
config.use_wandb = True
config.wandb_project = "magma-navigation-experiments"
config.wandb_run_name = "experiment-v1"
```

Key metrics tracked:
- Training/validation loss
- Learning rate schedule  
- GPU memory usage
- Episode generation statistics
- Model parameter counts

## 🧪 Evaluation Metrics

The system evaluates model performance on:

1. **Navigation Quality**: Does the response contain directional guidance?
2. **Safety Awareness**: Are hazards and obstacles mentioned?
3. **Landmark Usage**: Are environmental features referenced?
4. **Actionability**: Can instructions be followed?
5. **Response Length**: Appropriate level of detail?

## 🔬 Advanced Usage

### Custom Action Mappings

```python
def custom_action_to_instruction(action, observations, goal_position):
    """Create domain-specific navigation instructions."""
    if action == 1:  # MOVE_FORWARD
        return "Move forward carefully, watching for obstacles ahead."
    # ... custom logic
    
# Use in dataset generation
dataset._action_to_instruction = custom_action_to_instruction
```

### Multi-GPU Training

```python
# Configure for distributed training
config = NavigationTrainingConfig(
    per_device_train_batch_size=1,    # Reduce per device
    gradient_accumulation_steps=32,    # Increase accumulation
    dataloader_num_workers=4,
)

# Launch with torchrun
# torchrun --nproc_per_node=4 magma_lora_training.py
```

### Custom LoRA Targets

```python
config = NavigationTrainingConfig(
    target_modules=[
        "q_proj", "v_proj", "k_proj", "o_proj",  # Attention
        "gate_proj", "up_proj", "down_proj",     # MLP
        "vision_model.encoder.layers.*.self_attn.q_proj"  # Vision layers
    ]
)
```

## 🛠️ Troubleshooting

### Common Issues

**GPU Memory Errors:**
```bash
# Reduce batch size and increase gradient accumulation
--batch-size 1 --gradient-accumulation-steps 16
```

**Habitat Installation Issues:**
```bash
# Install from conda-forge
conda install habitat-sim habitat-lab -c conda-forge -c aihabitat

# Or build from source
git clone https://github.com/facebookresearch/habitat-sim.git
cd habitat-sim && pip install -e .
```

**Data Download Problems:**
- Ensure you've agreed to MP3D terms of use
- Check network connectivity for large downloads
- Verify disk space (dataset is ~15GB)

### Performance Optimization

1. **Use mixed precision**: Enabled by default with `bf16=True`
2. **Gradient checkpointing**: Reduces memory usage
3. **Smaller LoRA rank**: Try `lora_r=8` for faster training
4. **Episode caching**: Pre-generate and cache episodes

## 📈 Expected Results

After training, you should see:

- **Training Loss**: Decreases from ~3.0 to ~1.5
- **Validation Loss**: Stabilizes around 1.8-2.2  
- **Quality Scores**: 70-85/100 on navigation tasks
- **Inference Speed**: ~2-3 seconds per response

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/improvement`)
3. Add tests for new functionality
4. Submit pull request with detailed description

## 📜 License

This project extends your original `NavigationSupportSystem` and includes:
- Apache 2.0 License (Habitat components)
- MIT License (Training utilities)
- Check individual model licenses (Magma-8B)

## 📚 References

- [Magma-8B Model](https://huggingface.co/microsoft/Magma-8B)
- [Habitat Documentation](https://aihabitat.org/docs/)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [MP3D Dataset](https://niessner.github.io/Matterport/)

## 🆘 Support

For issues and questions:
1. Check existing GitHub issues
2. Review troubleshooting section
3. Create detailed bug report with:
   - System specifications
   - Error logs
   - Steps to reproduce

---

**Note**: This training setup requires significant computational resources. Consider using cloud platforms (AWS, GCP, Azure) with appropriate GPU instances for large-scale training.