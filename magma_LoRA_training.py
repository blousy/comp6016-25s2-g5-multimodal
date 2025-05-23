# magma_lora_training.py
"""
LoRA fine-tuning of Magma-8B on Habitat-MP3D dataset for indoor navigation tasks.
This script adapts the vision-language model for better navigation-specific performance.
"""

import json
import os
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import habitat
from habitat import Config, make_dataset
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    PeftModel,
    prepare_model_for_kbit_training
)
import wandb
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration and Setup
# ---------------------------------------------------------------------------

@dataclass
class NavigationTrainingConfig:
    """Configuration for navigation training setup."""
    
    # Model settings
    model_name: str = "microsoft/Magma-8B"
    cache_dir: str = "./model_cache"
    
    # LoRA configuration
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "v_proj", "k_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    
    # Training parameters
    output_dir: str = "./navigation_lora_checkpoints"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    
    # Dataset settings
    habitat_config_path: str = "./habitat_configs/pointnav_mp3d.yaml"
    max_episode_steps: int = 500
    num_episodes: int = 1000
    train_split: float = 0.8
    
    # Navigation-specific settings
    max_sequence_length: int = 512
    image_size: Tuple[int, int] = (224, 224)
    
    # Logging
    use_wandb: bool = True
    wandb_project: str = "magma-navigation-lora"
    wandb_run_name: Optional[str] = None

# ---------------------------------------------------------------------------
# Habitat-MP3D Dataset Handler
# ---------------------------------------------------------------------------

class HabitatNavigationDataset(Dataset):
    """Dataset class for Habitat-MP3D navigation data."""
    
    def __init__(
        self,
        config: NavigationTrainingConfig,
        processor,
        split: str = "train",
        max_episodes: Optional[int] = None
    ):
        self.config = config
        self.processor = processor
        self.split = split
        
        # Initialize Habitat environment
        self.habitat_config = self._load_habitat_config()
        self.env = habitat.Env(config=self.habitat_config)
        self.follower = ShortestPathFollower(
            self.env.sim, goal_radius=0.5, return_one_hot=False
        )
        
        # Generate navigation episodes
        self.episodes = self._generate_episodes(max_episodes or config.num_episodes)
        
    def _load_habitat_config(self) -> Config:
        """Load and configure Habitat environment."""
        config = habitat.get_config(self.config.habitat_config_path)
        
        # Customize config for navigation training
        config.DATASET.DATA_PATH = "data/datasets/pointnav/mp3d/v1/{split}/{split}.json.gz"
        config.DATASET.SCENES_DIR = "data/scene_datasets/mp3d/"
        config.ENVIRONMENT.MAX_EPISODE_STEPS = self.config.max_episode_steps
        
        # Configure sensors
        config.SIMULATOR.RGB_SENSOR.HEIGHT = self.config.image_size[0]
        config.SIMULATOR.RGB_SENSOR.WIDTH = self.config.image_size[1]
        config.SIMULATOR.DEPTH_SENSOR.HEIGHT = self.config.image_size[0]
        config.SIMULATOR.DEPTH_SENSOR.WIDTH = self.config.image_size[1]
        
        return config
    
    def _generate_episodes(self, num_episodes: int) -> List[Dict]:
        """Generate navigation episodes with vision-language pairs."""
        episodes = []
        
        print(f"Generating {num_episodes} navigation episodes...")
        
        for episode_id in tqdm(range(num_episodes)):
            try:
                observations = self.env.reset()
                episode_data = []
                
                # Get goal position
                goal_position = self.env.current_episode.goals[0].position
                
                for step in range(self.config.max_episode_steps):
                    # Get current observation
                    rgb_obs = observations["rgb"]
                    
                    # Get optimal action from shortest path follower
                    best_action = self.follower.get_next_action(goal_position)
                    
                    # Generate navigation instruction based on action
                    instruction = self._action_to_instruction(
                        best_action, rgb_obs, goal_position, observations
                    )
                    
                    episode_data.append({
                        "image": rgb_obs,
                        "instruction": instruction,
                        "action": best_action,
                        "step": step,
                        "episode_id": episode_id
                    })
                    
                    # Take action
                    observations = self.env.step(best_action)
                    
                    # Check if episode is done
                    if self.env.episode_over:
                        break
                
                episodes.extend(episode_data)
                
            except Exception as e:
                print(f"Error in episode {episode_id}: {e}")
                continue
        
        print(f"Generated {len(episodes)} training samples")
        return episodes
    
    def _action_to_instruction(
        self, 
        action: int, 
        rgb_obs: np.ndarray, 
        goal_position: np.ndarray,
        observations: Dict
    ) -> str:
        """Convert action and observations to navigation instruction."""
        
        # Action mappings for Habitat
        action_map = {
            0: "STOP",
            1: "MOVE_FORWARD", 
            2: "TURN_LEFT",
            3: "TURN_RIGHT"
        }
        
        action_name = action_map.get(action, "UNKNOWN")
        
        # Generate contextual instruction based on visual scene
        if action == 0:  # STOP
            return "You have reached your destination. Stop here."
        elif action == 1:  # MOVE_FORWARD
            return "The path ahead is clear. Move forward to continue toward your goal."
        elif action == 2:  # TURN_LEFT
            return "Turn left to align with the optimal path to your destination."
        elif action == 3:  # TURN_RIGHT
            return "Turn right to align with the optimal path to your destination."
        
        return f"Take action: {action_name}"
    
    def __len__(self) -> int:
        return len(self.episodes)
    
    def __getitem__(self, idx: int) -> Dict:
        episode = self.episodes[idx]
        
        # Convert image to PIL format
        image = Image.fromarray(episode["image"])
        
        # Create conversation format for Magma
        conversation = [
            {
                "role": "system",
                "content": "You are a navigation assistant. Given an image, provide the next navigation action and instruction."
            },
            {
                "role": "user", 
                "content": "<image_start><image><image_end>\nWhat should I do next to reach my navigation goal?"
            },
            {
                "role": "assistant",
                "content": episode["instruction"]
            }
        ]
        
        return {
            "image": image,
            "conversation": conversation,
            "episode_id": episode["episode_id"],
            "step": episode["step"]
        }

# ---------------------------------------------------------------------------
# Data Collator for Vision-Language Training
# ---------------------------------------------------------------------------

class NavigationDataCollator:
    """Custom data collator for navigation training."""
    
    def __init__(self, processor, max_length: int = 512):
        self.processor = processor
        self.max_length = max_length
    
    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        images = [item["image"] for item in batch]
        conversations = [item["conversation"] for item in batch]
        
        # Process conversations into prompts
        prompts = []
        for conv in conversations:
            prompt = self.processor.tokenizer.apply_chat_template(
                conv, tokenize=False, add_generation_prompt=False
            )
            prompts.append(prompt)
        
        # Process images and text together
        inputs = self.processor(
            images=images,
            texts=prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length
        )
        
        # Create labels for language modeling
        labels = inputs["input_ids"].clone()
        
        # Mask padding tokens in labels
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        
        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "pixel_values": inputs["pixel_values"],
            "image_sizes": inputs["image_sizes"],
            "labels": labels
        }

# ---------------------------------------------------------------------------
# LoRA Training Setup
# ---------------------------------------------------------------------------

class NavigationLoRATrainer:
    """Main trainer class for LoRA fine-tuning."""
    
    def __init__(self, config: NavigationTrainingConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize wandb if requested
        if config.use_wandb:
            wandb.init(
                project=config.wandb_project,
                name=config.wandb_run_name or f"magma-nav-{time.strftime('%Y%m%d_%H%M%S')}",
                config=config.__dict__
            )
    
    def setup_model_and_processor(self):
        """Load and setup model with LoRA."""
        print("Loading base model and processor...")
        
        # Load processor
        self.processor = AutoProcessor.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
            cache_dir=self.config.cache_dir
        )
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            cache_dir=self.config.cache_dir
        )
        
        # Prepare model for LoRA
        self.model = prepare_model_for_kbit_training(self.model)
        
        # Configure LoRA
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.target_modules,
            bias="none"
        )
        
        # Apply LoRA to model
        self.model = get_peft_model(self.model, lora_config)
        
        print(f"LoRA model parameters: {self.model.num_parameters()}")
        print(f"Trainable parameters: {self.model.num_parameters(only_trainable=True)}")
        
    def setup_datasets(self):
        """Setup training and validation datasets."""
        print("Setting up datasets...")
        
        # Create full dataset
        full_dataset = HabitatNavigationDataset(
            self.config, 
            self.processor, 
            max_episodes=self.config.num_episodes
        )
        
        # Split into train/val
        train_size = int(len(full_dataset) * self.config.train_split)
        val_size = len(full_dataset) - train_size
        
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size]
        )
        
        print(f"Training samples: {len(self.train_dataset)}")
        print(f"Validation samples: {len(self.val_dataset)}")
        
        # Setup data collator
        self.data_collator = NavigationDataCollator(
            self.processor, 
            max_length=self.config.max_sequence_length
        )
    
    def setup_training_args(self) -> TrainingArguments:
        """Setup training arguments."""
        return TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            warmup_steps=self.config.warmup_steps,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps,
            evaluation_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to="wandb" if self.config.use_wandb else None,
            dataloader_pin_memory=False,
            remove_unused_columns=False,
            bf16=True,
            gradient_checkpointing=True,
        )
    
    def train(self):
        """Run the training process."""
        print("Starting LoRA training...")
        
        # Setup model and datasets
        self.setup_model_and_processor()
        self.setup_datasets()
        
        # Setup trainer
        training_args = self.setup_training_args()
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            data_collator=self.data_collator,
            tokenizer=self.processor.tokenizer,
        )
        
        # Train model
        trainer.train()
        
        # Save final model
        final_model_path = os.path.join(self.config.output_dir, "final_model")
        trainer.save_model(final_model_path)
        
        print(f"Training completed! Model saved to {final_model_path}")
        
        return trainer

# ---------------------------------------------------------------------------
# Model Evaluation and Testing
# ---------------------------------------------------------------------------

class NavigationEvaluator:
    """Evaluate trained navigation model."""
    
    def __init__(self, model_path: str, base_model_name: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load processor
        self.processor = AutoProcessor.from_pretrained(
            base_model_name, trust_remote_code=True
        )
        
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        
        # Load LoRA weights
        self.model = PeftModel.from_pretrained(base_model, model_path)
        self.model.eval()
    
    def evaluate_navigation(self, test_images: List[Image.Image], tasks: List[str]) -> List[str]:
        """Evaluate model on navigation tasks."""
        results = []
        
        for image, task in zip(test_images, tasks):
            # Create conversation
            conversation = [
                {
                    "role": "system",
                    "content": "You are a navigation assistant. Provide clear navigation instructions."
                },
                {
                    "role": "user",
                    "content": f"<image_start><image><image_end>\n{task}"
                }
            ]
            
            # Generate response
            prompt = self.processor.tokenizer.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=True
            )
            
            inputs = self.processor(
                images=[image], texts=prompt, return_tensors="pt"
            ).to(self.device)
            
            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=0.2,
                    do_sample=True
                )
            
            # Decode response
            response = self.processor.decode(
                outputs[0][inputs["input_ids"].shape[-1]:],
                skip_special_tokens=True
            )
            
            results.append(response.strip())
        
        return results

# ---------------------------------------------------------------------------
# Main Training Script
# ---------------------------------------------------------------------------

def main():
    """Main training function."""
    
    # Configuration
    config = NavigationTrainingConfig(
        num_episodes=2000,  # Adjust based on your needs
        num_train_epochs=5,
        per_device_train_batch_size=1,  # Adjust based on GPU memory
        gradient_accumulation_steps=16,
        learning_rate=1e-4,
        use_wandb=True
    )
    
    # Create trainer and start training
    trainer = NavigationLoRATrainer(config)
    trained_model = trainer.train()
    
    print("Training completed successfully!")

if __name__ == "__main__":
    main()