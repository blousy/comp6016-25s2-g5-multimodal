# training_utilities.py
"""
Utility functions and helper scripts for Magma-8B LoRA training on Habitat-MP3D.
"""

import os
import json
import shutil
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import wandb
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Data Preparation Utilities
# ---------------------------------------------------------------------------

class HabitatDataPreparator:
    """Prepare and validate Habitat-MP3D data for training."""
    
    def __init__(self, data_root: str = "./data"):
        self.data_root = Path(data_root)
        self.scenes_dir = self.data_root / "scene_datasets" / "mp3d"
        self.dataset_dir = self.data_root / "datasets" / "pointnav" / "mp3d" / "v1"
    
    def check_data_availability(self) -> Dict[str, bool]:
        """Check if required data files are available."""
        checks = {
            "scenes_directory": self.scenes_dir.exists(),
            "train_episodes": (self.dataset_dir / "train" / "train.json.gz").exists(),
            "val_episodes": (self.dataset_dir / "val" / "val.json.gz").exists(),
            "test_episodes": (self.dataset_dir / "test" / "test.json.gz").exists(),
        }
        
        # Check for scene files
        if checks["scenes_directory"]:
            scene_files = list(self.scenes_dir.glob("*.glb"))
            checks["scene_files_count"] = len(scene_files)
            checks["has_scene_files"] = len(scene_files) > 0
        else:
            checks["scene_files_count"] = 0
            checks["has_scene_files"] = False
        
        return checks
    
    def download_instructions(self) -> str:
        """Provide instructions for downloading MP3D data."""
        return """
        To download the Habitat-MP3D dataset:
        
        1. Install Habitat-Sim and Habitat-Lab:
           conda install habitat-sim habitat-lab -c conda-forge -c aihabitat
        
        2. Download MP3D scenes (requires agreement to terms):
           python -m habitat_sim.utils.datasets_download --dataset mp3d --data-path ./data/
        
        3. Download PointNav episodes:
           python -m habitat_sim.utils.datasets_download --dataset pointnav_mp3d --data-path ./data/
        
        4. Verify data structure:
           data/
           ├── scene_datasets/mp3d/
           │   ├── *.glb (scene files)
           │   └── *.navmesh (navigation meshes)
           └── datasets/pointnav/mp3d/v1/
               ├── train/train.json.gz
               ├── val/val.json.gz
               └── test/test.json.gz
        """
    
    def validate_episodes(self, split: str = "train", max_check: int = 100) -> Dict:
        """Validate episode data integrity."""
        episode_file = self.dataset_dir / split / f"{split}.json.gz"
        
        if not episode_file.exists():
            return {"valid": False, "error": f"Episode file {episode_file} not found"}
        
        try:
            import gzip
            with gzip.open(episode_file, 'rt') as f:
                data = json.load(f)
            
            episodes = data.get("episodes", [])
            
            validation_results = {
                "valid": True,
                "total_episodes": len(episodes),
                "checked_episodes": min(max_check, len(episodes)),
                "issues": []
            }
            
            # Check episode structure
            for i, episode in enumerate(episodes[:max_check]):
                required_keys = ["episode_id", "scene_id", "start_position", "start_rotation", "goals"]
                
                for key in required_keys:
                    if key not in episode:
                        validation_results["issues"].append(f"Episode {i}: Missing {key}")
                
                # Check goals structure
                if "goals" in episode and episode["goals"]:
                    goal = episode["goals"][0]
                    if "position" not in goal:
                        validation_results["issues"].append(f"Episode {i}: Goal missing position")
            
            validation_results["valid"] = len(validation_results["issues"]) == 0
            
            return validation_results
            
        except Exception as e:
            return {"valid": False, "error": str(e)}

# ---------------------------------------------------------------------------
# Model Analysis and Monitoring
# ---------------------------------------------------------------------------

class TrainingMonitor:
    """Monitor training progress and model performance."""
    
    def __init__(self, checkpoint_dir: str, log_file: Optional[str] = None):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_file = log_file
        self.metrics_history = []
    
    def log_metrics(self, step: int, metrics: Dict[str, float]):
        """Log training metrics."""
        entry = {"step": step, "timestamp": torch.cuda.Event(), **metrics}
        self.metrics_history.append(entry)
        
        if self.log_file:
            with open(self.log_file, 'a') as f:
                f.write(f"{json.dumps(entry)}\n")
    
    def plot_training_curves(self, save_path: Optional[str] = None):
        """Plot training curves from logged metrics."""
        if not self.metrics_history:
            print("No metrics to plot")
            return
        
        # Extract data
        steps = [entry["step"] for entry in self.metrics_history]
        train_loss = [entry.get("train_loss", 0) for entry in self.metrics_history]
        eval_loss = [entry.get("eval_loss", 0) for entry in self.metrics_history]
        
        # Create plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Training loss
        ax1.plot(steps, train_loss, label="Training Loss", color="blue")
        ax1.set_xlabel("Steps")
        ax1.set_ylabel("Loss")
        ax1.set_title("Training Loss Over Time")
        ax1.legend()
        ax1.grid(True)
        
        # Evaluation loss
        if eval_loss and any(loss > 0 for loss in eval_loss):
            ax2.plot(steps, eval_loss, label="Evaluation Loss", color="red")
            ax2.set_xlabel("Steps")
            ax2.set_ylabel("Loss")
            ax2.set_title("Evaluation Loss Over Time")
            ax2.legend()
            ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        
        plt.show()
    
    def analyze_checkpoints(self) -> Dict[str, any]:
        """Analyze saved checkpoints."""
        checkpoints = list(self.checkpoint_dir.glob("checkpoint-*"))
        
        analysis = {
            "num_checkpoints": len(checkpoints),
            "checkpoint_steps": [],
            "total_size_mb": 0
        }
        
        for ckpt in sorted(checkpoints):
            # Extract step number
            step = int(ckpt.name.split("-")[1])
            analysis["checkpoint_steps"].append(step)
            
            # Calculate size
            size = sum(f.stat().st_size for f in ckpt.rglob("*") if f.is_file())
            analysis["total_size_mb"] += size / (1024 * 1024)
        
        analysis["total_size_mb"] = round(analysis["total_size_mb"], 2)
        
        return analysis

# ---------------------------------------------------------------------------
# Model Inference and Testing
# ---------------------------------------------------------------------------

class NavigationInferenceTester:
    """Test trained navigation model on various scenarios."""
    
    def __init__(self, model_path: str, base_model_name: str = "microsoft/Magma-8B"):
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoProcessor
        
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
    
    def test_navigation_scenarios(self, test_scenarios: List[Dict]) -> List[Dict]:
        """Test model on predefined navigation scenarios."""
        results = []
        
        for scenario in tqdm(test_scenarios, desc="Testing scenarios"):
            image = scenario["image"]
            task = scenario.get("task", "Navigate to your destination")
            expected = scenario.get("expected", None)
            
            # Generate response
            response = self._generate_navigation_response(image, task)
            
            # Evaluate response quality
            evaluation = self._evaluate_response(response, expected)
            
            results.append({
                "scenario_id": scenario.get("id", "unknown"),
                "task": task,
                "response": response,
                "expected": expected,
                "evaluation": evaluation,
                "image_path": scenario.get("image_path", None)
            })
        
        return results
    
    def _generate_navigation_response(self, image: Image.Image, task: str) -> str:
        """Generate navigation response for given image and task."""
        conversation = [
            {
                "role": "system",
                "content": "You are a navigation assistant for indoor environments. Provide clear, safe navigation instructions."
            },
            {
                "role": "user",
                "content": f"<image_start><image><image_end>\n{task}"
            }
        ]
        
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
                temperature=0.1,
                do_sample=True,
                pad_token_id=self.processor.tokenizer.eos_token_id
            )
        
        response = self.processor.decode(
            outputs[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True
        )
        
        return response.strip()
    
    def _evaluate_response(self, response: str, expected: Optional[str] = None) -> Dict:
        """Evaluate quality of navigation response."""
        evaluation = {
            "length": len(response.split()),
            "contains_direction": any(word in response.lower() for word in 
                                   ["left", "right", "forward", "backward", "straight", "turn"]),
            "contains_safety": any(word in response.lower() for word in 
                                 ["safe", "careful", "obstacle", "avoid", "hazard"]),
            "contains_landmarks": any(word in response.lower() for word in 
                                   ["door", "wall", "table", "chair", "window", "corner"]),
            "is_actionable": len([word for word in response.lower().split() 
                                if word in ["go", "move", "turn", "walk", "stop", "continue"]]) > 0
        }
        
        # Calculate overall quality score
        score = 0
        if evaluation["contains_direction"]: score += 25
        if evaluation["contains_safety"]: score += 20
        if evaluation["contains_landmarks"]: score += 20
        if evaluation["is_actionable"]: score += 25
        if 10 <= evaluation["length"] <= 50: score += 10  # Appropriate length
        
        evaluation["quality_score"] = score
        
        return evaluation
    
    def benchmark_against_baseline(self, test_images: List[Image.Image], 
                                 baseline_model_path: Optional[str] = None) -> Dict:
        """Compare performance against baseline model."""
        # This would compare against the original Magma-8B or another baseline
        # Implementation depends on specific evaluation metrics needed
        pass

# ---------------------------------------------------------------------------
# Training Pipeline Orchestrator
# ---------------------------------------------------------------------------

class TrainingPipeline:
    """Orchestrate the complete training pipeline."""
    
    def __init__(self, config_path: Optional[str] = None):
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            from types import SimpleNamespace
            self.config = SimpleNamespace(**config_dict)
        else:
            from magma_lora_training import NavigationTrainingConfig
            self.config = NavigationTrainingConfig()
    
    def run_full_pipeline(self):
        """Run the complete training pipeline from data validation to evaluation."""
        
        print("🚀 Starting Magma-8B LoRA Navigation Training Pipeline")
        print("=" * 60)
        
        # Step 1: Data validation
        print("\n📊 Step 1: Validating data...")
        data_prep = HabitatDataPreparator()
        data_status = data_prep.check_data_availability()
        
        if not data_status["has_scene_files"]:
            print("❌ MP3D scene files not found!")
            print(data_prep.download_instructions())
            return False
        
        print(f"✅ Found {data_status['scene_files_count']} scene files")
        
        # Validate episodes
        episode_validation = data_prep.validate_episodes("train", max_check=50)
        if not episode_validation["valid"]:
            print(f"❌ Episode validation failed: {episode_validation.get('error', 'Unknown error')}")
            return False
        
        print(f"✅ Validated {episode_validation['checked_episodes']} episodes")
        
        # Step 2: Training
        print("\n🔥 Step 2: Starting LoRA training...")
        from magma_lora_training import NavigationLoRATrainer
        
        trainer = NavigationLoRATrainer(self.config)
        trained_model = trainer.train()
        
        # Step 3: Evaluation
        print("\n📈 Step 3: Evaluating trained model...")
        self._run_evaluation()
        
        # Step 4: Generate report
        print("\n📋 Step 4: Generating training report...")
        self._generate_report()
        
        print("\n🎉 Training pipeline completed successfully!")
        return True
    
    def _run_evaluation(self):
        """Run model evaluation on test scenarios."""
        model_path = os.path.join(self.config.output_dir, "final_model")
        
        if not os.path.exists(model_path):
            print("⚠️ No trained model found for evaluation")
            return
        
        # Create test scenarios (you would load real test data here)
        test_scenarios = self._create_test_scenarios()
        
        # Run inference tests
        tester = NavigationInferenceTester(model_path)
        results = tester.test_navigation_scenarios(test_scenarios)
        
        # Save results
        results_path = os.path.join(self.config.output_dir, "evaluation_results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"✅ Evaluation results saved to {results_path}")
    
    def _create_test_scenarios(self) -> List[Dict]:
        """Create test scenarios for evaluation."""
        # This is a placeholder - you would load real test images and scenarios
        scenarios = [
            {
                "id": "hallway_navigation",
                "task": "Navigate down the hallway to the door at the end",
                "expected": "Move forward down the hallway, avoiding obstacles"
            },
            {
                "id": "room_exploration", 
                "task": "Find the kitchen in this house",
                "expected": "Look for kitchen appliances and navigate accordingly"
            }
        ]
        
        return scenarios
    
    def _generate_report(self):
        """Generate comprehensive training report."""
        report = {
            "training_config": vars(self.config),
            "training_completed": True,
            "model_path": os.path.join(self.config.output_dir, "final_model"),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Add checkpoint analysis
        monitor = TrainingMonitor(self.config.output_dir)
        checkpoint_analysis = monitor.analyze_checkpoints()
        report["checkpoint_analysis"] = checkpoint_analysis
        
        # Save report
        report_path = os.path.join(self.config.output_dir, "training_report.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"📋 Training report saved to {report_path}")

# ---------------------------------------------------------------------------
# Command Line Interface
# ---------------------------------------------------------------------------

def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(description="Magma-8B LoRA Navigation Training")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Data validation command
    data_parser = subparsers.add_parser("validate-data", help="Validate Habitat-MP3D data")
    data_parser.add_argument("--data-root", default="./data", help="Root directory for data")
    
    # Training command
    train_parser = subparsers.add_parser("train", help="Start LoRA training")
    train_parser.add_argument("--config", help="Path to training config JSON file")
    train_parser.add_argument("--episodes", type=int, default=1000, help="Number of episodes to generate")
    train_parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    train_parser.add_argument("--batch-size", type=int, default=2, help="Batch size per device")
    train_parser.add_argument("--learning-rate", type=float, default=2e-4, help="Learning rate")
    
    # Evaluation command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate trained model")
    eval_parser.add_argument("--model-path", required=True, help="Path to trained LoRA model")
    eval_parser.add_argument("--test-scenarios", help="Path to test scenarios JSON file")
    
    # Full pipeline command
    pipeline_parser = subparsers.add_parser("pipeline", help="Run full training pipeline")
    pipeline_parser.add_argument("--config", help="Path to training config JSON file")
    
    args = parser.parse_args()
    
    if args.command == "validate-data":
        prep = HabitatDataPreparator(args.data_root)
        status = prep.check_data_availability()
        
        print("Data Validation Results:")
        print("-" * 30)
        for key, value in status.items():
            print(f"{key}: {value}")
        
        if not status["has_scene_files"]:
            print("\nDownload Instructions:")
            print(prep.download_instructions())
    
    elif args.command == "train":
        from magma_lora_training import NavigationTrainingConfig, NavigationLoRATrainer
        
        if args.config:
            with open(args.config, 'r') as f:
                config_dict = json.load(f)
            config = NavigationTrainingConfig(**config_dict)
        else:
            config = NavigationTrainingConfig(
                num_episodes=args.episodes,
                num_train_epochs=args.epochs,
                per_device_train_batch_size=args.batch_size,
                learning_rate=args.learning_rate
            )
        
        trainer = NavigationLoRATrainer(config)
        trainer.train()
    
    elif args.command == "evaluate":
        tester = NavigationInferenceTester(args.model_path)
        
        if args.test_scenarios:
            with open(args.test_scenarios, 'r') as f:
                scenarios = json.load(f)
        else:
            scenarios = []  # Would create default scenarios
        
        results = tester.test_navigation_scenarios(scenarios)
        
        print("Evaluation Results:")
        print("-" * 30)
        for result in results:
            print(f"Scenario: {result['scenario_id']}")
            print(f"Quality Score: {result['evaluation']['quality_score']}/100")
            print()
    
    elif args.command == "pipeline":
        pipeline = TrainingPipeline(args.config)
        success = pipeline.run_full_pipeline()
        
        if success:
            print("🎉 Pipeline completed successfully!")
        else:
            print("❌ Pipeline failed. Check logs for details.")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()