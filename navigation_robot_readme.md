# Navigation-Aware Robot Assistant System

An intelligent navigation system for assistive robotics built on Microsoft's Magma-8B vision-language-action model. This system provides comprehensive scene understanding, navigation guidance, hazard detection, and goal-directed action planning for mobile robots.

## 🎯 Project Overview

This project implements a navigation-aware robot assistant that can:
- **Describe environments** with varying levels of detail
- **Provide navigation guidance** using spatial relationships and landmarks
- **Detect potential hazards** and safety concerns
- **Generate step-by-step action plans** for goal-directed tasks
- **Synthesize speech output** for real-time audio feedback

The system is designed primarily for Google Colab environments, as it provides range of GPUs to select, but can be adapted for other platforms.

## 🏗️ System Architecture

### Core Components

1. **NavigationSupportSystem** (Base Class)
   - Visual scene interpretation using Magma-8B
   - Environment description with configurable detail levels
   - Navigation guidance with spatial awareness
   - Hazard detection and safety assessment
   - Text-to-speech synthesis

2. **NavigationRobotSystem** (Extended Class)
   - Goal-directed action planning
   - Task decomposition into executable steps
   - Comprehensive task processing pipeline

### Key Features

- **Multi-modal Processing**: Combines computer vision and natural language processing
- **Configurable Detail Levels**: Adjustable output verbosity (low/medium/high)
- **Safety-First Design**: Prioritizes hazard detection and obstacle avoidance
- **Landmark-Based Navigation**: Uses environmental features for orientation
- **Real-time Audio Feedback**: Integrated text-to-speech for accessibility

## 🛠️ Installation & Setup

### Prerequisites

```bash
# Core ML libraries
pip install torch torchvision
pip install transformers
pip install pillow numpy matplotlib

# Audio processing
pip install gtts

# Additional dependencies
pip install open-clip-torch
```

### Hardware Requirements

- **GPU**: CUDA-compatible GPU recommended (falls back to CPU)
- **Memory**: Minimum 8GB RAM, 16GB+ recommended for optimal performance
- **Storage**: ~10GB for model weights and results

### Environment Setup

1. **Google Colab** (Recommended)
   ```python
   # Clone or upload the repository
   !git clone <your-repository-url>
   
   # Install dependencies
   !pip install open-clip-torch gtts
   ```

2. **Local Environment**
   ```bash
   git clone <your-repository-url>
   cd navigation-robot-system
   pip install -r requirements.txt
   ```

## 🚀 Usage

### Basic Usage

```python
from navigation_robot_system import NavigationRobotSystem

# Initialize the system
nrs = NavigationRobotSystem()

# Process a task with webcam input
description, guidance, plan, hazards, audio = nrs.process_task(
    image_source="webcam",
    task="Navigate to the kitchen door"
)

# Process with uploaded image
description, guidance, plan, hazards, audio = nrs.process_task(
    image_source="upload",
    task="Find the person in red shirt"
)
```

### Configuration Options

```python
# Customize system behavior
nrs.settings.update({
    "description_detail": "high",    # low | medium | high
    "guidance_detail": "high",       # low | medium | high
    "action_detail": "medium",       # low | medium | high
    "hazard_priority": True,         # prioritize safety
    "landmarks_focus": True,         # emphasize navigation landmarks
    "voice_speed": 1.0,             # TTS speed multiplier
    "max_waypoints": 10             # maximum steps in action plan
})
```

## 📁 Project Structure

```
navigation-robot-system/
├── navigation_robot_system.py    # Main system implementation
├── README.md                     # Project documentation
├── requirements.txt              # Python dependencies
├── navigation_robot_results/     # Generated outputs
│   ├── capture_*.jpg            # Captured images
│   ├── result_*.json            # Processing results
│   └── speech.mp3               # Generated audio
└── examples/                     # Usage examples
    ├── basic_usage.ipynb
    └── advanced_configuration.py
```

## 🔧 Technical Details

### Model Architecture
- **Base Model**: Microsoft Magma-8B (8-billion parameter vision-language model)
- **Vision Encoder**: Multi-modal transformer for image understanding
- **Language Decoder**: Causal language model for text generation
- **Processing**: Mixed precision (bfloat16) for efficiency

### Prompt Engineering
The system uses carefully crafted prompts optimized for:
- Scene description with spatial awareness
- Safety-oriented hazard detection
- Step-by-step action planning
- Accessibility-focused language

### Performance Optimizations
- **GPU Acceleration**: CUDA support with automatic fallback
- **Mixed Precision**: bfloat16 for memory efficiency
- **Inference Mode**: Optimized generation without gradient computation
- **Configurable Output**: Adjustable token limits for different use cases

## 📊 Example Outputs

### Environment Description
```
"A modern kitchen with white cabinets and granite countertops. There's a woman in a red shirt standing near the stove at 2 meters position, approximately 3 meters away. A dining table with chairs is visible at 10m. The floor appears clear with no immediate obstacles."
```

### Navigation Guidance
```
"Safe path forward: Walk straight ahead for 2 meters, then turn slightly right toward the kitchen island. Avoid the dining chair at 9 o'clock position. The woman you're looking for is at your 2 o'clock, near the cooking area."
```

### Action Plan
```
"1. Move forward 2 meters toward kitchen center
2. Turn 30 degrees right toward cooking area
3. Navigate around kitchen island on left side
4. Approach woman at stove (2 o'clock position)
5. Stop 1 meter away for comfortable interaction
Task accomplished"
```

## 🧪 Testing & Validation

### Test Scenarios
- **Indoor Navigation**: Living rooms, kitchens, offices
- **Obstacle Avoidance**: Furniture, pets, people
- **Person Finding**: Locating specific individuals
- **Object Retrieval**: Navigating to specific items

### Performance Metrics
- **Processing Speed**: ~3-5 seconds per image on GPU
- **Accuracy**: Evaluated on spatial relationship understanding
- **Safety**: Hazard detection precision and recall

## 🔮 Future Enhancements

- **Real-time Processing**: Integration with live camera feeds
- **Multi-language Support**: Expand beyond English
- **Advanced Path Planning**: Integration with SLAM systems
- **Personalization**: User-specific preferences and adaptations
- **Mobile Deployment**: Optimization for edge devices

## 🤝 Contributing

This is an academic project. For contributions or questions:

1. Fork the repository
2. Create a feature branch
3. Commit changes 
4. Push to branch 
5. Open a Pull Request

## 📚 Academic Context

This project demonstrates:
- **Computer Vision**: Scene understanding and spatial reasoning
- **Natural Language Processing**: Multi-modal prompt engineering
- **Human-Computer Interaction**: Accessibility-focused design
- **Assistive Technology**: AI applications for Navigation tasks
- **Robotics**: Navigation and path planning systems

## 📄 License

This project is developed for academic purposes, under Curtin University

## 👥 Authors

Namash Aggarwal - 21023169
[Curtin University - COMP 6016 
[Masters of Computing

## 🙏 Acknowledgments

- Microsoft Research for the Magma-8B model - https://huggingface.co/microsoft/Magma-8B
- Hugging Face Transformers library - https://huggingface.co/
- Google Colab for computational resources - https://colab.research.google.com/
---

*Last updated: May 2025*