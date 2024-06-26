## Llama3 ORPO LoRA Fine-Tune ⭐

This project focuses on fine-tuning the Llama 3 model using ORPO and LoRA techniques. It includes scripts for model configuration, dataset preparation, training, and finalization.


#### Setup ✔️
1. Install required packages:
   ```bash
   pip install -r requirements.txt
2. Run the main script:
    ```bash
    python main.py
    
### Directory Structure 📂
- **data/**
  - `datasets.py`: Script to load and preprocess the dataset.
- **config/**
  - `device_config.py`: Script to set device-specific parameters.
- **models/**
  - `__init__.py`: Marks the directory as a Python package.
  - `load_model.py`: Script to configure and load the model.
  - `train_model.py`: Script to train the model.
- **utils/**
  - `cleanup.py`: Script to clean up and finalize the model.
- **main.py**: The main script to run the project.


### Overview
Model Configuration and Loading: Configures and loads the Llama 3 model with ORPO and LoRA configurations.
Dataset Preparation: Loads and preprocesses the dataset using a chat template format.
Training: Trains the model using ORPO Trainer and logs the training process with WandB.
Cleanup and Finalization: Cleans up the environment and finalizes the model for deployment.
This project aims to provide an efficient and structured approach to fine-tuning language models using advanced techniques.
