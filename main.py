from data.datasets import load_datasets
from config.device_config import set_device_config
from models.load_model import load_model
from models.train_model import setup_wandb, train_model
from utils.cleanup import cleanup_and_finalize
from google.colab import userdata

def main():
    # Set up WandB
    wb_token = userdata.get('wandb')
    setup_wandb(wb_token)
    
    # Set Device-Specific Parameters
    attn_implementation, torch_dtype = set_device_config()

    # Load datasets
    dataset = load_datasets()
    
    # Model Configuration and Loading
    base_model = "meta-llama/Meta-Llama-3-8B"
    new_model = "OrpoLlama-3-8B"
    model, tokenizer, peft_config = load_model(base_model, torch_dtype)
    
    # Train the model
    train_model(model, tokenizer, peft_config, dataset)
    
    # Clean up and finalize
    cleanup_and_finalize(base_model, new_model)

if __name__ == "__main__":
    main()