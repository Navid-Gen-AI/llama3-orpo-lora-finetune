import wandb
from trl import ORPOConfig, ORPOTrainer
from transformers import TrainingArguments

def setup_wandb(wb_token):
    wandb.login(key=wb_token)

def train_model(model, tokenizer, peft_config, dataset):
    # ORPO Trainer Configuration
    orpo_config = ORPOConfig(
        model=model,
        reward_model=model,
        task_type="CAUSAL_LM",
        target_modules=['up_proj', 'down_proj', 'gate_proj', 'k_proj', 'q_proj', 'v_proj', 'o_proj']
    )

    orpo_args = TrainingArguments(
       learning_rate=8e-6,              # The learning rate controls how quickly the model learns by adjusting its weights.
        beta=0.1,                        # A hyperparameter relevant to the optimization process.
        lr_scheduler_type="linear",      # Specifies a linear learning rate scheduler.
        max_length=1024,                 # Maximum sequence length for model input.
        max_prompt_length=512,           # Maximum length for prompts in the dataset.
        per_device_train_batch_size=2,   # Batch size for training per device (e.g., per GPU).
        per_device_eval_batch_size=2,    # Batch size for evaluation per device.
        gradient_accumulation_steps=4,   # Number of steps to accumulate gradients before updating.
        optim="paged_adamw_8bit",        # Optimizer type using 8-bit precision for memory efficiency.
        num_train_epochs=1,              # Number of epochs for training (full passes over the dataset).
        evaluation_strategy="steps",     # Evaluates the model at regular step intervals.
        eval_steps=0.2,                  # Frequency of evaluation, every 20% of the steps in an epoch.
        logging_steps=1,                 # Logs metrics and other information after every step.
        warmup_steps=10,                 # Number of warmup steps for the learning rate scheduler.
        report_to="wandb",               # Uses Weights & Biases for logging metrics and information.
        output_dir="./results/", 
    )

    # Train the model
    trainer = ORPOTrainer(
        model=model,
        args=orpo_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        peft_config=peft_config,
        tokenizer=tokenizer,
    )
    trainer.train()
    trainer.save_model(new_model)