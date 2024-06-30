from datasets import load_dataset
#Handling the dataset
def load_datasets(tokenizer):
    # Dataset Preparation, load and shuffle then select the first 100 samples
    dataset_name = "2A2I/argilla-dpo-mix-7k-arabic"
    dataset = load_dataset(dataset_name, split="all")
    dataset = dataset.shuffle(seed=42).select(range(100))

    # Formats the data using a chat template
    def format_chat_template(row):
        row["chosen"] = tokenizer.apply_chat_template(row["chosen"], tokenize=False)
        row["rejected"] = tokenizer.apply_chat_template(row["rejected"], tokenize=False)
        return row

    dataset = dataset.map(
        format_chat_template,
        num_proc=os.cpu_count(),
    )
    # Split dataset
    dataset = dataset.train_test_split(test_size=0.01)
    
    return dataset
