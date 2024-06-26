import gc
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from trl import setup_chat_format

def cleanup_and_finalize(base_model, new_model):
    del trainer, model
    gc.collect()
    torch.cuda.empty_cache()

    # Reload tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        low_cpu_mem_usage=True,
        return_dict=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model, tokenizer = setup_chat_format(model, tokenizer)

    # Merge adapter with base model
    model = PeftModel.from_pretrained(model, new_model)
    model = model.merge_and_unload()

    model.push_to_hub(new_model, use_temp_dir=False)
    tokenizer.push_to_hub(new_model, use_temp_dir=False)