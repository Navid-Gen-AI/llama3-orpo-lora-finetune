import torch

def set_device_config():
    if torch.cuda.get_device_capability()[0] >= 8:
        !pip install -qqq flash-attn
        return "flash_attention_2", torch.bfloat16
    else:
        return "eager", torch.float16