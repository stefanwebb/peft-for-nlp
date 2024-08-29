import os
os.environ['HF_HOME'] = '/home/stefanwebb/models/fine-tuned'
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import gc
import multiprocessing

import torch
torch.random.manual_seed(0)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TextStreamer
from peft import LoraConfig
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

if torch.cuda.is_bf16_supported():
  compute_dtype = torch.bfloat16
else:
  compute_dtype = torch.float16

"""
    Load base model and tokenizer
"""
use_4bit = True
bnb_4bit_quant_type = "nf4"
use_double_quant = True

compute_dtype = torch.bfloat16
attn_implementation = 'flash_attention_2'

target_modules = ["all_linear"]

bnb_config = BitsAndBytesConfig(
        load_in_4bit=use_4bit,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=use_double_quant,
)

MODEL_ID = "/home/stefanwebb/models/llm/meta_llama3-8b-instruct"
NEW_MODEL_NAME = "stefans-debug-llama3-grammatical-error-correction"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map='auto',
    torch_dtype="auto",
    quantization_config=bnb_config,
    attn_implementation=attn_implementation
)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    use_fast=True,
    add_eos_token=True, #,
    add_bos_token=False)

# tokenizer.eos_token = "<|end_of_text|>"
# tokenizer.eos_token_id = 128001

# tokenizer.pad_token = tokenizer.eos_token

# tokenizer.pad_token = "<|reserved_special_token_0|>"
# tokenizer.pad_token_id = 128002

msgs = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": f"What is your name?"},
    {"role": "assistant", "content": "Bob."}
]

ids = tokenizer.apply_chat_template(msgs, tokenize=False)
print(ids)

ids = tokenizer.tokenize("Testing tokenizer.", add_special_tokens=True)
print(ids)

text_streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=False)

txt = "There is a book on the table. Look book"
chat = [
    {"role": "user", "content": "Edit the following imperative sentence for spelling and grammar mistakes: {txt}"},
]

formatted_prompt = tokenizer.apply_chat_template(
    chat, tokenize=False, add_generation_prompt=True, return_tensors="pt"
)

inputs = tokenizer(formatted_prompt, return_tensors="pt").to("cuda")

outputs = model.generate(
    **inputs,
    streamer=text_streamer,
    do_sample=True,
    max_new_tokens=500,
)