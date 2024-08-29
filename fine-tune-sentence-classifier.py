import os
os.environ['HF_HOME'] = '/home/stefanwebb/models/fine-tuned'
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import gc
import multiprocessing

import torch
torch.random.manual_seed(0)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig, PeftModel
from datasets import load_dataset
from trl import ORPOTrainer, ORPOConfig

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
NEW_MODEL_NAME = "stefans-debug-llama3-sentence-classifier"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map='auto',
    torch_dtype="auto",
    quantization_config=bnb_config,
    attn_implementation=attn_implementation
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

# tokenizer.pad_token = "<|reserved_special_token_0|>"
# tokenizer.pad_token_id = 128002
# tokenizer.eos_token = "<|end_of_text|>"
# tokenizer.eos_token_id = 128001

SPLIT = "train"
MAX_SEQ_LENGTH = 2048 # 8192 # 2048

# def format_prompt(s):
#         return f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{s.strip()}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

# def format_completion(s):
#         return f"{s.strip()}<|eot_id|><|end_of_text|>"

# def process(row):
#         return {
#                 "prompt": format_prompt(row["prompt"]),
#                 "chosen": format_completion([x["content"] for x in row["chosen"] if x["role"] == "assistant"][0]),
#                 "rejected": format_completion([x["content"] for x in row["rejected"] if x["role"] == "assistant"][0]),
#         }

"""
    Dataset
"""
from datasets import load_dataset
dataset = load_dataset("csv", data_files="/home/stefanwebb/code/python/train-sentence-classifier/sentence_types.csv")

def process(x):
    chat = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"Classify the following sentence as imperative, declarative, interrogative, or exclamative:\n\n{x['prompt'].strip()}"},
    ]

    formatted_prompt = tokenizer.apply_chat_template(
        chat, tokenize=False, add_generation_prompt=True, return_tensors="pt"
    )

    # TODO: Add end tokens to completions?

    return {'prompt': formatted_prompt, 'chosen': x['chosen'].strip(), 'rejected': x['rejected'].strip()}

ds = dataset['train'].map(
        process,
        num_proc=int(multiprocessing.cpu_count() // 4),
        load_from_cache_file=False,
        # batched=True
)

ds = ds.shuffle(seed=42).flatten_indices()

"""
    LoRA
"""
lora_config = LoraConfig(
    # r=128, # Rank of weight matrices
    # lora_alpha=64,  # 8,
    r=16,
    lora_alpha=8,
    lora_dropout=0.05,    # 0.1
    target_modules="all-linear",
    modules_to_save= ["embed_tokens", "lm_head"], # <= seems to be the key 
    bias="none",
    task_type="CAUSAL_LM",
)

args = ORPOConfig(
    output_dir=NEW_MODEL_NAME, # directory to save and repository id
    num_train_epochs=3, # 3,                     # number of training epochs
    per_device_train_batch_size=1,          # batch size per device during training
    gradient_accumulation_steps=16,          # number of steps before performing a backward/update pass
    gradient_checkpointing=True,            # use gradient checkpointing to save memory
    # optim="paged_adamw_32bit",              # use fused adamw optimizer
    optim="adamw_torch_fused",
    logging_steps=10,                       # log every 10 steps
    save_strategy="steps",                  # save checkpoint every epoch
    save_steps=0.1,
    # save_strategy="no",
    bf16=True,                              # use bfloat16 precision
    tf32=True,                              # use tf32 precision
    learning_rate=2e-4,                     # learning rate, based on QLoRA paper
    max_grad_norm=0.3,                      # max gradient norm based on QLoRA paper
    warmup_ratio=0.03,                      # warmup ratio based on QLoRA paper
    lr_scheduler_type="cosine",           # use cosine learning rate scheduler
    report_to="tensorboard",                # report metrics to tensorboard
    # torch_compile=True,
    # torch_compile_backend=
    # packing=True
    max_length=1024*8,
    max_prompt_length=1024
)


trainer = ORPOTrainer(
    model=model,
    tokenizer=tokenizer,
    args=args,
    train_dataset=ds,
    # eval_dataset=ds_eval,
    peft_config=lora_config
)

# device = 'cuda'

# Comment out for now
trainer.train()
trainer.save_model()

gc.collect()
torch.cuda.empty_cache()