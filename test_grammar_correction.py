import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import outlines
from outlines import samplers
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from transformers import TextStreamer, BitsAndBytesConfig
from peft import PeftModel

set_seed(1234)

if torch.cuda.is_bf16_supported():
  compute_dtype = torch.bfloat16
else:
  compute_dtype = torch.float16

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
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map='auto',
    torch_dtype="auto",
    quantization_config=bnb_config,
    attn_implementation=attn_implementation
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

# print(tokenizer.eos_token, tokenizer.eos_token_id, tokenizer.pad_token, tokenizer.pad_token_id)

peft_model_id = "/home/stefanwebb/code/python/train-sentence-classifier/stefans-debug-llama3-grammatical-error-correction-2/checkpoint-300" # 864"
model = PeftModel.from_pretrained(model, peft_model_id)


# prompt = """You are a sentiment-labelling assistant.
# Is the following review positive or negative?

# Review: This restaurant is so-so!
# """

# generator = outlines.generate.choice(model, ["Positive", "Negative"])
# answer = generator(prompt)

sample = "?"
prompt = f"There is a button. Press"

chat = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"Correct any grammatical errors in the following text, keeping the wording as close to the original as possible:\n\n{prompt}"},
]

formatted_prompt = tokenizer.apply_chat_template(
    chat, tokenize=False, add_generation_prompt=True, return_tensors="pt"
)

text_streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=False)

inputs = tokenizer(formatted_prompt, return_tensors="pt").to("cuda")
outputs = model.generate(
    **inputs,
    streamer=text_streamer,
    pad_token_id=tokenizer.eos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    do_sample=True,
    max_new_tokens=500,
)


# sampler = samplers.greedy()
# generator = outlines.generate.choice(
#     model, ["imperative", "declarative", "interrogative", "exclamative"], sampler
# )

# for _ in range(3):
#     answer = generator(formatted_prompt)
#     print(f"Answer: {answer}")
