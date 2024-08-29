import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from transformers import TextStreamer


set_seed(1234)

"""
    FPHam/Karen_TheEditor_V2_CREATIVE_Mistral_7B
"""
model_checkpoint = "FPHam/Karen_TheEditor_V2_CREATIVE_Mistral_7B"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

model = AutoModelForCausalLM.from_pretrained(
    model_checkpoint, torch_dtype=torch.bfloat16, device_map="cuda"
)

text_streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

txt = "There is a book on the table. Look book"
chat = [
    {"role": "user", "content": "Edit the following imperative sentence for spelling and grammar mistakes: {txt}"},
]

formatted_prompt = tokenizer.apply_chat_template(
    chat, tokenize=False, add_generation_prompt=True, return_tensors="pt"
)

formatted_prompt = f"""<|im_start|>system
<|im_end|>
<|im_start|>user
Edit the following text for spelling and grammar mistakes: {txt} <|im_end|>
<|im_start|>assistant"""

inputs = tokenizer(formatted_prompt, return_tensors="pt").to("cuda")

outputs = model.generate(
    **inputs,
    streamer=text_streamer,
    do_sample=True,
    max_new_tokens=500,
)
