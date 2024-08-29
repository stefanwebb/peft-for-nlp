# peft-for-nlp

This is an on-going project to train small PeFT adapters for simple NLP tasks like sentence classification, extracting the subject, counting number of clauses, grammar correction, sentence rewriting, etc.

My strategy is to generate synthetic data with `GPT-4` and train a small LoRA adapter on a 7B or 8B open-source LLM. I quantize the model (i.e. perform QLoRA) and train with either SFT or a preference learning objective.
