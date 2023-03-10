from transformers import pipeline

model_name = "EleutherAI/gpt-neo-2.7B"
max_new_tokens = 50

pipeline(model=model_name, model_kwargs={"device_map": "auto", "load_in_8bit": True}, max_new_tokens=max_new_tokens)