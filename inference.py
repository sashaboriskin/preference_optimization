import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "models/Mistral-7B-Instruct-v0.3_bf16_dpo_v1"
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

prompts = [
    "Write a negative review on product with name Amazon 5W USB Official OEM Charger and Power Adapter for Fire Tablets and Kindle eReaders,,, Amazon 5W USB Official OEM Charger and Power Adapter for Fire Tablets and Kindle eReaders,,,",
    "Write a negative review on product with name AmazonBasics AAA Performance Alkaline Batteries (36 Count)",
]

for prompt in prompts:
    print(f"Prompt: {prompt}\n")
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[1]
    
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=100)

    output_text = tokenizer.decode(output[0][input_len:], skip_special_tokens=True)
    print(f"Generation: {output_text}\n")
