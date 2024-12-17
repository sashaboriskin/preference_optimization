import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import argparse

def load_model(checkpoint_path, rev='main'):
    model = AutoModelForCausalLM.from_pretrained(checkpoint_path,
                                                 revision=rev,
                                                 torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    return model, tokenizer

def merge(model_path, adapter_path, new_model_path, rev='main'):
    model, tokenizer = load_model(model_path, rev)
    finetuned_model = PeftModel.from_pretrained(model=model, 
                                                model_id=adapter_path)
    
    finetuned_model = finetuned_model.merge_and_unload()
    
    finetuned_model.save_pretrained(new_model_path)
    tokenizer.save_pretrained(new_model_path)

def main():
    parser = argparse.ArgumentParser(description="LoRA merge script")
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--model_rev', type=str, required=False)
    parser.add_argument('--adapter_path', type=str, required=True)
    parser.add_argument('--new_model_path', type=str, required=True)
    args = parser.parse_args()

    merge(args.model_path, args.adapter_path, args.new_model_path, args.model_rev)

if __name__ == '__main__':
    main()