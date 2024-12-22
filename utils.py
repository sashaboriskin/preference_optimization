def split_chosen_rejected(example):
    return {
        "prompt": [item["content"] for item in eval(example["prompt"])][0],
        "chosen": [item["content"] for item in eval(example["completion"])["chosen"]][0],
        "rejected": [item["content"] for item in eval(example["completion"])["rejected"]][0],
    }

def split_prompt_completion_target(example): 
    return {
        "prompt": [item["content"] for item in eval(example["prompt"])][0],
        "completion": [item["content"] for item in eval(example["completion"])][0],
        "label": example["label"],
    }
