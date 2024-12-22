import os

from datasets import load_dataset, DatasetDict
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import (
    KTOConfig,
    KTOTrainer,
    ModelConfig,
    ScriptArguments,
    TrlParser
)

from alignment import get_peft_config
from utils import split_prompt_completion_target

os.environ["WANDB_PROJECT"] = "preference_optimization"

def main():
    parser = TrlParser((ScriptArguments, KTOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()

    ################
    # Model & Tokenizer
    ################
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path
    )
    tokenizer.pad_token = tokenizer.eos_token

    ################
    # Dataset
    ################
    dataset = load_dataset("csv", data_files=script_args.dataset_name)
    dataset['train'] = dataset['train'].map(split_prompt_completion_target)
    dataset = dataset['train'].train_test_split(test_size=0.2, seed=training_args.seed)

    dataset = DatasetDict({
        "train": dataset["train"],
        "test": dataset["test"]
    })
    
    ################
    # Training
    ################
    trainer = KTOTrainer(
        model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
        tokenizer=tokenizer,
        peft_config=get_peft_config(model_args),
    )
    
    train_result = trainer.train()
    metrics = train_result.metrics
    metrics["train_samples"] = len(dataset["train"])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)

if __name__ == "__main__":
    main()