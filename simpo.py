from datasets import load_dataset, DatasetDict
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments

import os

from trl import (
    CPOConfig, 
    CPOTrainer, 
    ScriptArguments,
    ModelConfig,
    ScriptArguments,
    TrlParser
)

from alignment import get_peft_config

os.environ["WANDB_PROJECT"] = "preference_optimization"

def main():
    parser = TrlParser((ScriptArguments, CPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()

    print("script_args")
    print(script_args)

    print("training_args")
    print(training_args)

    print("model_args")
    print(model_args)
    # ################
    # # Model & Tokenizer
    # ################
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path
    )
    #print(script_args)
    ################
    # Dataset
    ################
    print('DATASET')
    print(os.getcwd())
    #print(model_args.dataset_name)
    dataset = load_dataset("csv", data_files=script_args.dataset_name)

    train_test_split = dataset.train_test_split(test_size=0.2, seed=42)

    dataset = DatasetDict({
        "train": train_test_split["train"],
        "test": train_test_split["test"]
    })

    print(dataset)
    # print(type(dataset['train'][0]['chosen']))

#     ################
#     # Training
#     ################
#     trainer = CPOTrainer(
#         model,
#         args=training_args,
#         train_dataset=dataset['train'],
#         eval_dataset=dataset['test'],
#         processing_class=tokenizer,
#         peft_config=get_peft_config(model_args),
#     )

#     # train and save the model
        
#     train_result = trainer.train()
#     metrics = train_result.metrics
#     metrics["train_samples"] = len(dataset["train"])
#     trainer.log_metrics("train", metrics)
#     trainer.save_metrics("train", metrics)
#     trainer.save_state()

#     # Save and push to hub
#     trainer.save_model(training_args.output_dir)
#     if training_args.push_to_hub:
#         trainer.push_to_hub(dataset_name=script_args.dataset_name)

if __name__ == "__main__":
    main()