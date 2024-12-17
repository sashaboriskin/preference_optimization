from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments

import hydra
from omegaconf import DictConfig, OmegaConf
from trl import CPOConfig, CPOTrainer, ScriptArguments
from trl.trainer.utils import SIMPLE_CHAT_TEMPLATE
import os

from trl import (
    KTOConfig,
    KTOTrainer,
    ModelConfig,
    ScriptArguments,
    TrlParser
)

from alignment import (
    get_peft_config,
    H4ArgumentParser,
    ModelArguments,
    DataArguments,
)

os.environ["WANDB_PROJECT"] = "preference_optimization"

def main():
    parser = TrlParser((ModelArguments, DataArguments, CPOConfig))
    model_args, data_args, training_args = parser.parse()

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

    ################
    # Dataset
    ################
    dataset = load_from_disk(list(data_args.dataset_mixer.keys())[0])

    print(dataset)
    print(type(dataset['train'][0]['chosen']))

    ################
    # Training
    ################
    trainer = CPOTrainer(
        model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
    )

    # train and save the model
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(raw_datasets["train"])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)

if __name__ == "__main__":
    main()