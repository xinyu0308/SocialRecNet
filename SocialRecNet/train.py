#!/usr/bin/env python
# coding=utf-8
# Fine-tuning the library models for sequence-to-sequence speech recognition.

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import List
import datasets
import torch
import transformers
from transformers import (
    HfArgumentParser,
    TrainingArguments,
    Trainer,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers import LlamaTokenizer, LlamaForCausalLM, LlamaConfig
from transformers.deepspeed import is_deepspeed_zero3_enabled
from src.mulitimodal_data_prepare import load_speech_text_paired_dataset, SpeechTextPairedDataCollator
from src.modeling_SocialRecNet import SocialRecNet
from src.configuration import SocialRecNetConfig
from peft import LoraConfig, get_peft_model
import json

# Initialize logger
logger = logging.getLogger(__name__)

@dataclass
class LoraArguments:
    """
    Arguments for configuring Low-Rank Adaptation (LoRA) for model fine-tuning.
    """
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj", "k_proj"])
    bias: str = "none"

@dataclass
class ModelArguments:
    """
    Arguments for the model, including configuration and tokenizer for fine-tuning.
    """
    llama_model: str = field(default="lmsys/vicuna-13b-v1.1", metadata={"help": "Path to the base model"})

@dataclass
class DataTrainingArguments:
    """
    Arguments for data input, including paths for training and evaluation datasets.
    """
    data: str = field(metadata={"help": "Root path to load the dataset"})
    manifest_files: str = field(default="", metadata={"help": "Name of the training unit text paired set split to use."})

def setup_logging(training_args):
    """
    Configures logging based on training arguments.
    """
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

def load_model(model_args, lora_args):
    """
    Loads the Llama and SocialRecNet models, applying PEFT (Parameter-Efficient Fine-Tuning) configuration.
    """
    llama_config = LlamaConfig.from_pretrained(model_args.llama_model)
    socialrecnet_config = SocialRecNetConfig(llama_config.to_dict())
    model = SocialRecNet(socialrecnet_config)
    
    # Load base Llama model
    model.llama_model = LlamaForCausalLM.from_pretrained(
        model_args.llama_model, _fast_init=not is_deepspeed_zero3_enabled()
    )
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=lora_args.lora_r,
        lora_alpha=lora_args.lora_alpha,
        target_modules=lora_args.lora_target_modules,
        lora_dropout=lora_args.lora_dropout,
        bias=lora_args.bias,
        task_type="CAUSAL_LM",
    )
    
    # Apply LoRA configuration
    model.llama_model = get_peft_model(model.llama_model, lora_config)
    for param in model.llama_model.parameters():
        param.requires_grad = False

    return model

def main():
    # Parse input arguments
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments, LoraArguments))
    model_args, data_args, training_args, lora_args = parser.parse_args_into_dataclasses()

    # Setup logging
    setup_logging(training_args)

    logger.info(f"Training/evaluation parameters: {training_args}")
    logger.info(f"Model parameters: {model_args}")
    logger.info(f"Dataset parameters: {data_args}")

    # Detect last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to train from scratch."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training from {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model
    set_seed(training_args.seed)

    # Load tokenizer and dataset
    tokenizer = LlamaTokenizer.from_pretrained(model_args.llama_model)
    dataset = load_speech_text_paired_dataset(
        dataroot=data_args.data,
        manifest_files=data_args.manifest_files,
        tokenizer=tokenizer
    )

    # Load model with PEFT
    model = load_model(model_args, lora_args)

    # Define data collator
    data_collator = SpeechTextPairedDataCollator(pad_id=tokenizer.pad_token_id)

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    # Train the model if specified
    if training_args.do_train:
        checkpoint = training_args.resume_from_checkpoint if training_args.resume_from_checkpoint else last_checkpoint
        try:
            train_result = trainer.train(resume_from_checkpoint=checkpoint)
            trainer.save_model()
            trainer.log_metrics("train", train_result.metrics)
            trainer.save_metrics("train", train_result.metrics)
            trainer.save_state()
        except ZeroDivisionError:
            logger.error("ZeroDivisionError encountered during training. Please check your training setup.")

    # Save tokenizer and model for inference
    tokenizer.save_pretrained(training_args.output_dir)
    llm_path = os.path.join(training_args.output_dir, 'llm')
    os.makedirs(llm_path, exist_ok=True)
    model.llama_model.save_pretrained(llm_path)

if __name__ == "__main__":
    main()
