import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import fire
import soundfile as sf
import shutil
import numpy as np
import torch
import random
import datasets
import torch.nn.functional as F
from dataclasses import dataclass
import torchaudio
from transformers import LlamaTokenizer, WhisperFeatureExtractor

# Initialize logging
logger = logging.getLogger(__name__)
device = 'cuda:0'

# Initialize models
import s3prl.hub as hub
tera_model = getattr(hub, 'tera')().to(device)

def extract_tera_embeddings(waveforms, target_length=2048):
    """
    Extract TERA embeddings from waveforms. 
    Ensure each waveform is of the target length by padding or truncating.
    """
    embeddings = []
    for waveform in waveforms:
        if waveform.size(0) > 1:
            waveform = waveform[0, :]
        waveform = waveform.squeeze()
        if waveform.size(0) < target_length:
            waveform = F.pad(waveform, (0, target_length - waveform.size(0)), mode='constant', value=0)
        else:
            waveform = waveform[:target_length]
        waveform = waveform.to(device)  
        wavs = [waveform]
        with torch.no_grad():
            reps = tera_model(wavs)["hidden_states"][-1]  
            embeddings.append(reps)
    return embeddings

def process_dataset(batch, tokenizer):
    """
    Process each dataset batch by embedding audio and text, then formatting the data.
    """
    logger.info(f"Processing batch: {batch}")
    
    # Define instruction for the model
    instructions = (
        'You are an assistant skilled in analyzing children with autism through conversation analysis.\n'
        'Please score the following categories based on the provided embeddings and context information:\n'
        'CONV = [score of Conversation]\n'
        'QSOV = [score of Quality of initiating social intent]\n'
        'QSR = [score of Quality of social response]\n'
        'ARSC = [score of Mutual communication quantity]\n'
        'QQR = [score of Overall quality of relationships]\n'
        'Please ONLY SCORES = 0 or 1 or 2 or 3 are accepted, each score with the following meaning:\n'
        '0 points = No specific symptoms in this category or if no relevant information is available for this category\n'
        '1 point = There are some minor problems in this category\n'
        '2 points = There are some moderate problems in this category\n'
        '3 points = There are some severe problems in this category\n'
        'Please generate ADOS scores in the following format and fill in the scores where indicated with [?]. DO NOT PROVIDE ANY OTHER INFORMATION THANK YOU!:\n'
        'CONV = [?], QSOV= [?], QSR = [?], ARSC = [?], QQR = [?] \n'
    )
    prompt = f"Below is an instruction that describes a task, paired with an input. The input consists of text embedding, speech embedding, text reciprocity, and speech reciprocity. Based on this information, respond appropriately to complete the request.\n### Instruction:\n{instructions}\n### Input:\n"
    
    input_ids = tokenizer(f"###[Human]:{prompt}").input_ids
    attention_mask = [1] * len(input_ids)
    labels = [-100] * len(input_ids)
    
    audio_paths_turns = batch["audio_file"].split('+')
    text_inputs_turns = batch['input'].split('+')
    turn_embeddings = {f"turn{i+1}_{suffix}": None for i in range(len(audio_paths_turns)) for suffix in ["x1", "x2"]}
    text_embeddings = {f"turn{i+1}_{suffix}": None for i in range(len(text_inputs_turns)) for suffix in ["text_1", "text_2"]}
    
    for turn_index, (turn_audio_paths, turn_text_inputs) in enumerate(zip(audio_paths_turns, text_inputs_turns)):
        audio_paths = turn_audio_paths.split(',')
        text_inputs = turn_text_inputs.split(',')
        
        # Load and process waveforms
        waveforms_entrainment = [torchaudio.load(audio_file)[0] for audio_file in audio_paths]
        
        try:
            # Extract embeddings for each turn
            embeddings = extract_tera_embeddings(waveforms_entrainment)
            x1, x2 = embeddings[0].to(device), embeddings[1].to(device)
            turn_embeddings[f"turn{turn_index+1}_x1"] = x1
            turn_embeddings[f"turn{turn_index+1}_x2"] = x2
        except Exception as e:
            logger.error(f"Error while computing entrainment embedding for turn {turn_index+1}: {e}", exc_info=True)
            batch["is_readable"] = False
            continue

        # Tokenize text inputs
        if len(text_inputs) >= 2:
            text_embeddings[f"turn{turn_index+1}_text_1"] = tokenizer(text_inputs[0]).input_ids
            text_embeddings[f"turn{turn_index+1}_text_2"] = tokenizer(text_inputs[1]).input_ids

    batch.update(turn_embeddings)
    batch.update(text_embeddings)

    # Process and format labels
    suffix_input_ids = tokenizer("\n\n\n###[Assistant]:").input_ids[1:]
    suffix_attention_mask = [1] * len(suffix_input_ids)
    suffix_labels = [-100] * len(suffix_input_ids)
    
    new_input_ids = tokenizer(batch["output"]).input_ids[1:] + [tokenizer.eos_token_id]
    suffix_input_ids += new_input_ids
    suffix_attention_mask += [1] * len(new_input_ids)
    suffix_labels += new_input_ids

    batch.update({
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "suffix_input_ids": suffix_input_ids,
        "suffix_attention_mask": suffix_attention_mask,
        "suffix_labels": suffix_labels
    })

    return batch

def load_speech_text_paired_dataset(
    dataroot="",
    manifest_files="",
    tokenizer=None,
    instruction="",
    num_proc=1,
):  
    """
    Load and preprocess the dataset, saving the processed version to disk.
    """
    processed_path = os.path.join(dataroot, f"processed_{manifest_files}")
    if os.path.exists(processed_path):
        logger.warning("Loading processed dataset")
        return datasets.load_from_disk(processed_path)
    
    logger.warning(f"Loading dataset from scratch from {dataroot}/{manifest_files}")
    
    manifest_files_list = manifest_files.split(",")
    raw_dataset = datasets.load_dataset(
        'json', data_dir=dataroot, data_files=manifest_files_list, split="train", streaming=False
    )

    dataset = raw_dataset.map(
        process_dataset,
        fn_kwargs={"tokenizer": tokenizer, "instruction": instruction},
        remove_columns=raw_dataset.column_names,
        load_from_cache_file=False,
        num_proc=num_proc
    )

    dataset = dataset.filter(lambda flag: flag, input_columns=["is_readable"])
    dataset.save_to_disk(processed_path)

    return dataset

def collate_tokens(values: List[List[int]], pad_id: int):
    """
    Collate lists of token ids into a tensor with padding.
    """
    size = max(len(v) for v in values)
    batch_size = len(values)
    res = torch.LongTensor(batch_size, size).fill_(pad_id)
    
    for i, v in enumerate(values):
        res[i, :len(v)] = torch.LongTensor(v)

    return res

@dataclass
class SpeechTextPairedDataCollator:
    pad_id: int = 0
    
    def __call__(self, samples: List[Dict]):
        """
        Convert list of samples to tensors for model input.
        """
        input_ids = [sample["input_ids"] for sample in samples]
        attention_mask = [sample["attention_mask"] for sample in samples]
        labels = [sample["labels"] for sample in samples]
        suffix_input_ids = [sample["suffix_input_ids"] for sample in samples]
        suffix_attention_mask = [sample["suffix_attention_mask"] for sample in samples]
        suffix_labels = [sample["suffix_labels"] for sample in samples]
        score_labels = [sample["score_labels"] for sample in samples]
        
        turn_x1 = {f"turn{i+1}_x1": [sample.get(f"turn{i+1}_x1", []) for sample in samples] for i in range(3)}
        turn_x2 = {f"turn{i+1}_x2": [sample.get(f"turn{i+1}_x2", []) for sample in samples] for i in range(3)}
        turn_text_1 = {f"turn{i+1}_text_1": [sample.get(f"turn{i+1}_text_1", []) for sample in samples] for i in range(3)}
        turn_text_2 = {f"turn{i+1}_text_2": [sample.get(f"turn{i+1}_text_2", []) for sample in samples] for i in range(3)}

        return {
            "input_ids": collate_tokens(input_ids, self.pad_id),
            "attention_mask": collate_tokens(attention_mask, 0),
            "labels": collate_tokens(labels, -100),
            "suffix_input_ids": collate_tokens(suffix_input_ids, self.pad_id),
            "suffix_attention_mask": collate_tokens(suffix_attention_mask, 0),
            "suffix_labels": collate_tokens(suffix_labels, -100),
            "score_labels": collate_tokens(score_labels, -100),
            **{f"turn_{key}": collate_tokens(val, 0) for key, val in turn_x1.items()},
            **{f"turn_{key}": collate_tokens(val, 0) for key, val in turn_x2.items()},
            **{f"turn_{key}": collate_tokens(val, 0) for key, val in turn_text_1.items()},
            **{f"turn_{key}": collate_tokens(val, 0) for key, val in turn_text_2.items()},
        }

def offline_process(
    dataroot: str,
    manifest_files: str,
    lm_path: str,
    instruction: str,
    num_proc: int,
):
    """
    Process the dataset offline and save the results.
    """
    logger.info(f"Loading tokenizer from {lm_path}")
    text_tokenizer = LlamaTokenizer.from_pretrained(lm_path)
    dataset = load_speech_text_paired_dataset(
        dataroot=dataroot,
        manifest_files=manifest_files,
        tokenizer=text_tokenizer,
        instruction=instruction,
        num_proc=num_proc
    )
    collator = SpeechTextPairedDataCollator()
    
    processed_dir = os.path.join(dataroot, f"processed_{manifest_files}")
    os.makedirs(processed_dir, exist_ok=True)
    
    logger.info(f"Processing complete, dataset saved to {processed_dir}")

if __name__ == "__main__":
    fire.Fire(offline_process)
