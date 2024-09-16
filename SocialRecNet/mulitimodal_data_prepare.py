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
    logger.info(f"Processing batch: {batch}")
    
    
    instruction1 = 'You are an assistant skilled in analyzing children with autism through conversation analysis.\n'
    instruction2 = 'Please score the following categories based on the provided embeddings and context information:\n'
    instruction3 = ('CONV = [score of Conversation]\n'
                    'QSOV = [score of Quality of initiating social intent]\n'
                    'QSR = [score of Quality of social response]\n'
                    'ARSC = [score of Mutual communication quantity]\n'
                    'QQR = [score of Overall quality of relationships]\n')
    instruction5 = 'Based on the information provided, generate ADOS scores according to the following criteria:\n'
    instruction6 = ('Please ONLY SCORES = 0 or 1 or 2 or 3 are accepted, each score with the following meaning:\n'
                    '0 points = No specific symptoms in this category or if no relevant information is available for this category\n'
                    '1 point = There are some minor problems in this category\n'
                    '2 points = There are some moderate problems in this category\n'
                    '3 points = There are some severe problems in this category\n')
    instruction7 = 'Please generate ADOS scores in the following format and fill in the scores where indicated with [?]. DO NOT PROVIDE ANY OTHER INFORMATION THANK YOU!:\n'
    instruction8 = ('CONV = [?], QSOV= [?], QSR = [?], ARSC = [?], QQR = [?] \n')


    prompt = "".join([instruction1, instruction2, instruction3, instruction5, instruction6, instruction7, instruction8])
    full_prompt = (f"Below is an instruction that describes a task, paired with an input. "
                f"The input consists of text embedding, speech embedding, text reciprocity, and speech reciprocity combined together. "
                f"This combined input is processed by a GRU model to provide contextual information for scoring. "
                f"Based on this information, respond appropriately to complete the request.\n"
                f"### Instruction:\n{prompt}\n"
                f"### Input:\n")
    input_ids = tokenizer(f"###[Human]:{full_prompt}").input_ids
    attention_mask = [1] * len(input_ids)
    labels = [-100] * len(input_ids)
    audio_paths_turns = batch["audio_file"].split('+')
    is_readable = True
    text_inputs_turns = batch['input'].split('+')
    turn_embeddings = {f"turn{i+1}_x1": None for i in range(len(audio_paths_turns))}
    turn_embeddings.update({f"turn{i+1}_x2": None for i in range(len(audio_paths_turns))})
    text_embeddings = {f"turn{i+1}_text_1": None for i in range(len(text_inputs_turns))}
    text_embeddings.update({f"turn{i+1}_text_2": None for i in range(len(text_inputs_turns))})
    
    for turn_index, (turn_audio_paths, turn_text_inputs) in enumerate(zip(audio_paths_turns, text_inputs_turns)):
        audio_paths = turn_audio_paths.split(',')
        text_inputs = turn_text_inputs.split(',')
        waveforms_entrainment = []

        # 讀取每個turn的音檔
        for audio_file in audio_paths:
            waveform, sample_rate = torchaudio.load(audio_file)
            waveforms_entrainment.append(waveform)
        
        try:
            # 提取音檔的tera_embeddings
            embeddings = extract_tera_embeddings(waveforms_entrainment)
            x1, x2 = embeddings[0].to(device), embeddings[1].to(device)
            turn_embeddings[f"turn{turn_index+1}_x1"] = x1
            turn_embeddings[f"turn{turn_index+1}_x2"] = x2
            print(f"Turn {turn_index+1} - x1:", x1.shape)
            print(f"Turn {turn_index+1} - x2:", x2.shape)
        except Exception as e:
            is_readable = False
            logger.error(f"Error while computing entrainment embedding for turn {turn_index+1}: {e}", exc_info=True)
            raise e

        # 處理文本輸入，提取成對的句子
        if len(text_inputs) >= 2:
            # 使用特定格式轉換文本並提取嵌入
            text_1 = tokenizer(text_inputs[0]).input_ids
            text_2 = tokenizer(text_inputs[1]).input_ids
            text_embeddings[f"turn{turn_index+1}_text_1"] = text_1
            text_embeddings[f"turn{turn_index+1}_text_2"] = text_2
    

    suffix_input_ids, suffix_attention_mask, suffix_labels = [], [], []
    new_input_ids = tokenizer("\n\n\n###[Assistant]:").input_ids[1:]
    suffix_input_ids += new_input_ids
    suffix_attention_mask += [1] * len(new_input_ids)
    suffix_labels += [-100] * len(new_input_ids)
    #response 
    new_input_ids = tokenizer(batch["output"]).input_ids[1:] + [tokenizer.eos_token_id]
    suffix_input_ids += new_input_ids
    suffix_attention_mask += [1] * len(new_input_ids)
    suffix_labels += new_input_ids
    batch["is_readable"] = is_readable

    # 將各turn的x1和x2存入batch
    for key, value in turn_embeddings.items():
        batch[key] = value

    # 將各turn的文本嵌入存入batch
    for key, value in text_embeddings.items():
        batch[key] = value
    batch["input_ids"] = input_ids
    batch["attention_mask"] = attention_mask
    batch["labels"] = labels
    batch["suffix_input_ids"] = suffix_input_ids
    batch["suffix_attention_mask"] = suffix_attention_mask
    batch["suffix_labels"] = suffix_labels

    return batch

def load_speech_text_paired_dataset(
    dataroot="",
    manifest_files="",
    tokenizer=None,
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
        fn_kwargs={"tokenizer": tokenizer},
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
    """
    Data collator that will dynamically pad the inputs received.
    """
    pad_id: int = 0
    sampling_rate: int = 16000
    extractor: WhisperFeatureExtractor = WhisperFeatureExtractor()

    def __call__(self, samples: List[Dict]):
        # 收集所有的 `input_ids`, `attention_mask`, `labels`, 和 `suffix_` 屬性
        input_ids = [sample["input_ids"] for sample in samples]
        attention_mask = [sample["attention_mask"] for sample in samples]
        labels = [sample["labels"] for sample in samples]
        suffix_input_ids = [sample["suffix_input_ids"] for sample in samples]
        suffix_attention_mask = [sample["suffix_attention_mask"] for sample in samples]
        suffix_labels = [sample["suffix_labels"] for sample in samples]
        # 初始化空的字典來存儲 turn 屬性
        turn_x1 = {f"turn{i+1}_x1": [sample.get(f"turn{i+1}_x1", []) for sample in samples] for i in range(5)}
        turn_x2 = {f"turn{i+1}_x2": [sample.get(f"turn{i+1}_x2", []) for sample in samples] for i in range(5)}
        turn_text_1 = {f"turn{i+1}_text_1": [sample.get(f"turn{i+1}_text_1", []) for sample in samples] for i in range(5)}
        turn_text_2 = {f"turn{i+1}_text_2": [sample.get(f"turn{i+1}_text_2", []) for sample in samples] for i in range(5)}

        if self.pad_id is None:
            self.pad_id = 0


        # 動態填充數據
        input_ids = collate_tokens(input_ids, self.pad_id)
        attention_mask = collate_tokens(attention_mask, 0)
        labels = collate_tokens(labels, -100)
        suffix_input_ids = collate_tokens(suffix_input_ids, self.pad_id)
        suffix_attention_mask = collate_tokens(suffix_attention_mask, 0)
        suffix_labels = collate_tokens(suffix_labels, -100)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "suffix_input_ids": suffix_input_ids,
            "suffix_attention_mask": suffix_attention_mask,
            "suffix_labels": suffix_labels,
            **turn_x1,
            **turn_x2,
            **turn_text_1,
            **turn_text_2
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
