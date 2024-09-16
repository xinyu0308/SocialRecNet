import os
import argparse
import json
from tqdm import tqdm
import torch
import logging
from transformers import LlamaForCausalLM
from transformers import LlamaTokenizer
from transformers import GenerationConfig
from modeling_SocialRecNet import SocialRecNet
from peft import LoraConfig, get_peft_model
from peft import PeftModel, PeftConfig
from transformers.deepspeed import is_deepspeed_zero3_enabled
import torch.nn.functional as F
import torchaudio
import numpy as np

logger = logging.getLogger(__name__)

generation_config = GenerationConfig(
    max_new_tokens=75,
    min_new_tokens=40,
    do_sample=False,
    temperature=0.1,
    top_p=0.75,
    num_beams=1,
    num_return_sequences=1,
)

device = 'cuda:0'

import s3prl.hub as hub
tera_model = getattr(hub, 'tera')().to(device)

def extract_tera_embeddings(waveforms, target_length=2048):
    embeddings = []
    for waveform in waveforms:
        print("Original waveform shape:", waveform.shape)
        
        if waveform.size(0) > 1:
            waveform = waveform[0, :]

        waveform = waveform.squeeze()
        print("Waveform after squeeze:", waveform.shape)

        if waveform.size(0) < target_length:
            padding = target_length - waveform.size(0)
            waveform = F.pad(waveform, (0, padding), mode='constant', value=0)
        
        if waveform.size(0) > target_length:
            waveform = waveform[:target_length]

        print("Waveform after padding/cropping:", waveform.shape)
        
        waveform = waveform.to(device)  # Ensure waveform is on GPU
        wavs = [waveform for _ in range(1)]
        
        with torch.no_grad():
            reps = tera_model(wavs)["hidden_states"][-1]  # TERA embeddings computation on GPU
            embeddings.append(reps)

    return embeddings

def extract_case_name(audio_path):
    base_name = os.path.basename(audio_path).split('.')[0]
    parts = base_name.split('_')
    return '_'.join(parts[:-2])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file", type=str, default="",
        help="Path to the input file", 
    )
    parser.add_argument(
        "--output_file", type=str, default="",
        help="Path to the output file", 
    )
    parser.add_argument(
        "--SocialRecNet", type=str, default="",
        help="Path to the SocialRecNet model", 
    )
    # Arguments for generation
    parser.add_argument(
        "--max_new_tokens", type=int, default=1024,
        help="Maximum new tokens for generation"
    )
    parser.add_argument(
        "--min_new_tokens", type=int, default=1,
        help="Minimum new tokens for generation"
    )
    parser.add_argument(
        "--do_sample", action="store_true",
        help="Whether to sample. For ST task, we will use greedy search to ensure stable output"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.1,
        help="Temperature for generation"
    )
    parser.add_argument(
        "--top_p", type=float, default=0.75,
        help="Top-p for generation"
    )
    parser.add_argument(
        "--peft_model_id", type=str, default="",
        help="Checkpoint of PEFT model"
    )
    parser.add_argument(
        "--llama_model", type=str, default="",
        help="Checkpoint of Llama model"
    )
    args = parser.parse_args()

    tokenizer = LlamaTokenizer.from_pretrained(args.SocialRecNet)
    model = SocialRecNet.from_pretrained(args.SocialRecNet)
    
    generation_config.update(
        **{
            "max_new_tokens": args.max_new_tokens,
            "min_new_tokens": args.min_new_tokens,
            "do_sample": args.do_sample,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "pad_token_id": tokenizer.pad_token_id,
            "bos_token_id": tokenizer.bos_token_id,
            "eos_token_id": tokenizer.eos_token_id
        }
    )

    llama_model = LlamaForCausalLM.from_pretrained(args.llama_model, _fast_init=not is_deepspeed_zero3_enabled())
    model.llama_model = PeftModel.from_pretrained(llama_model, args.peft_model_id)
    model = model.cuda()
    model.eval()

    with open(args.input_file, "r") as fin, open(args.output_file, "w") as fout:
        for line in tqdm(fin):
            data = json.loads(line.strip())
            instruction1 = 'You are an assistant skilled in analyzing children with autism through conversation analysis.\n'
            instruction2 = 'Please score the following categories based on the provided embeddings and context information:\n'
            instruction3 = ('CONV = [score of Conversation]\n'
                            'QSOV = [score of Quality of initiating social intent]\n'
                            'QSR = [score of Quality of social response]\n'
                            'ARSC = [score of Mutual communication quantity]\n'
                            'QQR = [score of Overall quality of relationships]\n'
                            )
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
            input_ids = tokenizer(f"###[Human]:{full_prompt}", return_tensors="pt").input_ids.cuda()
            audio_paths_turns = data["audio_file"].split('+')
            output_score = data['output']
            
            is_readable = True
            text_inputs_turns = data['input'].split('+')
            turn_embeddings = {f"turn{i+1}_x1": None for i in range(len(audio_paths_turns))}
            turn_embeddings.update({f"turn{i+1}_x2": None for i in range(len(audio_paths_turns))})
            text_embeddings = {f"turn{i+1}_text_1": None for i in range(len(text_inputs_turns))}
            text_embeddings.update({f"turn{i+1}_text_2": None for i in range(len(text_inputs_turns))})
            
            for turn_index, (turn_audio_paths, turn_text_inputs) in enumerate(zip(audio_paths_turns, text_inputs_turns)):
                audio_paths = turn_audio_paths.split(',')
                text_inputs = turn_text_inputs.split(',')
                waveforms_entrainment = []
                audio_data = []
                case_name = extract_case_name(audio_paths[0])
                for audio_file in audio_paths:
                    
                    waveform, sample_rate = torchaudio.load(audio_file)
                    waveforms_entrainment.append(waveform)
                
                try:
                    embeddings = extract_tera_embeddings(waveforms_entrainment)
                    x1, x2 = embeddings[0], embeddings[1]
                    audio_data.append(x1)
                    audio_data.append(x2)
                    turn_embeddings[f"turn{turn_index+1}_x1"] = x1
                    turn_embeddings[f"turn{turn_index+1}_x2"] = x2
                    print(f"Turn {turn_index+1} - x1:", x1.shape)
                    print(f"Turn {turn_index+1} - x2:", x2.shape)
                except Exception as e:
                    logger.error(f"Error while computing entrainment embedding for turn {turn_index+1}: {e}", exc_info=True)
                    raise e

                # Process text inputs, extract paired sentences
                if len(text_inputs) >= 2:
                    # Use specific format to convert text and extract embeddings
                    text_1 = tokenizer(text_inputs[0]).input_ids
                    text_2 = tokenizer(text_inputs[1]).input_ids
                    text_embeddings[f"turn{turn_index+1}_text_1"] = text_1
                    text_embeddings[f"turn{turn_index+1}_text_2"] = text_2

            suffix_input_ids = tokenizer("\n\n\n###[Assistant]:", return_tensors="pt").input_ids[:,1:].cuda()
            # Initialize model_inputs dictionary
            model_inputs = {
                "input_ids": input_ids,
                "suffix_input_ids": suffix_input_ids,
                "generation_config": generation_config,
            }
            for key, value in turn_embeddings.items():
                model_inputs[key] = value

            # Add text embeddings to batch
            for key, value in text_embeddings.items():
                model_inputs[key] = value

            # Use model_inputs for generation
            output = model.generate(**model_inputs)
            response = tokenizer.decode(output[0], skip_special_tokens=True)

            json_string = json.dumps(
                {
                    "response": response,
                    "case_name": case_name,
                    "actual_score": output_score 
                },
                ensure_ascii=False
            )
            fout.write(json_string + "\n")
            

if __name__ == "__main__":
    main()
