import pandas as pd
import os
import json
from tqdm import tqdm
from sklearn.model_selection import train_test_split
##### need to change #####
# Read transcript data
transcript_path = 'path to your transcript file'
transcript_data = []
# Read scores data
scores_path = 'path to your score file'
scores_df = pd.read_csv(scores_path)

# Create JSONL data
audio_folder = 'path to your audio folder'
jsonl_data = []
# Define paths for training and testing JSONL files
jsonl_train_output_path = 'ADOS_train_5_turn_overlap.jsonl'
jsonl_test_output_path = 'ADOS_test_5_turn_overlap.jsonl'
#########################
with open(transcript_path, 'r', encoding='utf-8') as f:
    for line in f:
        filename, start_time, end_time, transcript, speaker = line.strip().split('\t')
        transcript_data.append({
            "filename": filename,
            "start_time": float(start_time),
            "end_time": float(end_time),
            "transcript": transcript,
            "speaker": speaker
        })


# Process combined dialogue data
case_data = {}
for item in tqdm(transcript_data):
    filename = item['filename']
    case_name = '_'.join(filename.split('_')[:-2])
    transcript = item['transcript']
    speaker = item['speaker'].split('_')[-1]

    # Determine speaker identity
    if speaker == 'K':
        speaker = "Kid"
    elif speaker == 'D':
        speaker = "Doctor"

    transcript_with_speaker = f"{speaker}: {transcript}"
    
    if case_name not in case_data:
        case_data[case_name] = {'transcripts': [], 'audio_files': []}
        
    case_data[case_name]['transcripts'].append(transcript_with_speaker)
    case_data[case_name]['audio_files'].append(os.path.join(audio_folder, filename + '.wav'))

# Split into training and testing datasets
case_names = list(case_data.keys())
train_case_names, test_case_names = train_test_split(case_names, test_size=0.2, random_state=42)

# Ensure the same case name does not appear in both sets
train_case_names_set = set(train_case_names)
test_case_names_set = set(test_case_names)



def write_jsonl_with_simple_overlap(case_names, output_path, turns_per_segment=5, sentences_per_turn=2):
    with open(output_path, "w", encoding='utf-8') as fout:
        for case_name in case_names:
            content = case_data[case_name]
            transcripts = content['transcripts']
            audio_files = content['audio_files']

            # Each turn contains a specified number of sentences, overlapping turns
            num_sentences = turns_per_segment * sentences_per_turn
            for i in range(0, len(transcripts) - num_sentences + 1):
                combined_transcript_parts = []
                combined_audio_parts = []

                for j in range(turns_per_segment):
                    start_idx = i + j * sentences_per_turn
                    end_idx = start_idx + sentences_per_turn
                    if end_idx <= len(transcripts):
                        turn_transcripts = transcripts[start_idx:end_idx]
                        turn_audio_files = audio_files[start_idx:end_idx]

                        combined_turn_transcript = ','.join(turn_transcripts)
                        combined_turn_audio = ','.join(turn_audio_files)

                        combined_transcript_parts.append(combined_turn_transcript)
                        combined_audio_parts.append(combined_turn_audio)

                # Combine every five turns into one record
                combined_transcript = '+'.join(combined_transcript_parts)  # Use '+' to separate different turns
                combined_audio = '+'.join(combined_audio_parts)  # Use '+' to separate audio files from different turns
                
                # Find corresponding scores
                scores = scores_df[scores_df['name'] == case_name]
                if not scores.empty:
                    score_row = scores.iloc[0].to_dict()  # Get all scores for the case
                    label = {
                        "CONV": score_row.get("CONV", 0),
                        "QSOV": score_row.get("QSOV", 0),
                        "QSR": score_row.get("QSR", 0),
                        "ARSC": score_row.get("ARSC", 0),
                        "OQR": score_row.get("OQR", 0),
                        
                    }
                    label_output = ', '.join(f"{k} = {v}" for k, v in label.items())
                else:
                    label_output = None

                fout.write(json.dumps({
                    "case_name": case_name,
                    "input": combined_transcript,
                    "output": label_output if label_output else "",
                    "audio_file": combined_audio
                }, ensure_ascii=False) + '\n')

# Execute the modified function to write training and testing JSONL files
write_jsonl_with_simple_overlap(train_case_names, jsonl_train_output_path)
write_jsonl_with_simple_overlap(test_case_names, jsonl_test_output_path)

print(f"Train JSONL file created at: {jsonl_train_output_path}")
print(f"Test JSONL file created at: {jsonl_test_output_path}")
