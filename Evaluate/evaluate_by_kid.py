import pandas as pd
from sklearn.metrics import mean_absolute_error
from scipy.stats import spearmanr, pearsonr
import numpy as np
import json
import os

def concordance_correlation_coefficient(y_true, y_pred):
    """Calculate the Concordance Correlation Coefficient (CCC)."""
    dct = {'y_true': y_true, 'y_pred': y_pred}
    df = pd.DataFrame(dct)
    df = df.dropna()
    y_true = df['y_true']
    y_pred = df['y_pred']
    cor = np.corrcoef(y_true, y_pred)[0][1]
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    sd_true = np.std(y_true)
    sd_pred = np.std(y_pred)
    numerator = 2 * cor * sd_true * sd_pred
    denominator = var_true + var_pred + (mean_true - mean_pred)**2
    return numerator / denominator

def read_jsonl(file_path):
    """Read JSON Lines file and convert it to a DataFrame."""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return pd.DataFrame(data)

def parse_scores(score_string):
    """Parse score strings into a dictionary with float values."""
    score_dict = {}
    valid_keys = {'BB6', 'BB7', 'BB8', 'BB9', 'BB10', 'AA4', 'AA7', 'AA8', 'ADOS_class', 'ADOS_S'}
    try:
        if '</s>' in score_string:
            score_parts = score_string.split('</s>')
            score_string = score_parts[0]
        scores = score_string.split(', ')
        for item in scores:
            if '=' in item:
                try:
                    key, value = item.split('=')
                    key = key.strip()
                    value = value.strip()
                    if key in valid_keys and value:
                        # Convert value to float
                        score_dict[key] = float(value)
                except ValueError:
                    print(f"Error converting value '{item}' to float.")
    except Exception as e:
        print(f"Error parsing score string '{score_string}': {e}")
    return score_dict

def extract_all_scores(data_df):
    """Extract all scores for each case and organize them into dictionaries."""
    all_scores = {}
    for idx, row in data_df.iterrows():
        case_name = row['case_name']
        response_scores = parse_scores(row['response'])
        actual_scores = parse_scores(row['actual_score'])
        
        if case_name not in all_scores:
            all_scores[case_name] = {'response': {}, 'actual': {}}
        
        for key in response_scores:
            if key in all_scores[case_name]['response']:
                all_scores[case_name]['response'][key].append(response_scores[key])
            else:
                all_scores[case_name]['response'][key] = [response_scores[key]]
        
        for key in actual_scores:
            if key in all_scores[case_name]['actual']:
                all_scores[case_name]['actual'][key].append(actual_scores[key])
            else:
                all_scores[case_name]['actual'][key] = [actual_scores[key]]
    
    return all_scores

def calculate_averages(all_scores):
    """Calculate average scores for each case."""
    averages = {}
    for case_name, scores in all_scores.items():
        averages[case_name] = {
            'response': {key: np.mean(val) for key, val in scores['response'].items()},
            'actual': {key: np.mean(val) for key, val in scores['actual'].items()}
        }
    return averages

def calculate_medians(all_scores):
    """Calculate median scores for each case."""
    medians = {}
    for case_name, scores in all_scores.items():
        medians[case_name] = {
            'response': {key: np.median(val) for key, val in scores['response'].items()},
            'actual': {key: np.median(val) for key, val in scores['actual'].items()}
        }
    return medians

def compute_metrics(averages_or_medians, metric_name):
    """Compute evaluation metrics (MAE, Spearman, Pearson, CCC) for each score."""
    results = []
    all_keys = ['AA4', 'AA7', 'AA8', 'BB6', 'BB7', 'BB8', 'BB9', 'BB10', 'ADOS_class', 'ADOS_S']
    
    for key in all_keys:
        response_values = []
        actual_values = []
        
        for case_name, scores in averages_or_medians.items():
            if key in scores['response'] and key in scores['actual']:
                response_values.append(scores['response'][key])
                actual_values.append(scores['actual'][key])
        
        if len(response_values) > 0:
            mae = mean_absolute_error(actual_values, response_values)
            spearman_corr, _ = spearmanr(actual_values, response_values)
            pearson_corr, _ = pearsonr(actual_values, response_values)
            ccc = concordance_correlation_coefficient(actual_values, response_values)
            
            results.append({
                'Metric': metric_name,
                'Score': key,
                'MAE': mae,
                'Spearman': spearman_corr,
                'Pearson': pearson_corr,
                'CCC': ccc
            })
    
    return pd.DataFrame(results)

# Load data
file_path = 'path to inference result'
data_df = read_jsonl(file_path)

# Extract and calculate scores
all_scores = extract_all_scores(data_df)

# Calculate average scores
averages = calculate_averages(all_scores)
average_results_df = compute_metrics(averages, 'Average')

# Calculate median scores
medians = calculate_medians(all_scores)
median_results_df = compute_metrics(medians, 'Median')

# Get the input file name
file_name = os.path.splitext(os.path.basename(file_path))[0]

# Save results to CSV files
average_results_df.to_csv(f'{file_name}_average_case_result.csv', index=False)
median_results_df.to_csv(f'{file_name}_median_case_result.csv', index=False)

# Print results
print(average_results_df)
print(median_results_df)
