

CUDA_VISIBLE_DEVICES=1 python SocialRecNet/inference.py\
            --input_file path to test jsonl file \
            --output_file path to save result \
            --SocialRecNet_model path to your model \
            --peft_model_id path to your model/llm \
            --llama_model "path to llama model"


