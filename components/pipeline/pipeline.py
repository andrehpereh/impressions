import os
import torch
import logging
import argparse
from components.data import tokenizer
from components.utils import gcs_utils
from components.training.train import Trainer
from components.config.Config import ConfigGPT, override_config_with_args
from components.data import youtube_captions, prepare_text_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import argparse
import torch

def parse_args():
    parser = argparse.ArgumentParser(description="Training Configuration Overrides")

    # General settings
    parser.add_argument('--out_dir', type=str, default='out', help='Output directory')
    parser.add_argument('--eval_interval', type=int, default=200, help='Evaluation interval')
    parser.add_argument('--log_interval', type=int, default=1, help='Log interval')
    parser.add_argument('--eval_iters', type=int, default=200, help='Number of iterations for evaluation')
    parser.add_argument('--eval_only', action='store_true', help='Run evaluation only')
    parser.add_argument('--always_save_checkpoint', type=bool, default=True, help='Always save checkpoint')
    parser.add_argument('--init_from', type=str, default='scratch', help='Initialization method')
    parser.add_argument('--dataset', type=str, default='openwebtext', help='Dataset to use')
    parser.add_argument('--block_size', type=int, default=128, help='Block size')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--n_layer', type=int, default=6, help='Number of layers')
    parser.add_argument('--n_head', type=int, default=32, help='Number of attention heads')
    parser.add_argument('--n_embd', type=int, default=128, help='Embedding size')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate')
    parser.add_argument('--bias', type=bool, default=False, help='Use bias in layers')
    parser.add_argument('--learning_rate', type=float, default=6e-4, help='Learning rate')
    parser.add_argument('--max_iters', type=int, default=600, help='Maximum number of iterations')
    parser.add_argument('--decay_lr', type=bool, default=True, help='Decay learning rate')
    parser.add_argument('--overwrite_video_captions', type=bool, default=False, help='Overwrite video captions')
    parser.add_argument('--min_lr', type=float, default=6e-5, help='Minimum learning rate')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='Gradient clipping value')
    parser.add_argument('--seed_offset', type=int, default=0, help='Seed offset')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to run the training on')
    parser.add_argument('--dtype', type=str, default='bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16', help='Data type')
    parser.add_argument('--compile', type=bool, default=True, help='Compile model')
    parser.add_argument('--model_name', type=str, default='GPT-Impressions', help='Model name')
    parser.add_argument('--url', type=str, default='https://www.youtube.com/watch?v=OuF9weSkS68&list=PL0iVR8sl9TiWqj_JmVjEgAzl4YhtTt9Wf', help='URL for the dataset')
    parser.add_argument('--id', type=str, default='OuF9weSkS68', help='ID')
    parser.add_argument('--id_type', type=str, default='playlist', help='ID type')
    parser.add_argument('--language', type=str, default='en', help='Language')
    parser.add_argument('--blob_name', type=str, default='ColdFusion', help='Blob name')
    parser.add_argument('--project_id', type=str, default='metal-sky-419309', help='Project ID')
    parser.add_argument('--bucket_name', type=str, default='metal-sky-419309-videos-v1', help='Bucket name')
    parser.add_argument('--development_percentage', type=int, default=1, help='Development percentage')
    parser.add_argument('--train_percentage', type=float, default=0.9, help='Train percentage')
    parser.add_argument('--tokenizer_type', type=str, default='gpt-4', help='Tokenizer type')
    parser.add_argument('--max_number_videos', type=int, default=20, help='Maximum number of videos')
    parser.add_argument('--sample_input_text', type=str, default='Welcome to the world of AI', help='Sample input text')

    return parser.parse_args()

def main():
    # Load default configuration
    config = ConfigGPT()
    print("Initial configuration:\n", config)
    # Parse command-line arguments
    args = parse_args()
    # Override configuration with command-line arguments
    override_config_with_args(config, args)
    # Display the final configuration
    print("Final configuration:\n", config)

    # Download video data
    print("Checking if video data exists in bucket...")
    if not gcs_utils.blob_exists(config.bucket_name, config.blob_name) or config.overwrite_video_captions:
        logger.info("Video data not found in bucket. Downloading...")
        videos = youtube_captions.get_videos_and_captions(config.url, config.id_type, config.language)
        if videos:
            gcs_utils.save_dict_to_gcs(videos, bucket_name=config.bucket_name, blob_name = config.blob_name)
        else:
            logger.error("No video data to save.")
        print("Video data downloaded.")

    print("Preparing data...")
    prepare_text_data.process_captions_and_prepare_data(
        config.bucket_name, config.blob_name, config.development_percentage,
        config.train_percentage, config.tokenizer_type, config.language
    )
    print("Data preparation complete.")

    print("Training...")
    trainer = Trainer(config)
    model = trainer.train()
    print("Training complete.")


    print("Sample generated text:")
    tokenizer_instance = tokenizer.TextTokenizer(tokenizer_type=trainer.meta['tokenizer_type'], stoi=trainer.stoi, itos=trainer.itos)
    encoded_input = tokenizer_instance.encode(config.sample_input_text)
    context = torch.tensor(encoded_input, dtype=torch.long, device=config.device)
    context = context.unsqueeze(0)
    generated_encoded_text = model.generate(context, max_new_tokens=50)[0].tolist()
    print(tokenizer_instance.decode(generated_encoded_text))
    print("Sample generated text complete.")

if __name__ == "__main__":
    main()
