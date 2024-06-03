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
    parser.add_argument('--out_dir', type=str, help='Output directory')
    parser.add_argument('--eval_interval', type=int, help='Evaluation interval')
    parser.add_argument('--log_interval', type=int, help='Log interval')
    parser.add_argument('--eval_iters', type=int, help='Number of iterations for evaluation')
    parser.add_argument('--eval_only', action='store_true', help='Run evaluation only')
    parser.add_argument('--always_save_checkpoint', type=bool, help='Always save checkpoint')
    parser.add_argument('--init_from', type=str, help='Initialization method')
    parser.add_argument('--dataset', type=str, help='Dataset to use')
    parser.add_argument('--block_size', type=int, help='Block size')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--n_layer', type=int, help='Number of layers')
    parser.add_argument('--n_head', type=int, help='Number of attention heads')
    parser.add_argument('--n_embd', type=int, help='Embedding size')
    parser.add_argument('--dropout', type=float, help='Dropout rate')
    parser.add_argument('--bias', type=bool, help='Use bias in layers')
    parser.add_argument('--learning_rate', type=float, help='Learning rate')
    parser.add_argument('--max_iters', type=int, help='Maximum number of iterations')
    parser.add_argument('--decay_lr', type=bool, help='Decay learning rate')
    parser.add_argument('--overwrite_video_captions', type=bool, help='Overwrite video captions')
    parser.add_argument('--min_lr', type=float, help='Minimum learning rate')
    parser.add_argument('--grad_clip', type=float, help='Gradient clipping value')
    parser.add_argument('--seed_offset', type=int, help='Seed offset')
    parser.add_argument('--device', type=str, help='Device to run the training on')
    parser.add_argument('--dtype', type=str, help='Data type')
    parser.add_argument('--compile', type=bool, help='Compile model')
    parser.add_argument('--model_name', type=str, help='Model name')
    parser.add_argument('--url', type=str, help='URL for the dataset')
    parser.add_argument('--id', type=str, help='ID')
    parser.add_argument('--id_type', type=str, help='ID type')
    parser.add_argument('--language', type=str, help='Language')
    parser.add_argument('--blob_name', type=str, help='Blob name')
    parser.add_argument('--project_id', type=str, help='Project ID')
    parser.add_argument('--bucket_name', type=str, help='Bucket name')
    parser.add_argument('--development_percentage', type=int, help='Development percentage')
    parser.add_argument('--train_percentage', type=float, help='Train percentage')
    parser.add_argument('--tokenizer_type', type=str, help='Tokenizer type')
    parser.add_argument('--max_number_videos', type=int, help='Maximum number of videos')
    parser.add_argument('--sample_input_text', type=str, help='Sample input text')

    return parser.parse_args()

def main():
    # Load default configuration
    config = ConfigGPT()
    print("Initial configuration:\n", config)
    # Parse command-line arguments
    args = parse_args()
    # Override configuration with command-line arguments
    config = override_config_with_args(config, args)
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
