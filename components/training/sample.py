import argparse
import logging
import torch
from components.data import tokenizer
from components.utils import gcs_utils
from components.training.train import Trainer, load_checkpoint
from components.config.Config import ConfigGPT, override_config_with_args

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Sampling Configuration Overrides")

    # General settings
    parser.add_argument('--blob_name', type=str, help='Blob name')
    parser.add_argument('--sample_input_text', type=str, help='Sample input text')
    parser.add_argument('--max_new_tokens', type=int, help='Maximum new tokens to generate')
    return parser.parse_args()

def main():
    logger.info("Starting the script...")

    # Load default configuration
    config = ConfigGPT()
    logger.info("Initial configuration: %s", config)
    
    # Parse command-line arguments
    args = parse_args()

    # Override configuration with command-line arguments
    config = override_config_with_args(config, args)
    logger.info("Configuration after overrides: %s", config)
    
    logger.info("Initializing trainer...")
    trainer = Trainer(config)

    logger.info("Preparing tokenizer...")
    tokenizer_instance = tokenizer.TextTokenizer(tokenizer_type=trainer.meta['tokenizer_type'], stoi=trainer.stoi, itos=trainer.itos)
 
    logger.info("Loading model checkpoint...")
    model, _, _, _ = load_checkpoint(config.device)
    logger.info("Model loaded.")

    logger.info("Generating text sample...")
    encoded_input = tokenizer_instance.encode(config.sample_input_text)
    context = torch.tensor(encoded_input, dtype=torch.long, device=config.device).unsqueeze(0)
    generated_encoded_text = model.generate(context, max_new_tokens=config.max_new_tokens)[0].tolist()
    
    generated_text = tokenizer_instance.decode(generated_encoded_text)
    logger.info("Generated text sample: %s", generated_text)
    logger.info("Text generation complete.")

if __name__ == "__main__":
    main()
