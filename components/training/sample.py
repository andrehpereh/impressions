import argparse
import logging
import os
import pickle
from contextlib import nullcontext

import torch

from components.data import tokenizer
from components.utils import gcs_utils
from components.training.model import GPTConfig, GPT
from components.training.train import Trainer, load_checkpoint
from components.config.Config import ConfigGPT, override_config_with_args

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Sampling Configuration Overrides")

    # General settings
    parser.add_argument('--blob_name', type=str, help='Blob name')
    parser.add_argument('--sample_input_text', type=str, help='Sample input text')
    parser.add_argument('--init_from', type=str, help='Sample input text')
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
    logger.info("Updated configuration with command-line arguments: %s", config)

    # Set default values for the generation parameters
    num_samples = 2
    max_new_tokens = config.max_new_tokens
    temperature = 0.7
    top_k = 300
    seed = 1337
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
    compile = True
    data_dir = os.path.join('components', 'data', config.blob_name)
    
    logger.info("Generation parameters set: device=%s, dtype=%s, data_dir=%s", device, dtype, data_dir)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    logger.info("Loading model...")

    # Load model checkpoint
    ckpt_path = os.path.join(config.checkpoint_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
    logger.info("Loaded checkpoint: iter_num=%d, best_val_loss=%f", iter_num, best_val_loss)

    state_dict = checkpoint['model_state_dict']
    unwanted_prefix = '_orig_mod.'
    for k in list(state_dict.keys()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    
    model.eval()
    model.to(device)
    if compile:
        model = torch.compile(model)  # requires PyTorch 2.0 (optional)

    logger.info("Model loaded and ready.")

    # Load metadata for tokenizer
    meta_path = os.path.join(data_dir, 'meta.pkl')
    logger.info("Loading metadata from %s...", meta_path)
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    stoi, itos = meta['stoi'], meta['itos']
    tokenizer_instance = tokenizer.TextTokenizer(tokenizer_type=meta['tokenizer_type'], stoi=stoi, itos=itos)
    logger.info("Tokenizer initialized with metadata.")

    # Encode the beginning of the prompt
    encoded_input = tokenizer_instance.encode(config.sample_input_text)
    context = torch.tensor(encoded_input, dtype=torch.long, device=config.device).unsqueeze(0)
    logger.info("Input text encoded.")

    # Run generation
    logger.info("Starting text generation...")
    with torch.no_grad():
        with ctx:
            for k in range(num_samples):
                generated_encoded_text = model.generate(context, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k)[0].tolist()
                generated_text = tokenizer_instance.decode(generated_encoded_text)
                logger.info("Generated text sample %d:", k + 1)
                print(generated_text)
                logger.info("Text generation complete.")

if __name__ == "__main__":
    main()
