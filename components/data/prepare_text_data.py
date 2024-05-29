import os
import json
import argparse
import numpy as np
import pickle
from components.utils.gcs_utils import download_from_gcs
from components.data import tokenizer

def download_and_load_captions(bucket_name, blob_name):
    """
    Download and load captions from GCS.
    """
    content = download_from_gcs(bucket_name, blob_name, read_only=True)
    return json.loads(content)

def get_development_content(content, development_percentage):
    """
    Reduce sample size for development based on the development percentage.
    """
    assert 0 <= development_percentage <= 1, "development_percentage must be between 0 and 1."
    if development_percentage:
        return list(content.items())[-int(len(content) * development_percentage):]
    return None

def create_master_captions(development_content, language="en"):
    """
    Create master captions with all the subtitles.
    """
    master_text = ""
    if development_content and isinstance(development_content, list):
        for video in development_content:
            try:
                video_id, video_data = video
                captions = video_data.get(f"captions_{language}", None)
                if captions:
                    master_text += captions + " "
                else:
                    print(f"Video {video_id} does not have captions in Spanish.")
            except (KeyError, TypeError, ValueError) as e:
                print(f"An error occurred with video {video_id}: {e}")
    else:
        print("No valid content data available.")
    return master_text

def tokenize_and_split_text(master_text, tokenizer_type, train_percentage):
    """
    Tokenize the master text and split it into training and validation sets.
    """
    tokenizer_instance = tokenizer.TextTokenizer(tokenizer_type=tokenizer_type, text_data=master_text)
    encoded = tokenizer_instance.encode(master_text) # Update function to utilize master_text only when init the class

    stoi = tokenizer_instance.get_stoi()
    print("This is the stoi type", len(stoi))
    itos = tokenizer_instance.get_itos()
    print("This is the itos type", len(itos))
    vocab_size = len(itos)
    print("This is the vocab size", vocab_size)

    assert len(itos) == len(stoi), "Vocab size with tokenizer and stoi does not match."

    encoded = np.array(encoded, dtype=np.uint16)

    n = len(encoded)
    train_ids = encoded[:int(n * train_percentage)]
    val_ids = encoded[int(n * train_percentage):]

    print(master_text[:1000])
    print(tokenizer_instance.decode(encoded[:1000])[:1000])
    assert master_text[:50] == tokenizer_instance.decode(encoded[:50])[:50], "Decoded text does not match."

    return train_ids, val_ids, vocab_size, stoi, itos

def save_data(blob_name, train_ids, val_ids, vocab_size, tokenizer_type, stoi, itos):
    """
    Save the training and validation data to bin files and meta information to a pickle file.
    """
    folder = os.path.join(os.path.dirname(__file__), blob_name)
    if not os.path.exists(folder):
        os.makedirs(folder)

    print(f"Saving data to {folder}")
    train_ids.tofile(os.path.join(folder, 'train.bin'))
    val_ids.tofile(os.path.join(folder, 'val.bin'))

    meta = {
        'tokenizer_type': tokenizer_type,
        'vocab_size': vocab_size,
        'stoi': stoi,
        'itos': itos
    }

    with open(os.path.join(folder, 'meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)
    print(f"Data saved to {folder}")


def process_captions_and_prepare_data(bucket_name, blob_name, development_percentage, train_percentage, tokenizer_type, language):

    content = download_and_load_captions(bucket_name, blob_name)
    development_content = get_development_content(content, development_percentage)
    master_text = create_master_captions(development_content, language)

    print(f"Master caption length: {len(master_text)}")

    train_ids, val_ids, vocab_size, stoi, itos = tokenize_and_split_text(master_text, tokenizer_type, train_percentage)

    print(f"train has {len(train_ids):,} tokens")
    print(f"val has {len(val_ids):,} tokens")

    save_data(blob_name, train_ids, val_ids, vocab_size, tokenizer_type, stoi, itos)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process captions from GCS and prepare training data.")
    parser.add_argument('--bucket_name', type=str, default="metal-sky-419309-videos-v1", help='GCS bucket name')
    parser.add_argument('--blob_name', type=str, default="captions.json", help='GCS blob name')
    parser.add_argument('--development_percentage', type=float, default=0.1, help='Development sample percentage')
    parser.add_argument('--train_percentage', type=float, default=0.9, help='Training data percentage')
    parser.add_argument('--tokenizer_type', type=str, default="gpt-4", help='Type of tokenizer to use')
    parser.add_argument('--language', type=str, default="en", help='Language of the captions')
    args = parser.parse_args()
    process_captions_and_prepare_data(args.bucket_name, args.blob_name, args.development_percentage, args.train_percentage, args.tokenizer_type, args.language)

