# Impressions - A GPT Language Model for YouTube Content

Impressions is a custom GPT language model trained on YouTube video captions. It allows you to train a GPT model on specific YouTube channels or playlists, generating text that aligns with the style and topics found in the source material.

## Key Features
- **YouTube Data Integration**: Extracts video captions directly from YouTube using the YouTube Transcript API.
- **Customizable Training**: Train the GPT model on specific channels, playlists, or even individual videos.
- **Flexible Configuration**: Easily adjust model parameters, training settings, and data sources through a configuration file and command-line arguments.
- **GCS Storage**: Utilizes Google Cloud Storage (GCS) to store video captions and model checkpoints.
- **Tokenization Options**: Choose from different tokenizer types (e.g., GPT-4) for text preprocessing.
- **Sampling and Text Generation**: Generate samples of text based on the trained model's understanding of the input data.

## Project Structure

.
├── checkpoints/n\
│ └── ckpt.ptn\
├── components/n\
│ ├── config/n\
│ │ ├── init.pyn\
│ │ └── Config.pyn\
│ ├── data/n\
│ │ ├── init.pyn\
│ │ ├── Dockerfilen\
│ │ ├── meta.pkln\
│ │ ├── prepare_text_data.pyn\
│ │ ├── tokenizer.pyn\
│ │ ├── train.binn\
│ │ ├── val.binn\
│ │ └── youtube_captions.pyn\
│ ├── training/n\
│ │ ├── init.pyn\
│ │ ├── Dockerfilen\
│ │ ├── model.pyn\
│ │ ├── sample.pyn\
│ │ └── train.pyn\
│ └── utils/n\
│ ├── init.pyn\
│ └── gcs_utils.pyn\
├── pipeline/n\
│ ├── Dockerfilen\
│ └── pipeline.pyn\
└── Dockerfilen\



## Project Pipeline
### Configuration and Argument Parsing (`components/config/Config.py`)
- `pipeline.py` initiates the project, loads default configurations, and parses command-line arguments.
- Command-line arguments override default settings in the configuration.
- Key functions: `parse_args()`, `override_config_with_args()`.

### Data Preparation (`components/data`)
- Downloads video captions from YouTube if they don't exist in the GCS bucket.
- Saves downloaded data to GCS.
- Preprocesses captions (tokenization, etc.) for training.

### Model Definition (`components/training/model.py`, `components/config/Config.py`)
#### Files Involved
- `components.training.model`
- `components.config.Config`

#### Description
- Defines the GPT model architecture.
- Includes classes for LayerNorm, CausalSelfAttention, MLP, Block, and GPT itself.
- Uses `GPTConfig` to set model parameters.

#### Key Classes
- **LayerNorm**: Custom LayerNorm with optional bias.
- **CausalSelfAttention**: Defines the self-attention mechanism.
- **MLP**: Multilayer Perceptron.
- **Block**: A transformer block consisting of LayerNorm, Self-Attention, and MLP.
- **GPT**: Main GPT model.

### Training Process (`components/training/train.py`)
- `Trainer` class orchestrates the training loop.
- Manages data batching, loss calculation, checkpointing, and model saving/loading.
- Key functions: `setup_device_and_context()`, `get_batch()`, `estimate_loss()`, `save_checkpoint()`, `load_checkpoint()`.

### Docker Integration **Note**: The project does not run containerized yet but is intended to in the future.
- Each core component (data, training, pipeline) has its own Dockerfile, facilitating containerized development and deployment.

## Getting Started
### Prerequisites
1. **Google Cloud Project**: Create a project and enable the YouTube Data API. Obtain a developer key and set it as the `DEVELOPER_KEY` environment variable.
2. **Google Cloud Storage Bucket**: Create a GCS bucket to store video captions and model checkpoints. Update the `bucket_name` configuration variable accordingly.
3. **Python Dependencies**: Install the required packages using `pip install -r requirements.txt`.
4. **Docker**: Make sure you have Docker installed to build and run the containers.

### Configuration
- `components/config/Config.py`: Modify settings like learning rate, batch size, etc.
- Command-line Arguments: Override configuration options with command-line arguments (run `python pipeline/pipeline.py --help` for details).

### Building Docker Images (Pending)
Navigate to each component directory (`components/data`, `components/training`, `pipeline`) and run `docker build -t <image-name> .` to build the Docker images.

### Running with Docker (Pending)
Use `docker run <image-name>` to run the specific component in its container.

### Sample Usage (without Docker)
```bash
python pipeline/pipeline.py \
  --out_dir ./output \
  --eval_interval 1000 \
  --log_interval 100 \
  --eval_iters 200 \
  --batch_size 8 \
  --learning_rate 0.0003 \
  --max_iters 10000 \
  --dataset youtube_captions \
  --url "https://youtube.com/some_video" \
  --bucket_name "your_gcs_bucket" \
  --blob_name "video_captions_blob"
```


Using the YouTube Data API to Extract Captions
To extract video captions, you will need to use the YouTube Data API. Here is a brief guide:

Enable YouTube Data API:

Go to the Google Developers Console.
Create a new project or select an existing project.
Enable the YouTube Data API v3.
Obtain API Key:

In the API & Services section, create credentials for an API key.
Set the DEVELOPER_KEY environment variable with this key.
Download Captions:

Use the youtube_captions.py script to download captions. This script uses the youtube_transcript_api library to fetch captions.
```
from youtube_transcript_api import YouTubeTranscriptApi

video_id = "your_video_id"
transcript = YouTubeTranscriptApi.get_transcript(video_id)
print(transcript)
```
Refer to the YouTube Data API documentation for more details on how to use the API.


## Acknowledgments

I would like to extend my sincere thanks to [Andrej Karpathy](https://karpathy.ai) for his invaluable contributions and insights,
particularly in the field of deep learning and neural networks. His [blog posts](https://karpathy.github.io/) and open-source projects [Github] https://github.com/karpathy,
such as [nanoGPT](https://github.com/karpathy/nanoGPT), have been fundamental in shaping the development of this project.
The core concepts and structures implemented here heavily relied on by his work.
