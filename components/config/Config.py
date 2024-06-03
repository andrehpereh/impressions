import torch

class ConfigGPT:
    def __init__(self):
        # General settings
        self.out_dir = 'out'
        self.eval_interval = 200
        self.log_interval = 1
        self.eval_iters = 200
        self.eval_only = False
        self.always_save_checkpoint = True
        self.init_from = 'scratch'
        self.dataset = 'openwebtext'
        self.block_size = 64
        self.batch_size = 16
        self.n_layer = 6
        self.n_head = 6
        self.n_embd = 36
        self.dropout = 0.0
        self.bias = False
        self.learning_rate = 6e-4
        self.max_iters = 300
        self.decay_lr = True
        self.min_lr = 6e-5
        self.grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0s
        self.seed_offset = 0
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
        self.compile = True
        self.model_name = 'GPT-Impressions'
        self.url =  'https://www.youtube.com/watch?v=OuF9weSkS68&list=PL0iVR8sl9TiWqj_JmVjEgAzl4YhtTt9Wf' #'https://www.youtube.com/watch?v=pCX_3p40Efc'
        self.id = 'OuF9weSkS68'
        self.id_type = 'playlist'
        self.language = 'en'
        self.blob_name = 'ColdFusion' # 'SentdexChannel'
        self.project_id = 'metal-sky-419309'
        self.bucket_name = 'metal-sky-419309-videos-v1'
        self.development_percentage = 1
        self.train_percentage = 1
        self.tokenizer_type = 'gpt-4'
        self.max_number_videos = 20
        self.sample_input_text = ' welcome to the world of AI'
        self.overwrite_video_captions = False,
        self.checkpoint_dir = 'checkpoints'

    def __str__(self):
        config_str = "Configuration:\n"
        for attr, value in self.__dict__.items():
            config_str += f"{attr}: {value}\n"
        return config_str

def override_config_with_args(config, args):
    for key, value in vars(args).items():
        if value is not None:
            setattr(config, key, value)
    return config