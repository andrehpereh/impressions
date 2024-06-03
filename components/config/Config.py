import torch

class ConfigGPT:
    def __init__(self):
        # General settings
        self.out_dir = 'out'
        self.eval_interval = 200
        self.log_interval = 1
        self.eval_iters = 400
        self.eval_only = False
        self.always_save_checkpoint = False
        self.init_from = 'checkpoints'
        self.dataset = 'openwebtext'
        self.block_size = 1024
        self.batch_size = 12
        self.n_layer = 12
        self.n_head = 12
        self.n_embd = 768
        self.dropout = 0.15
        self.bias = False
        self.learning_rate = 6e-4
        self.max_iters = 15000
        self.decay_lr = True
        self.min_lr = 1e-5
        self.grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0s
        self.seed_offset = 0
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
        self.compile = True
        self.model_name = 'GPT-Impressions'
        self.url =  ['https://www.youtube.com/watch?v=_0xBOMWJkgM&list=PL22J3VaeABQBlN8DUor7SKWCwSghcqlY5', 'https://www.youtube.com/watch?v=4tQOlQRp3gQ&list=PL22J3VaeABQCn5nTAx65NRlh1EsKD0UQD']
        # self.id = 'OuF9weSkS68'
        self.id_type = 'playlist'
        self.language = 'en'
        self.blob_name = 'JordanPetersonPersonality' # 'SentdexChannel'
        self.project_id = 'able-analyst-416817'
        self.bucket_name = 'impressions_v1'
        self.development_percentage = 1
        self.train_percentage = .9
        self.tokenizer_type = 'gpt-4'
        self.max_number_videos = 20
        self.sample_input_text = ' your personality '
        self.overwrite_video_captions = False
        self.checkpoint_dir = 'checkpoints'
        self.max_new_tokens = 200
        
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