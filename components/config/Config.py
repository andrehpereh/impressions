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
        self.block_size = 128
        self.batch_size = 32
        self.n_layer = 4
        self.n_head = 8
        self.n_embd = 72
        self.dropout = 0.0
        self.bias = False
        self.learning_rate = 6e-4
        self.max_iters = 2000
        self.decay_lr = True
        self.min_lr = 6e-5
        self.grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
        self.seed_offset = 0
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
        self.compile = True
        self.model_name = 'Impressions'

    def __str__(self):
        config_str = "Configuration:\n"
        print("Simon")
        for attr, value in self.__dict__.items():
            config_str += f"{attr}: {value}\n"
        return config_str

def override_config_with_args(config, args):
    for key, value in vars(args).items():
        if value is not None:
            setattr(config, key, value)
    return config