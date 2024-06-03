import os
import pickle
from contextlib import nullcontext
import numpy as np
import torch
from components.utils.gcs_utils import upload_to_gcs
from components.data import tokenizer
from torch.optim.lr_scheduler import _LRScheduler
from components.training.model import GPTConfig, GPT
from components.config.Config import ConfigGPT

def setup_device_and_context(config):
    # torch.manual_seed(1337 + config.seed_offset)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device_type = 'cuda' if 'cuda' in config.device else 'cpu'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[config.dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    return ptdtype, ctx

def get_batch(split, data_dir, block_size, batch_size, device, device_type):
    data = np.memmap(os.path.join(data_dir, f'{split}.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

def estimate_loss(model, eval_iters, get_batch, ctx):
    model.eval()
    out = {}
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def save_checkpoint(model, optimizer, iter_num, model_args, best_val_loss, bucket_name, destination_blob_name, checkpoint_dir="checkpoints"):
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, 'ckpt.pt')
    torch.save({
        'iter_num': iter_num,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'model_args': model_args,
        'best_val_loss': best_val_loss,
    }, checkpoint_path)
    upload_to_gcs(bucket_name=bucket_name, source_file_name=checkpoint_path, destination_blob_name=os.path.join(checkpoint_dir, destination_blob_name))
    print(f"Checkpoint saved locally at {checkpoint_path}")
    print(f"Checkpoint uploaded to GCS at {os.path.join(bucket_name, destination_blob_name)}")



def resume_training(config, device):
    print(f"Resuming training from {config.checkpoint_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(config.checkpoint_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    model_args = config_to_model_args(checkpoint_model_args)
    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model_state_dict']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    state_dict = fix_state_dict_keys(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
    return model, state_dict, iter_num, best_val_loss


def config_to_model_args(config):
    model_args = {}
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = config[k]
    return model_args


def fix_state_dict_keys(state_dict):
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    return state_dict


def load_checkpoint(checkpoint_path, model, optimizer):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    iter_num = checkpoint['iter_num']
    print(f"Checkpoint loaded from {checkpoint_path} at iteration {iter_num}")
    return iter_num

class StepLRWithMinLr(_LRScheduler):
    def __init__(self, optimizer, step_size, gamma=0.1, min_lr=1e-6, last_epoch=-1):
        self.step_size = step_size
        self.gamma = gamma
        self.min_lr = min_lr
        super(StepLRWithMinLr, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if (self.last_epoch == 0) or (self.last_epoch % self.step_size != 0):
            return [max(group['lr'], self.min_lr) for group in self.optimizer.param_groups]
        return [max(group['lr'] * self.gamma, self.min_lr) for group in self.optimizer.param_groups]

    def get_last_lr(self):
        return [max(group['lr'], self.min_lr) for group in self.optimizer.param_groups]

class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = config.device
        self.init_from = config.init_from
        self.device_type = config.device
        self.data_dir = os.path.join('components', 'data', config.blob_name)
        self.ptdtype, self.ctx = setup_device_and_context(config)

        os.makedirs(config.out_dir, exist_ok=True)

        meta_path = os.path.join(self.data_dir, 'meta.pkl')
        print(f"loading meta data from {meta_path}")
        self.meta_vocab_size = None
        if os.path.exists(meta_path):
            with open(meta_path, 'rb') as f:
                meta = pickle.load(f)
            self.meta = meta
            self.meta_vocab_size = meta['vocab_size']
            self.stoi = meta['stoi']
            self.itos = meta['itos']
            print(f"found vocab_size = {self.meta_vocab_size} (inside {meta_path})")

        self.model_args = dict(
            n_layer=config.n_layer,
            n_head=config.n_head,
            n_embd=config.n_embd,
            block_size=config.block_size,
            bias=config.bias,
            vocab_size=self.meta_vocab_size if self.meta_vocab_size is not None else 50304,
            dropout=config.dropout
        )
        print(f"model args: {self.model_args}")
    
        self.iter_num = 0
        self.best_val_loss = 1e9

        if self.init_from == 'scratch':
            gptconf = GPTConfig(**self.model_args)
            self.model = GPT(gptconf)
            # self.model.to(self.device)
        elif self.init_from == 'checkpoints':
            self.model, state_dict, self.iter_num, self.best_val_loss = resume_training(config, self.device)
            self.model.load_state_dict(state_dict)
        else:
            raise ValueError(f"unknown init_from {self.init_from}")
        
        
        self.config.max_iters = self.config.max_iters + self.iter_num if self.iter_num > 0 else self.config.max_iters
        self.model.to(self.device)

        self.scaler = torch.cuda.amp.GradScaler(enabled=(config.dtype == 'float16'))

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.learning_rate)
        self.scheduler = StepLRWithMinLr(self.optimizer, step_size=config.max_iters // 10, gamma=0.8, min_lr=config.min_lr)

        if config.compile:
            print("compiling the model... (takes a ~minute)")
            self.model = torch.compile(self.model) # requires PyTorch 2.0

    def get_batch(self, split):
        return get_batch(split, self.data_dir, self.config.block_size, self.config.batch_size, self.device, self.device_type)

    def train(self):
        for iter_num in range(self.config.max_iters):
            if iter_num % self.config.eval_interval == 0:
                losses = estimate_loss(self.model, self.config.eval_iters, self.get_batch, self.ctx)
                print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
                if losses['val'] < self.best_val_loss or self.config.always_save_checkpoint and iter_num > 0:
                    save_checkpoint(
                        self.model, self.optimizer, iter_num, self.model_args, self.best_val_loss,
                        self.config.bucket_name, self.config.blob_name
                    )
            if iter_num == 0 and self.config.eval_only:
                break

            xb, yb = self.get_batch('train')
            with self.ctx:
                logits, loss = self.model(xb, yb)
            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            if self.config.grad_clip != 0.0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            if self.config.decay_lr:
                self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"Step: {iter_num}, Loss: {loss.item():.4f}, Learning Rate: {current_lr:.6f}")
        return self.model

if __name__ == "__main__":
    config = ConfigGPT()
    print(config)
    print("Training...")
    trainer = Trainer(config)
    model = trainer.train()
    print("Generating sample...")
    tokenizer_instance = tokenizer.TextTokenizer(tokenizer_type=trainer.meta['tokenizer_type'], stoi=trainer.stoi, itos=trainer.itos)
    init_token = tokenizer_instance.encode("create ")
    context = torch.tensor(init_token, dtype=torch.long, device=config.device)
    context = context.unsqueeze(0)
    generated_encoded_text = model.generate(context, max_new_tokens=50)[0].tolist()
    print(tokenizer_instance.decode(generated_encoded_text))
    print("Done!")




