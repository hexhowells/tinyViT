import torch
import torch.nn as nn
from torch.nn import functional as F

import math

from tinyvit.utils import load_config


class CausalSelfAttention(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        assert config['n_embd'] % config['n_head'] == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config['n_embd'], 3 * config['n_embd'])
        # output projection
        self.c_proj = nn.Linear(config['n_embd'], config['n_embd'])
        # regularization
        self.attn_dropout = nn.Dropout(config['attn_pdrop'])
        self.resid_dropout = nn.Dropout(config['resid_pdrop'])
        
        self.n_head = config['n_head']
        self.n_embd = config['n_embd']


    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # multiply raw x (shape C) by a linear layer to get Q, K, V (shape C each)
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)  # split on the embedding dim (B, T, n_embd)

        # split output embedding layer C into multiple heads
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, n_head, T, head_dim)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, n_head, T, head_dim)

        # causal self-attention; Self-attend: (B, n_head, T, head_dim) x (B, n_head, head_dim, T) -> (B, n_head, T, T)
        y = F.scaled_dot_product_attention(
            q, k, v, 
            attn_mask=None, 
            dropout_p=self.attn_dropout.p if self.training else 0.0, 
            is_causal=True
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        
        return y


class MLP(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.c_fc = nn.Linear(config['n_embd'], 4 * config['n_embd'])
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config['n_embd'], config['n_embd'])
        self.dropout = nn.Dropout(config['resid_pdrop'])

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    """single transformer block"""
    def __init__(self, config: dict):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config['n_embd'])
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config['n_embd'])
        self.mlp = MLP(config)


    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        
        return x


class PatchEmbedding(nn.Module):
    def __init__(self, img_size=256, patch_size=16, n_embed=512):
        assert img_size % patch_size == 0, f'The patch size of {patch_size} is not a factor of the image size {img_size}.'
        self.patch_size = patch_size
        self.num_patches = (img_size//patch_size)**2

        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, n_embed))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, n_embed))
        self.patch_proj = nn.Conv2d(3, n_embed, patch_size, patch_size)


    def forward(self, x):
        x = self.patch_proj(x)
        x = x.flatten(2).transpose(1, 2)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # expand batch dim of class token
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.pos_embed

        return x


class ViT(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        assert config['vocab_size'] is not None
        assert config['context_size'] is not None
        self.context_size = config['context_size']

        model_type = config['model_type']
        
        # overwrite model params if model spec defined
        if model_type in config['models']:
            config['n_layer'] = config['models'][model_type]['n_layer']
            config['n_head'] = config['models'][model_type]['n_head']
            config['n_embd'] = config['models'][model_type]['n_embd']

        self.transformer: nn.ModuleDict = nn.ModuleDict(dict(
            wte = nn.Embedding(config['vocab_size'], config['n_embd']),  # weight token embedding
            wpe = nn.Embedding(config['context_size'], config['n_embd']),  # weight position embedding
            drop = nn.Dropout(config['embd_pdrop']),
            h = nn.ModuleList([Block(config) for _ in range(config['n_layer'])]),  # hidden layers
            ln_f = nn.LayerNorm(config['n_embd']),
        ))
        self.lm_head = nn.Linear(config['n_embd'], config['vocab_size'], bias=False)

        # init all weights, and apply a special scaled init to the residual projections, per GPT-2 paper
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config['n_layer']))

        # report number of parameters (note we don't count the decoder parameters in lm_head)
        n_params = sum(p.numel() for p in self.transformer.parameters())
        print("number of parameters: %.2fM" % (n_params/1e6,))


    def _init_weights(self, module):
        """initialise the GPT model weights"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    

    def forward(self, idx, targets=None) -> tuple[torch.Tensor, torch.Tensor|None]:
        device = idx.device
        _, t = idx.size()
        assert t <= self.context_size, f"Cannot forward sequence of length {t}, block size is only {self.context_size}"
        
        # pos is an array of integers as torch.nn.Embedding performs a direct array lookup 
        # instead of computing the entire linear forward pass
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)  # pos = [[0, 1, 2, 3, 4, ..., t]]

        tok_emb = self.transformer.wte(idx)  # type: ignore
        pos_emb = self.transformer.wpe(pos)  # type: ignore
        x = self.transformer.drop(tok_emb + pos_emb)  # type: ignore
        
        for block in self.transformer.h:  # type: ignore
            x = block(x)
        
        x = self.transformer.ln_f(x)  # type: ignore
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        return logits, loss


    def configure_optimizers(self, train_config: dict) -> torch.optim.Optimizer:
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        
        for module_name, m in self.named_modules():
            for param_name, _ in m.named_parameters():
                full_param_name = '%s.%s' % (module_name, param_name) if module_name else param_name # full param name
                # random note: because named_modules and named_parameters are recursive
                # we will see the same tensors p many many times. but doing it this way
                # allows us to know which parent module any tensor p belongs to...
                if param_name.endswith('bias'):
                    no_decay.add(full_param_name)
                elif param_name.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    decay.add(full_param_name)
                elif param_name.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    no_decay.add(full_param_name)

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}

        decay.discard('lm_head.weight')
        no_decay.discard('lm_head.weight')
        
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config['weight_decay']},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config['learning_rate'], betas=train_config['betas'])
        
        return optimizer
