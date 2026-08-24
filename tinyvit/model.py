import torch
import torch.nn as nn
from torch.nn import functional as F

import math

from tinyvit.utils import load_config


class SelfAttention(nn.Module):
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

        # self-attention; Self-attend: (B, n_head, T, head_dim) x (B, n_head, head_dim, T) -> (B, n_head, T, T)
        y = F.scaled_dot_product_attention(
            q, k, v, 
            attn_mask=None, 
            dropout_p=self.attn_dropout.p if self.training else 0.0, 
            is_causal=False
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
        self.attn = SelfAttention(config)
        self.ln_2 = nn.LayerNorm(config['n_embd'])
        self.mlp = MLP(config)


    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        
        return x


class PatchEmbedding(nn.Module):
    def __init__(self, img_size=256, patch_size=16, n_embed=512):
        super().__init__()
        assert img_size % patch_size == 0, f'The patch size of {patch_size} is not a factor of the image size {img_size}.'
        self.patch_size = patch_size
        self.num_patches = (img_size//patch_size)**2

        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches+1, n_embed))
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
        model_type = config['model_type']
        
        # overwrite model params if model spec defined
        if model_type in config['models']:
            config['n_layer'] = config['models'][model_type]['n_layer']
            config['n_head'] = config['models'][model_type]['n_head']
            config['n_embd'] = config['models'][model_type]['n_embd']

        self.patch_embed = PatchEmbedding(config['img_size'], config['patch_size'], config['n_embd'])
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config['n_layer'])])
        self.mlp_head = nn.Linear(config['n_embd'], config['num_classes'])

        # init all weights, and apply a special scaled init to the residual projections, per GPT-2 paper
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config['n_layer']))

        # report number of parameters
        n_params = sum(p.numel() for p in self.parameters())
        head_params = sum(p.numel() for p in self.mlp_head.parameters())
        print("number of parameters: %.2fM" % ((n_params - head_params)/1e6,))


    def _init_weights(self, module):
        """initialise the model weights"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    

    def forward(self, x) -> torch.Tensor:
        x = self.patch_embed(x)
        x = self.blocks(x)
        x = x[:, 0, :]  # extract class token
        x = self.mlp_head(x)

        return x


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
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv2d)
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

        no_decay.add('patch_embed.pos_embed')
        no_decay.add('patch_embed.cls_token')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        
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
