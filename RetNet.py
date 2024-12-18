import inspect
import math
import os
import time

import numpy as np
import tiktoken
import torch
import torch._dynamo
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from rotary_embedding_torch import RotaryEmbedding
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

from RMSNorm import RMSNorm


class RetnetConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layers: int = 12
    n_heads: int = 8
    n_embd: int = 1024
    valuebd: int = 2048
    max_size: int = 2048
    dropout: float = 0.1


update_recurrent = {}


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.Linear(config.n_embd, 2 * config.n_embd)
        self.gelu = nn.GELU()
        self.proj = nn.Linear(2 * config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.ln1(x)
        x = self.gelu(x)
        x = self.proj(x)
        x = self.dropout(x)
        return x


class Retention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.block_size = config.block_size
        self.heads = config.n_heads
        self.n_embd = config.n_embd
        self.valuebd = config.valuebd
        self.key_dim = config.n_embd // self.heads
        self.value_dim = config.valuebd // self.heads
        gamma = torch.log(1 - 2 ** (-5 - torch.arange(self.heads, dtype=torch.float)))
        self.register_buffer("gamma", gamma)
        self.to_qk = nn.Linear(config.n_embd, 2 * config.n_embd)
        self.to_v = nn.Linear(config.n_embd, self.valuebd)
        self.scaling = config.n_embd**-0.5
        self.groupnorm = RMSNorm(self.value_dim)
        self.to_g = nn.Linear(self.n_embd, self.valuebd)
        self.to_logits = nn.Linear(self.valuebd, self.n_embd)
        self.rotary_embed = RotaryEmbedding(self.key_dim)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.to_qk.weight, gain=2**-2.5)
        nn.init.xavier_uniform_(self.to_v.weight, gain=2**-2.5)
        nn.init.xavier_uniform_(self.to_g.weight, gain=2**-2.5)
        nn.init.xavier_uniform_(self.to_logits.weight, gain=2**-1)

    def ParrallelRetention(self, q, k, v, mask):
        qk = q @ k.transpose(-1, -2)
        qk_mask = qk * mask
        qk_mask = qk_mask / qk_mask.abs().sum(dim=-1, keepdim=True).clamp(
            min=1, max=5e4
        )
        out = torch.matmul(qk_mask, v)
        out = out.transpose(1, 2)
        return out

    def RecurerentRetention(self, q, k, v, decay, update_recurrent):
        kv = k.transpose(-1, -2) @ v
        if "prev_kv" in update_recurrent:
            prev_kv = update_recurrent["prev_kv"]
            prev_scale = update_recurrent["prev_scale"]
            scale = prev_scale * decay + 1
            kv = prev_kv * (prev_kv.sqrt() * decay / scale.sqrt()).view(
                self.heads, 1, 1
            ) + kv / scale.sqrt().view(self.heads, 1, 1)
        else:
            scale = torch.ones_like(decay)

        update_recurrent["prev_kv"] = kv
        update_recurrent["scale"] = scale
        output = torch.matmul(q, kv)
        output = torch.sum(output, dim=3)
        return output

    def ChunkWiseRetention(self, q, k, v, info_mask):
        B, H, N, D = v.size()
        chunk_size = N // self.block_size
        q = q.view(B, chunk_size, self.block_size, H, self.key_dim).transpose(2, 3)
        k = k.view(B, chunk_size, self.block_size, H, self.key_dim).transpose(2, 3)
        v = v.view(B, chunk_size, self.block_size, H, self.value_dim).transpose(2, 3)
        inner_decay, cross_decay, query_decay, value_decay = info_mask
        qk = q @ k.transpose(-1, -2)
        qk_mask = qk * inner_decay
        inner_scale = qk_mask.detach().abs().sum(dim=-1, keepdim=True).clamp(min=1)
        qk_mask = qk_mask / inner_scale
        qk_mask = self.attn_dropout(qk_mask)
        inner_output = torch.matmul(qk_mask, v)
        kv = k.transpose(-1, -2) @ v
        kv = kv * value_decay

        kv_recurrent = []
        scale_recurrent = []
        scale = torch.ones_like(inner_decay)
        kv_state = torch.zeros_like(kv)

        for i in range(chunk_size):
            kv_recurrent.append(kv_state / scale)
            scale_recurrent.append(scale)
            kv_state = kv_state * cross_decay + kv[:, i]
            scale = (
                kv_state.detach()
                .abs()
                .sum(dim=-2, keepdim=True)
                .max(dim=-1, keepdim=True)
                .values.clamp(min=1)
            )

        kv_recurrent = torch.stack(kv_recurrent, dim=1)
        scale = torch.stack(scale, dim=1)
        all_scale = torch.maximum(inner_scale, scale)
        align_inner_scale = all_scale / inner_scale
        align_cross_scale = all_scale / scale

        cross_output = (q * query_decay) @ kv_recurrent
        output = inner_output / align_inner_scale + kv_recurrent / align_cross_scale
        output = output.transpose(2, 3)
        return output

    def forward(self, x, use_chunkwise=False, update_recurrent=None):
        B, T, C = x.size()
        qk = self.to_qk(x)
        q, k = qk.split(self.n_embd, dim=-1)
        k = k * self.scaling
        v = self.to_v(x)
        q = q.view(B, T, self.heads, self.key_dim).transpose(1, 2)
        k = k.view(B, T, self.heads, self.key_dim).transpose(1, 2)
        v = v.view(B, T, self.heads, self.value_dim).transpose(1, 2)
        q = self.rotary_embed.rotate_queries_or_keys(q)
        k = self.rotary_embed.rotate_queries_or_keys(k)
        if use_chunkwise is False and update_recurrent is None:
            mask = self.get_info_mask(T)[0]
            retention = self.ParrallelRetention(q, k, v, mask=mask)
        elif use_chunkwise != False and update_recurrent is None:
            info_mask = self.get_info_mask(self.block_size)
            retention = self.ChunkWiseRetention(q, k, v, info_mask)
        else:
            retention = self.RecurerentRetention(
                q, k, v, self.gamma, update_recurrent=update_recurrent
            )

        y = self.groupnorm(retention).reshape(B, T, self.value_dim * self.heads)

        non_linear = F.silu(self.to_g(x)) * y
        output = self.resid_dropout(self.to_logits(non_linear))

        return output

    def get_info_mask(self, sequenece_len):
        block_idx = torch.arange(sequenece_len).to(self.gamma)
        mask = torch.tril(torch.ones(sequenece_len, sequenece_len)).to(self.gamma)
        mask = torch.masked_fill(
            block_idx[:, None] - block_idx[None, :], ~mask.bool(), float("inf")
        )
        mask = torch.exp(self.gamma[:, None, None] * mask)
        mask = torch.nan_to_num(mask)
        scale = mask.sum(dim=-1, keepdim=True).sqrt()
        inner_mask = mask / scale

        value_decay = mask[:, -1]
        value_decay = value_decay.unsqueeze(-1)

        cross_decay = torch.exp(self.gamma * sequenece_len)
        cross_decay = cross_decay[:, None, None]

        query_decay = torch.exp(self.gamma[:, None] * (block_idx + 1))
        query_decay = query_decay[:, :, None] / (
            scale / mask[:, -1].sum(dim=-1)[:, None, None]
        )

        info_mask = (inner_mask, cross_decay, query_decay, value_decay)
        return info_mask


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.retention = Retention(config)
        self.ln1 = RMSNorm(config.n_embd)
        self.mlp = FeedForward(config)
        self.ln2 = RMSNorm(config.n_embd)

    def forward(self, x, use_chunkwise=False, update_recurrent=None):
        x = x + self.ln1(
            self.retention(
                x, use_chunkwise=use_chunkwise, update_recurrent=update_recurrent
            )
        )
        x = x + self.ln2(self.mlp(x))

        return x


class RetNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.block_size = config.block_size
        self.RetNet = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                RetNetBlock=nn.ModuleList(
                    [Block(config) for i in range(config.n_layers)]
                ),
                ln1=RMSNorm(config.n_embd),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)

    def forward(self, x, target=None, training_mode=True):
        x = self.RetNet.wte(x)

        B, T, C = x.size()
        use_chunkwise = False
        if T > self.config.block_size:
            use_chunkwise = True
        elif training_mode:
            update_recurrent = None
        else:
            update_recurrent = {}
        for block in self.RetNet.RetNetBlock:
            x = block(x, use_chunkwise, update_recurrent)

        x = self.RetNet.ln1(x)
        logits = self.lm_head(x)
        loss = None
        if target is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), target.view(-1), ignore_index=-1
            )

        # else:
        #     logits = self.lm_head(x[:, [-1], :])
        #     loss = None

        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"number of parameters decay {num_decay_params}")
        print(f"number of nodecay_params {num_nodecay_params}")
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas, **extra_args
        )
        print(f"using AdamW: {use_fused}")
        return optimizer

    @torch.no_grad()
    def generate(self, txt, max_new_tokens, top_k=1, temperature=1):
        for i in range(max_new_tokens):
            ind_tokens = (
                txt if txt.size(1) < self.block_size else txt[:, -self.block_size]
            )
            logits, loss = self(ind_tokens)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, k=top_k)
                logits[logits < v[:, [-1]]] = -float("Inf")

            idx_new = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            txt = torch.cat((txt, idx_new), dim=1)
        return txt
    #topno
    @torch.no_grad()
    def generate_topno(self,txt,max_new_tokens,tokenizer,temperature=1,n=1):
        for i in range(max_new_tokens):
            ind_tokens = (
                    txt if txt.size(1) < self.block_size else txt[:, -self.block_size]
                )
            logits,loss=self(ind_tokens)
            logit=logits[:,-1,:]/temperature
            M_values,_=torch.max(logit,dim=1)
            std=torch.std(logit,dim=1)
            threshold=M_values-n*std
            mask=logit[0,:]<threshold
            indices=torch.nonzero(mask).squeeze()
            logit[0,indices]=-1e9
            probs=F.softmax(logit,dim=-1)
            indx=torch.multinomial(probs,num_samples=1)
            txt=torch.cat((txt,indx),dim=1)
            
        return txt
