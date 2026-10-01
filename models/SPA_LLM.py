import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from transformers import GPT2Config, GPT2Model, GPT2Tokenizer


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class LowRankAnchorProjection(nn.Module):
    """
    Low-rank vocabulary distillation: projects full vocabulary embeddings
    [D, V] -> [D, V'] through a rank-m bottleneck, where V' << V.
    """
    def __init__(self, vocab_size, pool_size, d_model, rank=128):
        super().__init__()
        self.rank = rank
        self.pool_size = pool_size

        self.proj_down = nn.Linear(vocab_size, rank, bias=False)
        self.proj_up = nn.Linear(rank, pool_size, bias=False)
        self.norm = nn.LayerNorm(rank)

        nn.init.xavier_uniform_(self.proj_down.weight)
        nn.init.xavier_uniform_(self.proj_up.weight)

        total_params = vocab_size * rank + rank * pool_size
        print(f"LowRankAnchorProjection: rank={rank}, params={total_params:,}")
        print(f"Compression ratio: {vocab_size * pool_size / total_params:.1f}x")

    def forward(self, E):
        hidden = self.proj_down(E)
        hidden = self.norm(hidden)
        output = self.proj_up(hidden)
        return output


class SPA_LLM(nn.Module):
    """
    SPA-LLM: Phase-Aware Semantic Alignment for LLM-Based
    Long-Term Time Series Forecasting.

    Key components:
    - Semantic-Enhanced Structured Tokenization: phase patching + adaptive
      pooling compresses P seasonal phases into P' compact tokens.
    - Natural Language Semantic Mixing: low-rank vocabulary distillation
      produces V' distilled anchors; learnable phase queries attend over
      anchors to inject token-level semantic priors.
    """
    def __init__(self, configs):
        super(SPA_LLM, self).__init__()

        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.channels = configs.enc_in
        self.d_model = configs.d_model

        self.period_len = configs.period_len
        self.seg_num_x = (self.seq_len + self.period_len - 1) // self.period_len
        self.seg_num_y = (self.pred_len + self.period_len - 1) // self.period_len

        self.decomp_kernel = configs.decomp_kernel
        self.decomp = series_decomp(self.decomp_kernel)

        gpt2_root = getattr(configs, "gpt2_root", "gpt2")
        config = GPT2Config.from_pretrained(gpt2_root)
        self.gpt2 = GPT2Model.from_pretrained(gpt2_root, config=config)

        for name, param in self.gpt2.named_parameters():
            param.requires_grad = "ln" in name

        self.gpt2.h = self.gpt2.h[:configs.llm_layers]

        self.tokenizer = GPT2Tokenizer.from_pretrained(gpt2_root, trust_remote_code=True)
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token or '[PAD]'

        mlp_hidden = getattr(configs, "mlp_hidden", None)
        if mlp_hidden is None:
            self.season_embed = nn.Linear(self.seg_num_x, self.d_model)
        else:
            self.season_embed = nn.Sequential(
                nn.Linear(self.seg_num_x, mlp_hidden),
                nn.ReLU(inplace=True),
                nn.Linear(mlp_hidden, self.d_model),
            )

        self.phase_tokens = configs.pp
        self.phase_pool = nn.AdaptiveAvgPool1d(self.phase_tokens)

        self.season_out_proj = nn.Linear(self.d_model, self.seg_num_y)
        self.out_bias_proj = nn.Linear(self.d_model, self.seg_num_y)

        self.trend_linear = nn.Linear(self.seq_len, self.pred_len)

        self.pool_size = configs.pool_size
        self.rank = configs.rank

        vocab_size = self.gpt2.config.vocab_size
        self.anchor_projection = LowRankAnchorProjection(
            vocab_size=vocab_size,
            pool_size=self.pool_size,
            d_model=self.d_model,
            rank=self.rank
        )

        self.phase_queries = nn.Parameter(torch.empty(self.phase_tokens, self.d_model))
        nn.init.normal_(self.phase_queries, mean=0.0, std=0.02)

        self.a1 = configs.a1
        self.a2 = configs.a2

    def _build_anchors(self):
        E = self.gpt2.wte.weight
        E_T = E.transpose(0, 1)
        E_prime_T = self.anchor_projection(E_T)
        E_prime = E_prime_T.transpose(0, 1)
        return E_prime

    def _build_phase_memory(self):
        anchors = self._build_anchors()

        q = F.normalize(self.phase_queries, dim=-1)
        a = F.normalize(anchors, dim=-1)

        attn = torch.matmul(q, a.transpose(0, 1)) / 0.01
        attn = F.softmax(attn, dim=-1)

        phase_memory = torch.matmul(attn, anchors)

        return phase_memory

    def _normalize(self, x):
        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x = x / stdev
        return x, means, stdev

    def _renormalize(self, x, means, stdev):
        return x * stdev + means

    def _phase_patch(self, x):
        B, L, C = x.shape
        P = self.period_len
        x = x.permute(0, 2, 1)

        remainder = L % P
        if remainder > 0:
            pad_len = P - remainder
            pad_data = x[:, :, :pad_len]
            x = torch.cat([pad_data, x], dim=-1)

        N = x.shape[-1] // P
        x = x.reshape(B, C, N, P)
        x = x.permute(0, 1, 3, 2)
        x = rearrange(x, 'b c p n -> (b c) p n')
        return x

    def forward(self, x, x_mark=None):
        B, L, C = x.shape

        x, means, stdev = self._normalize(x)

        seasonal, trend = self.decomp(x)

        season_patches = self._phase_patch(seasonal)

        season_patches = self.phase_pool(
            season_patches.transpose(1, 2)
        ).transpose(1, 2)

        season_tokens = self.season_embed(season_patches)

        phase_memory = self._build_phase_memory()
        season_tokens = season_tokens + self.a1 * phase_memory.unsqueeze(0)

        out_season = self.gpt2(inputs_embeds=season_tokens).last_hidden_state

        season_proj = self.season_out_proj(out_season)

        phase_bias = self.out_bias_proj(phase_memory).unsqueeze(0)
        season_proj = season_proj + self.a2 * phase_bias

        season_proj = F.interpolate(
            season_proj.transpose(1, 2),
            size=self.period_len,
            mode='linear',
            align_corners=False
        ).transpose(1, 2)

        season_proj = season_proj.view(B, C, self.period_len, self.seg_num_y)
        season_pred = season_proj.permute(0, 1, 3, 2).reshape(B, C, -1)
        season_pred = season_pred[:, :, :self.pred_len].permute(0, 2, 1)

        trend_bcL = trend.permute(0, 2, 1)
        trend_bcP = self.trend_linear(trend_bcL)
        trend_pred = trend_bcP.permute(0, 2, 1)

        pred = season_pred + trend_pred
        pred = self._renormalize(pred, means, stdev)
        return pred
