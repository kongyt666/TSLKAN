import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft

from layers.Embed import PositionalEmbedding, TemporalEmbedding, TimeFeatureEmbedding

from layers.KAN.fastKAN import FastKAN

import numpy as np


class temporalBlock(nn.Module):
    def __init__(self, configs, bias=True):
        super(temporalBlock, self).__init__()

        # self.position_embedding = PositionalEmbedding(configs.d_model)

        self.temporal1 = nn.Sequential(
            FastKAN([configs.seq_len, configs.d_model]),
            nn.Dropout(configs.dropout),
            FastKAN([configs.d_model, configs.d_model]),
            nn.Dropout(configs.dropout),
        )

        self.temporal2 = nn.Sequential(
            FastKAN([configs.d_model, configs.d_model]),
            nn.Dropout(configs.dropout),
            FastKAN([configs.d_model, configs.seq_len]),
            nn.Dropout(configs.dropout)
        )

    def forward(self, x, x_mark_enc):
        # x: [B, T, C]

        r = x
        # x = x.transpose(1, 2)
        x = self.temporal1(x.transpose(1, 2))
        # x + self.position_embedding(x)
        # pos_enc = x + self.position_embedding(x)  # (1, seq_len, d_model)
        # enhanced_input = torch.cat([x, pos_enc.expand(x.size(0), -1, -1)], dim=-1)
        x = r + self.temporal2(x).transpose(1, 2)

        return x


class channelBlock(nn.Module):
    def __init__(self, configs, bias=True):
        super(channelBlock, self).__init__()
        self.position_embedding = PositionalEmbedding(configs.d_ff)

        self.channel1 = nn.Sequential(
            FastKAN([configs.enc_in, configs.d_ff]),
            # nn.Dropout(configs.dropout),
            FastKAN([configs.d_ff, configs.d_ff]),
            # nn.Dropout(configs.dropout),

        )
        self.channel2 = nn.Sequential(
            FastKAN([configs.d_ff * 2, configs.d_ff * 2]),
            nn.Dropout(configs.dropout),
            FastKAN([configs.d_ff * 2, configs.d_ff]),
            nn.Dropout(configs.dropout),
            FastKAN([configs.d_ff, configs.enc_in]),
            nn.Dropout(configs.dropout)
        )

    def forward(self, x, x_mark_enc):
        # x: [B, T, C]
        r = x
        x = self.channel1(x)

        pos_enc = self.position_embedding(x)
        enhanced_input = torch.cat([x, pos_enc.expand(x.size(0), -1, -1)], dim=-1)
        x = r + self.channel2(enhanced_input)

        return x


class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.use_norm = configs.use_norm

        self.mixer = nn.ModuleList([temporalBlock(configs)
                                    for _ in range(configs.e_layers)])
        self.channel = nn.ModuleList([channelBlock(configs)
                                      for _ in range(configs.d_layers)])

        self.projection = nn.Sequential(
            FastKAN([configs.seq_len, configs.pred_len]),
            nn.Dropout(configs.dropout)
        )

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):

        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev
        B, T, C = x_enc.shape
 
        # x: [B, T, C]
        for i in range(self.configs.e_layers):
            x_enc = self.mixer[i](x_enc, x_mark_enc)
        for i in range(self.configs.d_layers):
            x_enc = self.channel[i](x_enc, x_mark_enc)

        dec_out = self.projection(x_enc.transpose(1, 2)).transpose(1, 2)

        if self.use_norm:
            # De-NormaliSzation from Non-stationary Transformer
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        return None













































