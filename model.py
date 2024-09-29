import os
import sys
from pathlib import Path
import numpy as np
import random
from os.path import exists
import json
import torch
import torch.nn as nn
from torch.nn.functional import log_softmax, pad
import math
import copy
import time
from torch.optim.lr_scheduler import LambdaLR
import pandas as pd
import altair as alt
from torch.utils.data import DataLoader
import spacy
import GPUtil
import warnings
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from psql.PostgreSQL import PGHypo as PG
import logging


# 随机扰动掩码函数
def random_masking(input_sequence, mask_prob=0.1):
    """
    随机掩码扰动函数，给定一个输入序列，随机将一部分位置的值置为零。

    Parameters:
    - input_sequence (numpy array): 输入序列，可以是一个掩码矩阵
    - mask_prob (float): 控制掩码的概率，默认为0.2

    Returns:
    - masked_sequence (numpy array): 扰动后的序列
    """
    mask = (torch.rand_like(input_sequence, dtype=torch.float32) > mask_prob).float()

    masked_sequence = input_sequence * mask

    return masked_sequence


# clone function
def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


# Transformer class
class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """

    def __init__(self, encoder, decoder, generator):
        super(EncoderDecoder, self).__init__()
        # encoder
        self.encoder1 = encoder
        self.encoder2 = copy.deepcopy(encoder)
        # decoder
        self.decoder = decoder
        # generator
        self.generator = generator

    def forward(self, src, tgt, src_mask1, src_mask2, src_mask3, tgt_mask):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask1, src_mask2), src_mask3, tgt, tgt_mask)

    def encode(self, src, src_mask1, src_mask2):
        embedding = torch.add(self.encoder1(src, src_mask1), self.encoder2(src, src_mask2))
        # gauss noise
        stddev = 0.1
        noise = torch.randn_like(embedding) * stddev
        if self.training:
            embedding = embedding + noise
        return embedding


    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(tgt, memory, src_mask, tgt_mask)


def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    # mask
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    # softmax
    p_attn = scores.softmax(dim=-1)
    # dropout
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


# multi-head attention
class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        # mask
        if mask is not None:
            if len(mask.shape) == 2:
                mask = mask.unsqueeze(0)
            else:
                mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [
            lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(
            query, key, value, mask=mask, dropout=self.dropout
        )

        # 3) "Concat" using a view and apply a final linear.
        x = (
            x.transpose(1, 2)
            .contiguous()
            .view(nbatches, -1, self.h * self.d_k)
        )
        del query
        del key
        del value
        return self.linears[-1](x)


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        # 【1】* len（features）
        self.a_2 = nn.Parameter(torch.ones(features))
        # 【0】* len（features）
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


# Encoder layer
class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        if self.training:
            mask = random_masking(mask)
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


# Encoder class
class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


# Decoder layer
class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


# Decoder类
class Decoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class Generator(nn.Module):
    "Define standard linear + softmax generation step."

    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return log_softmax(self.proj(x), dim=-1)


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(
        torch.uint8
    )
    return subsequent_mask == 0


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(self.w_1(x).relu()))


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)


def make_model(
    tgt_vocab, N=6, d_model=38, d_ff=2048, h=2, dropout=0.15
):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    # multi-head attention
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        Generator(d_model, tgt_vocab),
    )

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


class Batch_train:
    """Object for holding a batch of data with mask during training."""

    def __init__(self, d_model, src, src_mask1, src_mask2, tgt, tgt_y, pad=0):  # 2 = <blank>
        self.src = src
        self.src_mask1 = src_mask1
        self.src_mask2 = src_mask2
        self.src_mask3 = (src != torch.tensor([0] * d_model)).any(axis=2).unsqueeze(1)
        # self.src_mask3 = torch.cat((self.src_mask3, self.src_mask3), dim=2)
        if tgt is not None:
            max_tgt_length = max(len(t) for t in tgt)
            padded_sequences = [seq + [[0] * d_model for _ in range(max_tgt_length - len(seq))] for seq in tgt]
            padded_sequences_y = [seq + [0] * (max_tgt_length - len(seq)) for seq in tgt_y]
            target_tensor = torch.tensor(padded_sequences)
            target_tensor_y = torch.tensor(padded_sequences_y)
            pe = torch.zeros(max_tgt_length, d_model)
            position = torch.arange(0, max_tgt_length).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, d_model // 2 * 2, 2) * -(math.log(10000.0) / d_model)
            )
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)
            target_tensor = target_tensor + pe[:, : target_tensor.size(1)].requires_grad_(False)
            self.tgt = target_tensor[:, :-1]
            self.tgt_y = target_tensor_y[:, 1:]
            tgt_mask = torch.tensor([[[True] * np.count_nonzero(seq) + [False] * (max_tgt_length - 1 - np.count_nonzero(seq))] for seq in self.tgt_y])
            tgt_mask = tgt_mask & subsequent_mask(tgt_mask.size(-1)).type_as(
                tgt_mask.data
            )
            self.tgt_mask = tgt_mask
            self.ntokens = (self.tgt_y != pad).data.sum()


    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(
            tgt_mask.data
        )
        return tgt_mask


class Batch_test:
    """Object for holding a batch of data with mask during training."""

    def __init__(self, d_model, workload, reward, src, src_mask1, src_mask2, vocab, tgt=None, pad=0):  # 2 = <blank>
        self.src = src
        self.src_mask1 = src_mask1
        self.src_mask2 = src_mask2
        self.src_mask3 = (src != torch.tensor([0] * d_model)).any(axis=2).unsqueeze(1)
        self.tgt_o = copy.deepcopy(tgt)
        self.workload = workload
        self.reward = reward
        if tgt is not None:
            tgt_y = copy.deepcopy(tgt)
            for t in tgt:
                new_t = []
                new_ty = []
                ts = t.split(";")
                new_ts = ['<start>']
                for item in ts:
                    new_ts += item.split(',')
                    new_ts.append(';')
                new_ts.append('<end>')
                for item in new_ts:
                    src_new = src[tgt.index(t)]
                    src_copy = []
                    # <pad>
                    src_copy.append([0] * d_model)
                    # <start>
                    src_copy.append([1] * (d_model // 2) + [0] * (d_model // 2))
                    # ;
                    src_copy.append([1] * d_model)
                    # <end>
                    src_copy.append([0] * (d_model // 2) + [1] * (d_model // 2))
                    src_copy += copy.deepcopy(src_new).tolist()
                    new_t.append(src_copy[vocab.index(item)])
                    new_ty.append(vocab.index(item))
                tgt[tgt.index(t)] = new_t
                tgt_y[tgt_y.index(t)] = new_ty
            max_tgt_length = max(len(t) for t in tgt)
            padded_sequences = [seq + [[0] * d_model for _ in range(max_tgt_length - len(seq))] for seq in tgt]
            padded_sequences_y = [seq + [0] * (max_tgt_length - len(seq)) for seq in tgt_y]
            target_tensor = torch.tensor(padded_sequences)
            target_tensor_y = torch.tensor(padded_sequences_y)
            pe = torch.zeros(max_tgt_length, d_model)
            position = torch.arange(0, max_tgt_length).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, d_model // 2 * 2, 2) * -(math.log(10000.0) / d_model)
            )
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)
            target_tensor = target_tensor + pe[:, : target_tensor.size(1)].requires_grad_(False)
            self.tgt = target_tensor[:, :-1]
            self.tgt_y = target_tensor_y[:, 1:]
            tgt_mask = torch.tensor([[[True] * np.count_nonzero(seq) + [False] * (max_tgt_length - 1 - np.count_nonzero(seq))] for seq in self.tgt_y])
            tgt_mask = tgt_mask & subsequent_mask(tgt_mask.size(-1)).type_as(
                tgt_mask.data
            )
            self.tgt_mask = tgt_mask
            self.ntokens = (self.tgt_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(
            tgt_mask.data
        )
        return tgt_mask


class LabelSmoothing(nn.Module):
    "Implement label smoothing."

    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction="sum")
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, true_dist.clone().detach())


def loss(x, crit):
    d = x + 3 * 1
    predict = torch.FloatTensor([[0, x / d, 1 / d, 1 / d, 1 / d]])
    return crit(predict.log(), torch.LongTensor([1])).data


def data_gen_train(data, d_model, batch_size, nbatches, mask1=None, mask2=None):
    "Generate data for a src-tgt copy task."
    data_len = len(data)
    for _ in range(nbatches):
        data_src = []
        data_tgt = []
        data_tgt_y = []
        for _ in range(batch_size):
            d = data[random.randint(0, data_len - 1)]
            data_src.append(list(d['columns'].values()))
            data_tgt.append(d['tgt'])
            data_tgt_y.append(d['tgt_y'])

        # batch_size(32) * len(column)(tpch:61) * len(column_embedding)(10)
        data_src = torch.tensor(data_src)
        # 32
        yield Batch_train(d_model, data_src, mask1, mask2, data_tgt, data_tgt_y, 0)


def data_gen_test(vocab, data, d_model, batch_size, nbatches, mask1=None, mask2=None):
    "Generate data for a src-tgt copy task."
    data_len = len(data)
    for _ in range(nbatches):
        data_src = []
        data_tgt = []
        data_workload = []
        data_reward = []
        for _ in range(batch_size):
            d = data[random.randint(0, data_len - 1)]
            data_src.append(list(d['columns'].values()))
            data_tgt.append(d['index'])
            data_workload.append(d['workload'])
            data_reward.append(d['reward'])

        # batch_size(32) * len(column)(tpch:61) * len(column_embedding)(10)
        src = torch.tensor(data_src)
        # 32
        tgt = data_tgt
        yield Batch_test(d_model, data_workload, data_reward, src, mask1, mask2, vocab, tgt, 0)


class SimpleLossCompute:
    "A simple loss compute and train function."

    def __init__(self, generator, criterion):
        self.generator = generator
        self.criterion = criterion

    def __call__(self, x, y, norm):
        x = self.generator(x)
        sloss = (
            self.criterion(
                x.contiguous().view(-1, x.size(-1)), y.contiguous().view(-1)
            )
            / norm
        )
        return sloss.data * norm, sloss