import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import time


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.1, device=None):
        """

        :param d_model: dimensionality of model's input --> here d_model = 1 for bitflip noise
        :param n_head: number of attention heads to split the input into
        :param dropout:
        """
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_head == 0, "d_model must be divisible by n_head"

        self.d_model = d_model
        self.n_head = n_head
        self.d_k = d_model // n_head

        self.W_q = nn.Linear(d_model, d_model, device=device)  # could map to a higher dimension here
        self.W_k = nn.Linear(d_model, d_model, device=device)
        self.W_v = nn.Linear(d_model, d_model, device=device)
        self.W_o = nn.Linear(d_model, d_model, device=device)
        self.dropout = nn.Dropout(dropout)

    def scaled_dot_product_attention(self, q, k, v, mask=None):
        start_time = time.time()
        attn_score = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        # useful for preventing attention to certain parts like padding
        if mask is not None:
            attn_score = attn_score.masked_fill(mask == 0,
                                                -1e9)  # from docs: Fills elements of self tensor with value where mask is True. The shape of mask must be broadcastable with the shape of the underlying tensor.

        # print("--- %s seconds first product---" % (time.time() - start_time))
        # start_time = time.time()
        # softmax to obtain attention probabilities
        attn_prob = torch.softmax(attn_score, dim=-1)
        # print("--- %s seconds softmax---" % (time.time() - start_time))
        # start_time = time.time()
        output = torch.matmul(attn_prob, v)
        # print("--- %s seconds second product---" % (time.time() - start_time))
        return output

    def split_heads(self, x):
        batch_size, seq_length, d_model = x.size()  # should be B, D^2, 1 for bitflip noise
        return x.view(batch_size, seq_length, self.n_head, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)

    def forward(self, q, k, v, mask=None):
        start_time = time.time()
        q = self.split_heads(self.W_q(q))
        k = self.split_heads(self.W_k(k))
        v = self.split_heads(self.W_v(v))
        # print("--- %s seconds split heads---" % (time.time() - start_time))
        start_time = time.time()
        attn_output = self.scaled_dot_product_attention(q, k, v, mask)

        output = self.W_o(self.combine_heads(attn_output))
        # print("--- %s seconds calc output---" % (time.time() - start_time))
        return self.dropout(output)


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1, device=None):
        super(PositionWiseFeedForward, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(d_model, d_ff, device=device)
        self.fc2 = nn.Linear(d_ff, d_model, device=device)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.dropout2(self.fc2(self.dropout1(self.relu(self.fc1(x)))))
        # return self.relu(self.fc2(self.relu(self.fc1(x))))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)].to(x.device)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_head, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, n_head, dropout)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + attn_output)
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_head, d_ff, dropout=0.1, device=None):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, n_head, device=device)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, device=device)
        self.norm1 = nn.LayerNorm(d_model, device=device)
        self.norm2 = nn.LayerNorm(d_model, device=device)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # start_time = time.time()
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + attn_output)
        # print("--- %s seconds attention---" % (time.time() - start_time))
        start_time = time.time()
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)
        # print("--- %s seconds ff---" % (time.time() - start_time))
        return x
