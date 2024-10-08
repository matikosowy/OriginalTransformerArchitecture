import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, n_heads):
        super().__init__()
        self.embed_size = embed_size
        self.n_heads = n_heads
        self.head_dim = embed_size // n_heads
        
        assert self.head_dim * n_heads == embed_size, "Embedding size indivisible by number of heads!"
        
        self.query_layer = nn.Linear(embed_size, embed_size)
        self.key_layer = nn.Linear(embed_size, embed_size)
        self.value_layer = nn.Linear(embed_size, embed_size)
        self.output_layer = nn.Linear(embed_size, embed_size)
        
    def forward(self, query, keys, values, mask=None):
        bs = query.shape[0]
        Q_len = query.shape[1]
        K_len= keys.shape[1]
        V_len = values.shape[1]
        
        # bs x Q/K/V_len x embed_size:
        Q = self.query_layer(query)
        K = self.key_layer(keys)
        V = self.value_layer(values)
        
        # bs x n_heads x Q/K/V_len x head_dim:
        Q = Q.reshape(bs, Q_len, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.reshape(bs, K_len, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.reshape(bs, V_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        # (bs * n_heads) x Q/K_len x head_dim:
        Q = Q.reshape(bs * self.n_heads, Q_len, self.head_dim)
        K = K.reshape(bs * self.n_heads, K_len, self.head_dim)
        
        K_transposed = K.transpose(1, 2) # <-- (bs * n_heads) x head_dim x K_len
        compatibility = torch.bmm(Q, K_transposed) # <-- (bs * n_heads) x Q_len x K_len
        
        if mask is not None:
            compatibility = compatibility.masked_fill(mask == 0, float('-inf'))
        
        # scale factor = sqrt(head_dim), because head_dim is equal to the dimension of the key dk (as in the paper)
        compatibility_scaled = compatibility / (self.head_dim ** 0.5)
        attention = F.softmax(compatibility_scaled, dim=-1)
    
        V = V.reshape(bs * self.n_heads, V_len, self.head_dim) # <-- (bs * n_heads) x V_len x head_dim
        output = torch.bmm(attention, V) # <-- (bs * n_heads) x Q_len x head_dim
        
        output = output.reshape(bs, self.n_heads, Q_len, self.head_dim).transpose(1, 2) # <-- bs x Q_len x n_heads x head_dim
        output = output.reshape(bs, Q_len, self.embed_size) # <-- bs x Q_len x embed_size
        output = self.output_layer(output)
        
        return output
        
        
class TransformerBlock(nn.Module):
    def __init__(self, embed_size, n_heads, drop_rate, ff_expansion):
        super().__init__()
        
        self.attention = MultiHeadAttention()
        self.l_norm1 = nn.LayerNorm(embed_size)
        self.l_norm2 = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(drop_rate)
        
        self.ff_net = nn.Sequential(
            nn.Linear(embed_size, embed_size * ff_expansion),
            nn.ReLU(),
            nn.Linear(embed_size * ff_expansion, embed_size)
        )
        
    def forward():
        pass
        
        
# test

attention = MultiHeadAttention()

Q = torch.rand(64, 10, 512)
K, V = torch.rand(64, 20, 512), torch.rand(64, 20, 512)
output = attention(Q, K, V)
print(output.shape)
        
        
