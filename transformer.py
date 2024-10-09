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
        compatibility = compatibility.reshape(bs, self.n_heads, Q_len, K_len) # <-- bs x n_heads x Q_len x K_len
        
        if mask is not None:
            compatibility = compatibility.masked_fill(mask == 0, float('-inf'))
        
        # scale factor = sqrt(head_dim), because head_dim is equal to the dimension of the key dk (as in the paper)
        compatibility_scaled = compatibility / (self.head_dim ** 0.5)
        attention = F.softmax(compatibility_scaled, dim=-1)
        attention = attention.reshape(bs * self.n_heads, Q_len, K_len) # <-- (bs * n_heads) x Q_len x K_len

        V = V.reshape(bs * self.n_heads, V_len, self.head_dim) # <-- (bs * n_heads) x V_len x head_dim
        output = torch.bmm(attention, V) # <-- (bs * n_heads) x Q_len x head_dim
        
        output = output.reshape(bs, self.n_heads, Q_len, self.head_dim).transpose(1, 2) # <-- bs x Q_len x n_heads x head_dim
        output = output.reshape(bs, Q_len, self.embed_size) # <-- bs x Q_len x embed_size
        output = self.output_layer(output)
        
        return output
  

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, n_heads, drop_rate, ff_expansion, decoder=False):
        super().__init__()
        self.decoder = decoder
        if decoder:
            self.dec_norm = nn.LayerNorm(embed_size)
            self.dec_attention = MultiHeadAttention(embed_size, n_heads)
        
        self.attention = MultiHeadAttention(embed_size, n_heads)
        self.l_norm1 = nn.LayerNorm(embed_size)
        self.l_norm2 = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(drop_rate)
        
        self.ff_net = nn.Sequential(
            nn.Linear(embed_size, embed_size * ff_expansion),
            nn.ReLU(),
            nn.Linear(embed_size * ff_expansion, embed_size)
        )
        
    def forward(self, query, key, value, x_mask=None, y_mask=None):
        x = query
        
        if self.decoder:
            attention = self.dec_attention(x, x, x, y_mask)
            query = self.dec_norm(attention + x)
            query = self.dropout(query)
        
        attention = self.attention(query, key, value, x_mask)
  
        x = self.l_norm1(attention + x)
        x = self.dropout(x)
        ff = self.ff_net(x)
        x = self.l_norm2(ff + x)
        x = self.dropout(x)
        
        return x
        
class Encoder(nn.Module):
    def __init__(self, x_vocab_size, embed_size, n_heads, n_layers, ff_expansion, drop_rate, max_len, device):
        super().__init__()
        self.embed_size = embed_size
        self.device = device
        
        self.pos_embedding = nn.Embedding(max_len, embed_size)
        self.token_embedding = nn.Embedding(x_vocab_size, embed_size)
        
        self.layers = nn.ModuleList([
            TransformerBlock(embed_size, n_heads, drop_rate, ff_expansion
                             )for _ in range(n_layers)
        ])
        self.dropout = nn.Dropout(drop_rate)
        
    def forward(self, x, mask):
        bs, seq_len = x.shape
        positions = torch.arange(0, seq_len).expand(bs, seq_len).to(self.device)
        
        x = self.token_embedding(x) + self.pos_embedding(positions)
        x = self.dropout(x)
        
        for layer in self.layers:
            x = layer(x, x, x, mask)
        # QKV are created inside the TransformerBlock with a separate linear layer
        return x
    
class Decoder(nn.Module):
    def __init__(self, y_vocab_size, embed_size, n_heads, n_layers, ff_expansion, drop_rate, max_len, device):
        super().__init__()
        self.embed_size = embed_size
        self.device = device
        
        self.pos_embedding = nn.Embedding(max_len, embed_size)
        self.token_embedding = nn.Embedding(y_vocab_size, embed_size)
        
        self.layers = nn.ModuleList([
            TransformerBlock(embed_size, n_heads, drop_rate, ff_expansion, decoder=True
                                ) for _ in range(n_layers)
        ])
        
        self.fc_net = nn.Linear(embed_size, y_vocab_size)
        self.dropout = nn.Dropout(drop_rate)
        
    def forward(self, x, encoder_out, x_mask, y_mask):
        bs, seq_len = x.shape
        positions = torch.arange(0, seq_len).expand(bs, seq_len).to(self.device)
        
        x = self.token_embedding(x) + self.pos_embedding(positions)
        x = self.dropout(x)
        
        for layer in self.layers:
            x = layer(x, encoder_out, encoder_out, x_mask, y_mask)
        
        x = self.fc_net(x)
        return x
        

class Transformer(nn.Module):
    def __init__(self, x_vocab_size, y_vocab_size, x_pad_tok, y_pad_tok,
                 embed_size=512, n_heads=8, n_layers=6, ff_expansion=4, 
                 drop_rate=0.1, max_len=100, device='cpu'):
        
        super().__init__()
        
        self.encoder = Encoder(x_vocab_size, embed_size, n_heads, n_layers, 
                               ff_expansion, drop_rate, max_len, device)
        
        self.decoder = Decoder(y_vocab_size, embed_size, n_heads, n_layers, 
                               ff_expansion, drop_rate, max_len, device)
        
        self.x_pad_tok = x_pad_tok
        self.y_pad_tok = y_pad_tok
        self.device = device
        
    def x_mask_create(self, x):
        return (x != self.x_pad_tok).unsqueeze(1).unsqueeze(2).to(self.device)
    
    def y_mask_create(self, y):
        bs, y_len = y.shape
        return torch.tril(torch.ones(y_len, y_len)).expand(bs, 1, y_len, y_len).to(self.device)
    
    def forward(self, x, y):
        x_mask = self.x_mask_create(x)
        y_mask = self.y_mask_create(y)
        
        encoder_out = self.encoder(x, x_mask)
        decoder_out = self.decoder(y, encoder_out, x_mask, y_mask)
        
        return decoder_out
               
# Testing
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # <PAD> is 0
    # <BOS> is 1
    # <EOS> is 2
    
    x_sentence1 = [1, 5, 6, 4, 3, 9, 5, 2, 0]
    x_sentence2 = [1, 8, 7, 3, 4, 5, 6, 7, 2]
    x = torch.tensor([x_sentence1, x_sentence2]).to(device)

    y_sentence1 = [1, 7, 4, 3, 5, 9, 2, 0]
    y_sentence2 = [1, 5, 6, 2, 4, 7, 6, 2]
    y = torch.tensor([y_sentence1, y_sentence2]).to(device)

    # Padding tokens are 0
    x_pad_tok = 0
    y_pad_tok = 0
    
    x_vocab_size = 10
    y_vocab_size = 10
    
    # Input shape: (bs, seq_len)
    print(x.shape, y.shape)
    
    model = Transformer(x_vocab_size, y_vocab_size, x_pad_tok, y_pad_tok).to(device)
    out = model(x, y[:, :-1])
    
    # Output shape: (bs, seq_len, y_vocab_size)
    # a probability distribution over the vocabulary for each token in the sequence
    print(out.shape)