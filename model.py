import torch.nn as nn
import torch
import numpy as np

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
    
class TransformerNet(nn.Module):
  def __init__(self, num_src_vocab, num_tgt_vocab, embedding_dim, hidden_size, nheads, n_layers, max_src_len, max_tgt_len, dropout):
    super(TransformerNet, self).__init__()
    # embedding layers
    self.enc_embedding = nn.Embedding(num_src_vocab, embedding_dim)
    self.dec_embedding = nn.Embedding(num_tgt_vocab, embedding_dim)

    # positional encoding layers
    self.enc_pe = PositionalEncoding(embedding_dim, max_len = max_src_len)
    self.dec_pe = PositionalEncoding(embedding_dim, max_len = max_tgt_len)

    # encoder/decoder layers
    enc_layer = nn.TransformerEncoderLayer(embedding_dim, nheads, hidden_size, dropout)
    dec_layer = nn.TransformerDecoderLayer(embedding_dim, nheads, hidden_size, dropout)
    self.encoder = nn.TransformerEncoder(enc_layer, num_layers = n_layers)
    self.decoder = nn.TransformerDecoder(dec_layer, num_layers = n_layers)

    # final dense layer
    self.dense = nn.Linear(embedding_dim, num_tgt_vocab)
    self.log_softmax = nn.LogSoftmax()

  def forward(self, src, tgt):
    src, tgt = self.enc_embedding(src).permute(1, 0, 2), self.dec_embedding(tgt).permute(1, 0, 2)
    src, tgt = self.enc_pe(src), self.dec_pe(tgt)
    memory = self.encoder(src)
    transformer_out = self.decoder(tgt, memory)
    final_out = self.dense(transformer_out)
    return self.log_softmax(final_out)

class MTDataset(torch.utils.data.Dataset):
  def __init__(self, eng_sentences, deu_sentences):
    # import and initialize dataset    
    self.source = np.array(eng_sentences, dtype = int)
    self.target = np.array(deu_sentences, dtype = int)
    
  def __getitem__(self, idx):
    # get item by index
    return self.source[idx], self.target[idx]
  
  def __len__(self):
    # returns length of data
    return len(self.source)
