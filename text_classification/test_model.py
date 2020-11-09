#!/usr/bin/env python
# coding: utf-8

import numpy as np
import torch
import torch.nn.utils.rnn as rnn_utils
import torch.nn.functional as F
import gensim
import argparse
from gensim.models import Word2Vec
import os

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--num_epoch', type=int, default=50)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--weight_decay', type=float, default=1e-3)
parser.add_argument('--ckp', type=str, default='./ckp/model_2.pt')
parser.add_argument('--acc_min', type=float, default=0.909207)
parser.add_argument('--nums_channels', type=int, default=50)
args = parser.parse_args()#args=[])


if torch.cuda.is_available():
    print("using cuda")
    device = torch.device('cuda:1')

with open('./test_split.txt', encoding='utf-8') as ftest_feature:
    test_feature_line = [line.strip() for line in ftest_feature.readlines()]
test_feature_line = [line.split(" ") for line in test_feature_line]

w2v_model = Word2Vec.load('word2vec_model.txt')
word2id = dict(zip(w2v_model.wv.index2word,range(len(w2v_model.wv.index2word))))                              # word -> id
id2word = {idx:word for idx,word in enumerate(w2v_model.wv.index2word)}                                       # id -> word
unk = word2id['[UNK]']              #UNK:低频词
padding_value = word2id['[PAD]']    #PAD:填充词

#获得数据序列
test_feature = [[word2id[word] if word in word2id else unk for word in line] for line in test_feature_line]

class TextCNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, embeding_vector, kernel_sizes, num_channels):
        super().__init__()
        self.hidden_size = hidden_size
        #不参与训练的嵌入层
        self.embedding = torch.nn.Embedding(num_embeddings=input_size, embedding_dim=hidden_size)
        self.embedding.weight.data.copy_(torch.from_numpy(embeding_vector))  #使用预训练的词向量
        self.embedding.weight.requires_grad = False
        #参与训练的嵌入层
        self.constant_embedding = torch.nn.Embedding(num_embeddings=input_size, embedding_dim=hidden_size)
        self.constant_embedding.weight.data.copy_(torch.from_numpy(embeding_vector))  #使用预训练的词向量
        self.dropout = torch.nn.Dropout(0.5)
        self.out_linear = torch.nn.Linear(sum(num_channels), output_size)
        self.pool = GlobalMaxPool1d()
        self.convs = torch.nn.ModuleList()
        for c, k in zip(num_channels, kernel_sizes):
            self.convs.append(torch.nn.Conv1d(in_channels=2*hidden_size, out_channels=c, kernel_size=k))
        
    def forward(self, x):
        embeddings = torch.cat((self.embedding(x), self.constant_embedding(x)), dim=2).permute(0,2,1)
        out = torch.cat([self.pool(F.relu(conv(embeddings))).squeeze(-1) for conv in self.convs], dim=1)
        out = self.out_linear(self.dropout(out))
        return out

class GlobalMaxPool1d(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return F.max_pool1d(x, kernel_size = x.shape[2])

# 让Embedding层使用训练好的Word2Vec权重
embedding_matrix = w2v_model.wv.vectors
input_size = embedding_matrix.shape[0]   #37125, 词典的大小
hidden_size = embedding_matrix.shape[1]  #50, 隐藏层单元个数
kernel_size = [3, 4, 5]
nums_channels = [args.nums_channels, args.nums_channels, args.nums_channels]
model = TextCNN(input_size, hidden_size, 2, embedding_matrix, kernel_size, nums_channels).to(device)
if os.path.exists(args.ckp):
    print("loading model......")
    model.load_state_dict(torch.load(args.ckp))
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

f=open('result.txt','w')
for data_x in test_feature:
    model.eval()
    with torch.no_grad():
        out = model(torch.Tensor(data_x).unsqueeze(0).long().to(device))
        prediction = out.argmax(dim=1).data.cpu().numpy()
        if prediction[0] == 0:
            f.write('0\n')
        else:
            f.write('1\n')
f.close()




