#!/usr/bin/env python
# coding: utf-8

from collections import Counter
import torch
import argparse
from gensim.models import Word2Vec
import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--num_epoch', type=int, default=30)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--weight_decay', type=float, default=1e-7)
args = parser.parse_args()#args=[])

f1_min = 0.865402
if torch.cuda.is_available():
    print("using cuda")
    device = torch.device('cuda:6')


#读入训练集
with open('./CoNLL2003_NER/train/seq.in', encoding='utf-8') as ftrain_feature:
    train_feature_line = [line.strip() for line in ftrain_feature.readlines()]
with open('./CoNLL2003_NER/train/seq.out', encoding='utf-8') as ftrain_label:
    train_label_line = [line.strip() for line in ftrain_label.readlines()]
#读入测试集
with open('./conll03.test', encoding='utf-8') as ftest_feature:
    test_feature_line = [line.strip() for line in ftest_feature.readlines()]

#转换大小写并用split分隔开存入列表
train_feature_line = [line.lower().split(" ") for line in train_feature_line]
test_feature_line = [line.lower().split(" ") for line in test_feature_line]

#获得单词字典
word_counter = []
for line in train_feature_line:
    word_counter.extend(line)
word_counter = Counter(word_counter).most_common()                                          #len(counter):21009
vocab = ['[UNK]','[PAD]'] + [word[0] for word in word_counter[:int(len(word_counter)*0.8)]] #UNK:低频词；PAD:填充词
word2id = dict(zip(vocab,range(len(vocab))))                                                # word -> id
id2word = {idx:word for idx,word in enumerate(vocab)}                                       # id -> word
#获得标签字典
label2id = {'O':0, 'B-LOC':1, 'B-PER':2, 'B-ORG':3, 'I-PER':4, 'I-ORG':5, 'B-MISC':6, 'I-LOC':7, 'I-MISC':8}
id2label = {idx:word for idx,word in enumerate(label2id)}

#获得数据和标签序列
test_feature = [[word2id[word] if word in word2id else 0 for word in line] for line in test_feature_line]
#转成Tensor的形式
test_feature = [torch.Tensor(line).long() for line in test_feature]

def get_data(sample_features):
    sample_data = []                                                    #为了能够将data放到DataLoader中
    for i in range(len(sample_features)):
        temp = []
        temp.append(sample_features[i])
        temp.append(torch.zeros(1,len(sample_features[i])))
        sample_data.append(temp)
    return sample_data

def collate_fn(sample_data):
    sample_data.sort(key=lambda data: len(data[0]), reverse=True)                          #倒序排序
    sample_features, sample_labels = [], []
    for data in sample_data:
        sample_features.append(data[0])
        sample_labels.append(data[1])
    data_length = [len(data[0]) for data in sample_data]                                   #取出所有data的长度             
    sample_features = torch.nn.utils.rnn.pad_sequence(sample_features, batch_first=True, padding_value=word2id['[PAD]']) 
    return sample_features, sample_labels, data_length

test_data = get_data(test_feature)
test_dataloader = torch.utils.data.DataLoader(test_data, args.batch_size, collate_fn=collate_fn, shuffle=False)


# In[3]:


class BiLSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, embedding_vector):
        super(BiLSTM,self).__init__()
        
        self.hidden_size = hidden_size
        self.embedding = torch.nn.Embedding(input_size, hidden_size)
        self.embedding.weight.data.copy_(torch.from_numpy(embedding_vector))
        self.embedding.weight.requires_grad = False
        self.bilstm = torch.nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=1, 
                                    batch_first=True, dropout=0.5, bidirectional=True)
        self.out = torch.nn.Linear(2 * hidden_size, output_size)
    
    def forward(self, x, batch_seq_len):
        batch_size = x.size(0)
        seq_len = x.size(1)
        x = self.embedding(x)
        h = torch.zeros(2, batch_size, self.hidden_size).to(device) 
        c = torch.zeros(2, batch_size, self.hidden_size).to(device) 
        
        x = torch.nn.utils.rnn.pack_padded_sequence(x,batch_seq_len, batch_first=True)
        output, hidden = self.bilstm(x, (h, c))
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(output,batch_first=True)
        output = self.out(output)
        
        return output

#通过batch_seq_len留下out和label中不重复的部分进行loss计算和指标计算
def processing_len(out, batch_seq_len):
    out_pred = out[:batch_seq_len[0],:]
    for i in range(1,len(batch_seq_len)):
        out_pred = torch.cat((out_pred,out[i*batch_seq_len[0]:i*batch_seq_len[0]+batch_seq_len[i],:]),dim=0)
    return out_pred

f_re = open("bilstm_result.txt", "w")

def test_evaluate(model, test_dataloader, batch_size):
    test_l, test_p, test_r, test_f, n = 0.0, 0.0, 0.0, 0.0, 0
    model.eval()
    with torch.no_grad():
        for data_x, data_y, batch_seq_len in test_dataloader:
            out = model(data_x.to(device),batch_seq_len).view(-1, 9)
            out = out.argmax(dim=1).data.cpu().numpy().tolist()
            out = [id2label[idx] for idx in out]
            out = ' '.join(out)
            f_re.write(out+'\n')


loss_func = torch.nn.CrossEntropyLoss()
w2v_model = Word2Vec.load('word2vec_model.txt')
embedding_matrix = w2v_model.wv.vectors
input_size = embedding_matrix.shape[0]   
hidden_size = embedding_matrix.shape[1]
model = BiLSTM(input_size, hidden_size, 9, embedding_matrix).to(device)
model.load_state_dict(torch.load('./ckp/model_3.pt'))
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)  #
test_evaluate(model, test_dataloader, args.batch_size)
