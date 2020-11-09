#!/usr/bin/env python
# coding: utf-8

import torch
import argparse
from gensim.models import Word2Vec
import datetime
from sklearn.metrics import classification_report
import os

parser = argparse.ArgumentParser()
parser.add_argument('--cuda', type=str, default='cuda:4')
parser.add_argument('--batch_size', type=int, default=1024)
parser.add_argument('--num_epoch', type=int, default=30)
parser.add_argument('--lr', type=float, default=1e-1)
parser.add_argument('--weight_decay', type=float, default=1e-8)
parser.add_argument('--num_layers', type=int, default=1)
parser.add_argument('--ckp', type=str, default='./ckp/model_1.pt')
parser.add_argument('--f1_min', type=float, default=0.7)
args = parser.parse_args()

if torch.cuda.is_available():
    print("using cuda")
    device = torch.device(args.cuda)

# ======================================= 加载数据集并处理 ======================================= #

#读入训练集
with open('./CoNLL2003_NER/train/seq.in', encoding='utf-8') as ftrain_feature:
    train_feature_line = [line.strip() for line in ftrain_feature.readlines()]
with open('./CoNLL2003_NER/train/seq.out', encoding='utf-8') as ftrain_label:
    train_label_line = [line.strip() for line in ftrain_label.readlines()]
#读入验证集
with open('./CoNLL2003_NER/test/seq.in', encoding='utf-8') as ftest_feature:
    test_feature_line = [line.strip() for line in ftest_feature.readlines()]
with open('./CoNLL2003_NER/test/seq.out', encoding='utf-8') as ftest_label:
    test_label_line = [line.strip() for line in ftest_label.readlines()]

#转换大小写并用split分隔开存入列表
train_feature_line = [line.lower().split(" ") for line in train_feature_line]
train_label_line = [line.split(" ") for line in train_label_line]
test_feature_line = [line.lower().split(" ") for line in test_feature_line]
test_label_line = [line.split(" ") for line in test_label_line]

#获得单词字典
#model_word2vec = Word2Vec([['[UNK]'],['[PAD]']]+train_feature_line, sg=1, min_count=1, size=128, window=5)
#model_word2vec.save('word2vec_model.txt')
w2v_model = Word2Vec.load('word2vec_model.txt')
word2id = dict(zip(w2v_model.wv.index2word,range(len(w2v_model.wv.index2word))))                              # word -> id
id2word = {idx:word for idx,word in enumerate(w2v_model.wv.index2word)}                                       # id -> word
unk = word2id['[UNK]']              #UNK:低频词
padding_value = word2id['[PAD]']    #PAD:填充词
#获得标签字典
label2id = {'O':0, 'B-LOC':1, 'B-PER':2, 'B-ORG':3, 'I-PER':4, 'I-ORG':5, 'B-MISC':6, 'I-LOC':7, 'I-MISC':8, 'START':9, 'STOP':10}

#获得数据和标签序列
train_feature = [[word2id[word] if word in word2id else unk for word in line] for line in train_feature_line]
train_label = [[label2id[word] for word in line] for line in train_label_line]
test_feature = [[word2id[word] if word in word2id else unk for word in line] for line in test_feature_line]
test_label = [[label2id[word] for word in line] for line in test_label_line]

#转成Tensor的形式
train_feature = [torch.Tensor(line).long() for line in train_feature]
train_label = [torch.Tensor(line).long() for line in train_label]
test_feature = [torch.Tensor(line).long() for line in test_feature]
test_label = [torch.Tensor(line).long() for line in test_label]

def get_data(sample_features, sample_labels):
    sample_data = []                                                    #为了能够将data放到DataLoader中
    for i in range(len(sample_features)):
        temp = []
        temp.append(sample_features[i])
        temp.append(sample_labels[i])
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

train_data = get_data(train_feature, train_label)
test_data = get_data(test_feature, test_label)

#处理非定长序列
train_dataloader = torch.utils.data.DataLoader(train_data, args.batch_size, collate_fn=collate_fn, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_data, args.batch_size, collate_fn=collate_fn, shuffle=True)

class BiLSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, embedding_vector):
        super(BiLSTM,self).__init__()
        
        self.hidden_size = hidden_size
        self.embedding = torch.nn.Embedding(input_size, hidden_size)
        self.embedding.weight.data.copy_(torch.from_numpy(embedding_vector))
        self.embedding.weight.requires_grad = False
        self.bilstm = torch.nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=args.num_layers, 
                                    batch_first=True, dropout=0.5, bidirectional=True)
        self.out = torch.nn.Linear(2 * hidden_size, output_size)
    
    def forward(self, x, batch_seq_len):
        batch_size = x.size(0)
        seq_len = x.size(1)
        x = self.embedding(x)
        h = torch.zeros(2*args.num_layers, batch_size, self.hidden_size).to(device) 
        c = torch.zeros(2*args.num_layers, batch_size, self.hidden_size).to(device) 
        
        x = torch.nn.utils.rnn.pack_padded_sequence(x,batch_seq_len, batch_first=True)
        output, hidden = self.bilstm(x, (h, c))
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(output,batch_first=True)
        output = self.out(output)
        
        return output

#通过batch_seq_len留下out和label中不重复的部分进行loss计算和指标计算
def processing_len(out, label, batch_seq_len):
    out_pred = out[:batch_seq_len[0],:]
    out_true = label[:batch_seq_len[0]]
    for i in range(1,len(batch_seq_len)):
        out_pred = torch.cat((out_pred,out[i*batch_seq_len[0]:i*batch_seq_len[0]+batch_seq_len[i],:]),dim=0)
        out_true = torch.cat((out_true,label[i*batch_seq_len[0]:i*batch_seq_len[0]+batch_seq_len[i]]),dim=0)

    return out_pred, out_true

# ======================================= 模型测试 ======================================= #
f1_min = args.f1_min
target_names = ['O', 'B-LOC', 'B-PER', 'B-ORG', 'I-PER', 'I-ORG', 'B-MISC', 'I-LOC', 'I-MISC']    
def test_evaluate(model, test_dataloader):
    test_l, n = 0.0, 0
    out_epoch, label_epoch = [], []
    global f1_min
    model.eval()
    with torch.no_grad():
        for data_x, data_y, batch_seq_len in test_dataloader:

            out = model(data_x.to(device),batch_seq_len).view(-1, 9)
            label = [line.numpy().tolist() for line in data_y]
            for line in label:
                for i in range(data_x.shape[1]-len(line)):
                    line.append(line[len(line)-1])
            label = torch.tensor(label).view(-1,1).squeeze(-1).to(device)

            out, label = processing_len(out, label,batch_seq_len)
            prediction = out.argmax(dim=1).data.cpu().numpy()
            label = label.data.cpu().numpy()

            #测试集评价指标
            out_epoch.extend(prediction)
            label_epoch.extend(label)
            test_l += loss.item()
            n += 1

        print(classification_report(label_epoch, out_epoch, target_names=target_names, digits=6))
        report = classification_report(label_epoch, out_epoch, output_dict=True)
        if report['macro avg']['f1-score'] > f1_min : 
            f1_min = report['macro avg']['f1-score']
            torch.save(model.state_dict(), args.ckp)
            print("save model......")

    return test_l/n


# 让Embedding层使用训练好的Word2Vec权重
# model_word2vec = Word2Vec(train_feature_line, sg=1, min_count=1, size=128, window=5)
# model_word2vec.save('word2vec_model.txt')
if os.path.exists('word2vec_model.txt'):
    print("loading word2vec_model......")
    w2v_model = Word2Vec.load('word2vec_model.txt')
embedding_matrix = w2v_model.wv.vectors
input_size = embedding_matrix.shape[0]   
hidden_size = embedding_matrix.shape[1]
loss_func = torch.nn.CrossEntropyLoss()
model = BiLSTM(input_size, hidden_size, 9, embedding_matrix).to(device)
model.load_state_dict(torch.load(args.ckp, map_location='cpu'))
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)  #

for epoch in range(args.num_epoch):
    model.train()
    train_l, n = 0.0, 0
    start = datetime.datetime.now()

    for data_x, data_y, batch_seq_len in train_dataloader:
        
        out = model(data_x.to(device),batch_seq_len).view(-1, 9)
        label = [line.numpy().tolist() for line in data_y]
        for line in label:
            for i in range(data_x.shape[1]-len(line)):
                line.append(line[len(line)-1])
        label = torch.tensor(label).view(-1,1).squeeze(-1).to(device)
        
        out, label = processing_len(out, label,batch_seq_len)
        loss = loss_func(out, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        prediction = out.argmax(dim=1).data.cpu().numpy()
        label = label.data.cpu().numpy()

        train_l += loss.item()
        n += 1

    test_loss = test_evaluate(model, test_dataloader)
    end = datetime.datetime.now()

    print('epoch %d, train_loss %f, train_loss %f, fi_max %f, time %s'% (epoch+1, train_l/n, test_loss, f1_min, end-start))




