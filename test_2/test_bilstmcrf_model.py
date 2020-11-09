#!/usr/bin/env python
# coding: utf-8

import torch
import argparse
import gensim
import os

if torch.cuda.is_available():
    print("using cuda")
    device = torch.device('cuda:1')

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--num_layers', type=int, default=2)
parser.add_argument('--ckp', type=str, default='./ckp/model_6.pt')
args = parser.parse_args()

# ======================================= 加载数据集并处理 ======================================= #

#读入验证集
with open('./conll03.test', encoding='utf-8') as ftest_feature:
    test_feature_line = [line.strip() for line in ftest_feature.readlines()]
test_feature_line = [line.lower().split(" ") for line in test_feature_line]

#获得单词字典
#w2v_model = Word2Vec.load('word2vec_model.txt')
w2v_model = gensim.models.KeyedVectors.load_word2vec_format('./w2v_model/word2vec_model_2.txt',binary=False, encoding='utf-8')
print("loading word2vec_model......")
word2id = dict(zip(w2v_model.wv.index2word,range(len(w2v_model.wv.index2word))))                              # word -> id
id2word = {idx:word for idx,word in enumerate(w2v_model.wv.index2word)}                                       # id -> word
unk = 0#word2id['[UNK]']              #UNK:低频词
padding_value = 0#word2id['[PAD]']    #PAD:填充词
#获得标签字典
label2id = {'O':0, 'B-LOC':1, 'B-PER':2, 'B-ORG':3, 'I-PER':4, 'I-ORG':5, 'B-MISC':6, 'I-LOC':7, 'I-MISC':8, 'START':9, 'STOP':10}
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
    sample_features = torch.nn.utils.rnn.pad_sequence(sample_features, batch_first=True, padding_value=0) 
    return sample_features, sample_labels, data_length

test_data = get_data(test_feature)
#处理非定长序列
test_dataloader = torch.utils.data.DataLoader(test_data, args.batch_size, collate_fn=collate_fn, shuffle=False)

# ======================================= 模型定义 ======================================= #
class BiLSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, embedding_vector, label2id):
        super(BiLSTM,self).__init__()
        
        # ============================ BiLSTM的系列参数 ============================ #
        self.hidden_size = hidden_size
        self.embedding = torch.nn.Embedding(input_size, hidden_size)
        self.embedding.weight.data.copy_(torch.from_numpy(embedding_vector))
        self.embedding.weight.requires_grad = False
        self.bilstm = torch.nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=args.num_layers, 
                                    batch_first=True, dropout=0.5, bidirectional=True)
        self.out = torch.nn.Linear(2 * hidden_size,output_size)
        
        # ============================= CRF的系列参数 ============================= #
        self.tagset_size = len(label2id)
        self.tag_to_ix = label2id
        #转移矩阵，transitions[i][j]表示从label j转移到label i的概率,虽然是随机生成的但是后面会迭代更新
        self.transitions = torch.nn.Parameter(torch.randn(self.tagset_size,self.tagset_size))
        #这两个语句强制执行了这样的约束条件：我们永远不会转移到开始标签，也永远不会从停止标签转移
        self.transitions.data[self.tag_to_ix['START'], :] = -10     #从任何标签转移到START_TAG不可能
        self.transitions.data[:, self.tag_to_ix['STOP']] = -10      #从STOP_TAG转移到任何标签不可能

# ======================================= 模型前向传播 ======================================= #
    def forward(self, x, batch_seq_len):
        
        lstm_feats = self.BiLSTM(x, batch_seq_len) #batch_seq_len是每个batch中的序列的真实长度
        score, tag_seq = self.CRF(lstm_feats, batch_seq_len) #BiLSTM处理结果作为CRF的输入,输出为分数和预测的标签序列
        return score, tag_seq

# ======================================= BiLSTM部分 ======================================= #
    def BiLSTM(self, x, batch_seq_len):
        batch_size = x.size(0)
        seq_len = x.size(1)
        x = self.embedding(x)
        
        h = torch.zeros(2*args.num_layers, batch_size, self.hidden_size).to(x.device) 
        c = torch.zeros(2*args.num_layers, batch_size, self.hidden_size).to(x.device) 
        
        #pack_padded_sequence和pad_packed_sequence可以避免很多不必要的计算
        x = torch.nn.utils.rnn.pack_padded_sequence(x,batch_seq_len, batch_first=True)
        output, hidden = self.bilstm(x, (h, c))
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(output,batch_first=True)
        lstm_feats = self.out(output)     
        return lstm_feats
     
# ======================================= CRF的decode部分 ======================================= #
    def CRF(self, feats, batch_seq_len):
        # 加入batch的信息
        batch_path_score = []
        batch_best_path = []
        for batch in range(feats.shape[0]):
            backpointers = []

            init_vvars = torch.full((1, self.tagset_size), -10.).to(device)    #保证了一定是从START到其他标签
            init_vvars[0][self.tag_to_ix['START']] = 0                            #START的位置变成0
            forward_var_list = []
            forward_var_list.append(init_vvars)                           
            for j, feat in enumerate(feats[batch]):
                if j >= batch_seq_len[batch]:         #按照真实长度截取计算的路径
                    break
                gamar_r_l = torch.stack([forward_var_list[j]] * feats[batch].shape[1])
                gamar_r_l = torch.squeeze(gamar_r_l)
                next_tag_var = gamar_r_l + self.transitions
                # bptrs_t=torch.argmax(next_tag_var,dim=0)
                viterbivars_t, bptrs_t = torch.max(next_tag_var, dim=1)

                t_r1_k = torch.unsqueeze(feats[batch][j], 0)
                forward_var_new = torch.unsqueeze(viterbivars_t, 0) + t_r1_k

                forward_var_list.append(forward_var_new)
                backpointers.append(bptrs_t.tolist())

            terminal_var = forward_var_list[-1] + self.transitions[self.tag_to_ix['STOP']]
            best_tag_id = torch.argmax(terminal_var).tolist()
            path_score = terminal_var[0][best_tag_id]

            # 根据动态规划，由最后的节点，向前选取最佳的路径
            best_path = [best_tag_id]
            for bptrs_t in reversed(backpointers):
                best_tag_id = bptrs_t[best_tag_id]
                best_path.append(best_tag_id)
            start = best_path.pop()
            assert start == self.tag_to_ix['START']
            best_path.reverse()                                 # 把从后向前的路径正过来
            
            batch_path_score.append(path_score)
            batch_best_path.append(best_path)

        return batch_path_score, batch_best_path

# ======================================= 模型训练 ======================================= #

# 让Embedding层使用训练好的Word2Vec权重
embedding_matrix = w2v_model.wv.vectors
input_size = embedding_matrix.shape[0]   
hidden_size = embedding_matrix.shape[1]  
model = BiLSTM(input_size, hidden_size, 11, embedding_matrix,label2id).to(device)
if os.path.exists(args.ckp):
    print("loading model......")
    model.load_state_dict(torch.load(args.ckp))

f_re = open("result.txt", "w")
model.eval()
with torch.no_grad():
    for data_x, data_y, batch_seq_len in test_dataloader:
        _, out = model(data_x.to(device),batch_seq_len)                      #out就是路径序列 [10, 40]
        out = [id2label[idx] for idx in out[0]]
        out = ' '.join(out)
        f_re.write(out+'\n')
f_re.close()
