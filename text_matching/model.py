import torch
import torch.nn as nn
import torch.nn.functional as F

class ESIM(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, embedding_vector):
        super(ESIM,self).__init__()
        
        self.hidden_size = hidden_size
        self.embedding = torch.nn.Embedding(input_size, hidden_size)
        self.embedding.weight.data.copy_(torch.from_numpy(embedding_vector))
        self.embedding.weight.requires_grad = False
        self.bilstm1 = torch.nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, batch_first=True, dropout=0.5, bidirectional=True)
        self.bilstm2 = torch.nn.LSTM(input_size=hidden_size * 8, hidden_size=hidden_size, batch_first=True, dropout=0.5, bidirectional=True)
        self.fc = nn.Sequential(
            nn.BatchNorm1d(hidden_size * 8),
            nn.Linear(hidden_size * 8, 2),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(2),
            nn.Dropout(0.5),
            nn.Linear(2, 2),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(2),
            nn.Dropout(0.5),
            nn.Linear(2, output_size),
            nn.Softmax(dim=-1)
        )
    
    def attention(self, seq1, seq2, mask1, mask2):
        # 首先计算出eij也就是相似度
        eik = torch.matmul(seq2, seq1.transpose(1, 2))
        ekj = torch.matmul(seq1, seq2.transpose(1, 2))
        
        # mask操作：将相似度矩阵中值为1（.的填充id）的那些值全部用-1e9mask掉
        eik = eik.masked_fill(mask1.unsqueeze(-1) == 1, -1e9)
        ekj = ekj.masked_fill(mask2.unsqueeze(-1) == 1, -1e9)        
        
        # 归一化用于后续加权计算
        eik = F.softmax(eik, dim=-1)
        ekj = F.softmax(ekj, dim=-1)
        
        # 通过相似度和b的加权和计算出ai，通过相似度和a的加权和计算出bj
        ai = torch.matmul(ekj, seq2)
        bj = torch.matmul(eik, seq1)
        return ai, bj
    
    def submul(self, x1, x2):
        # 计算差和积
        sub = x1 - x2
        mul = x1 * x2
        return torch.cat([sub, mul], -1)
    
    def pooling(self, x):
        # 通过平均池和最大池获得固定长度的向量，并拼接送至最终分类器
        p1 = F.avg_pool1d(x.transpose(1,2), x.size(1)).squeeze(-1)
        p2 = F.max_pool1d(x.transpose(1,2), x.size(1)).squeeze(-1)
        return torch.cat([p1,p2], 1)

    def forward(self, seq1, seq2, mask1, mask2):
        
        # ==================== embedding ==================== #
        seq1 = self.embedding(seq1)
        seq2 = self.embedding(seq2)

        # ==================== bilstm  ==================== #
        bi1_1, _ = self.bilstm1(seq1)
        bi1_2, _ = self.bilstm1(seq2)

        # ==================== attention ==================== #
        ai, bj = self.attention(bi1_1, bi1_2, mask1, mask2)
        # 计算差和积然后和原向量合并，对应论文中 ma=[-a;~a;-a-~a;-a*~a] 和 mb=[-b;~b;-b-~b;-b*~b]
        ma = torch.cat([bi1_1, ai, self.submul(bi1_1, ai)], -1)
        mb = torch.cat([bi1_2, bj, self.submul(bi1_2, bj)], -1)

        # ==================== bilstm ==================== #
        bi2_1, _ = self.bilstm2(ma)
        bi2_2, _ = self.bilstm2(mb)

        # ==================== fc ==================== #
        output_1 = self.pooling(bi2_1)
        output_2 = self.pooling(bi2_2)
        output = torch.cat([output_1, output_2], -1)
        output = self.fc(output)
        
        return output

""" class ESIM(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, embedding_vector):
        super(ESIM,self).__init__()
        
        self.hidden_size = hidden_size
        self.embedding = torch.nn.Embedding(input_size, hidden_size)
        self.embedding.weight.data.copy_(torch.from_numpy(embedding_vector))
        self.embedding.weight.requires_grad = False
        self.bilstm1 = torch.nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, batch_first=True, dropout=0.5, bidirectional=True)
        self.bilstm2 = torch.nn.LSTM(input_size=hidden_size*8, hidden_size=hidden_size, batch_first=True, dropout=0.5, bidirectional=True)
        self.fc = nn.Sequential(
            nn.BatchNorm1d(self.hidden_size * 8),
            nn.Linear(self.hidden_size * 8, 2),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(2),
            nn.Dropout(0.5),
            nn.Linear(2, 2),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(2),
            nn.Dropout(0.5),
            nn.Linear(2, output_size),
            nn.Softmax(dim=-1)
        )
    
    def soft_attention_align(self, x1, x2):

        attention = torch.matmul(x1, x2.transpose(1, 2))
        weight1 = F.softmax(attention, dim=-1)
        x1_align = torch.matmul(weight1, x2)
        weight2 = F.softmax(attention.transpose(1, 2), dim=-1)
        x2_align = torch.matmul(weight2, x1)
        return x1_align, x2_align
    
    def submul(self, x1, x2):
        mul = x1 * x2
        sub = x1 - x2
        return torch.cat([sub, mul], -1)
    
    def apply_multiple(self, x):
        p1 = F.avg_pool1d(x.transpose(1, 2), x.size(1)).squeeze(-1)
        p2 = F.max_pool1d(x.transpose(1, 2), x.size(1)).squeeze(-1)
        return torch.cat([p1, p2], 1)

    def forward(self, seq1, seq2):
        
        # ==================== embedding ==================== #
        batch_size, seq_len1, seq_len2 = seq1.shape[0], seq1.shape[1], seq2.shape[1]
        seq1 = self.embedding(seq1)
        seq2 = self.embedding(seq2)
        #print(seq1.shape)
        #print(seq2.shape)
        #h = torch.zeros(2, batch_size, self.hidden_size).to(device) 
        #c = torch.zeros(2, batch_size, self.hidden_size).to(device) 

        # ==================== bilstm  ==================== #
        #x1 = torch.nn.utils.rnn.pack_padded_sequence(x1,batch_seq_len1, batch_first=True)
        output1, _ = self.bilstm1(seq1) #, (h, c)
        #output1, _ = torch.nn.utils.rnn.pad_packed_sequence(output1,batch_first=True)
        #x2 = torch.nn.utils.rnn.pack_padded_sequence(x2,batch_seq_len2, batch_first=True)
        output2, _ = self.bilstm1(seq2) #, (h, c)
        #output2, _ = torch.nn.utils.rnn.pad_packed_sequence(output2,batch_first=True)
        #print(output1.shape) #torch.Size([2, 50, 128])
        #print(output2.shape) #torch.Size([2, 50, 128])

        # ==================== attention ==================== #
        q1_align, q2_align = self.soft_attention_align(output1, output2)
        q1_combined = torch.cat([output1, q1_align, self.submul(output1, q1_align)], -1)
        q2_combined = torch.cat([output2, q2_align, self.submul(output2, q2_align)], -1)
        #print(q1_align.shape) #torch.Size([2, 50, 256])
        #print(q2_align.shape) #torch.Size([2, 50, 256])
        #print(q1_combined.shape) #torch.Size([2, 50, 256])
        #print(q2_combined.shape) #torch.Size([2, 50, 256])

        # ==================== bilstm ==================== #
        q1_compose, _ = self.bilstm2(q1_combined)
        #print(q1_compose.shape)
        q2_compose, _ = self.bilstm2(q2_combined)
        #print(q2_compose.shape)

        # ==================== fc ==================== #
        q1_rep = self.apply_multiple(q1_compose)
        q2_rep = self.apply_multiple(q2_compose)
        x = torch.cat([q1_rep, q2_rep], -1)
        similarity = self.fc(x) 
        
        return similarity """