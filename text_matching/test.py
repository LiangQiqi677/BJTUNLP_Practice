#encoding: utf-8
import torch
from gensim.models import Word2Vec, KeyedVectors
import model
from torchtext.data import Field, Example, Dataset, BucketIterator, Iterator

if torch.cuda.is_available():
    print("using cuda......")
    device = torch.device('cuda:6')

#读入测试集
with open('./snli.test', 'r', encoding='utf-8') as ftest_feature:
    test_feature_line = [line.lower().strip() for line in ftest_feature.readlines()]
#转换大小写并用split分隔开存入列表
test_feature1_line, test_feature2_line = [], []
for line in test_feature_line:
    temp1, temp2 = line.split(' ||| ')
    temp1 = temp1.replace('.', ' .')
    temp1 = temp1.replace(',', ' ,')
    temp2 = temp2.replace('.', ' .')
    temp2 = temp2.replace(',', ' ,')
    test_feature1_line.append(temp1)
    test_feature2_line.append(temp2)
test_feature1_line = [line.split(" ") for line in test_feature1_line]
test_feature2_line = [line.split(" ") for line in test_feature2_line]

#生成词向量和词表
w2v_model = KeyedVectors.load_word2vec_format('w2v_model/word2vec_model_1.txt',binary=False, encoding='utf-8')
print('loading word2vec_model......')
word2id = dict(zip(w2v_model.wv.index2word,range(len(w2v_model.wv.index2word))))                              # word -> id
id2word = {idx:word for idx,word in enumerate(w2v_model.wv.index2word)}                                       # id -> word
feature_pad = 0
label2id = {'neutral':0, 'entailment':1, 'contradiction':2, '-':3}
id2label = {idx:word for idx,word in enumerate(label2id)}   

test_feature1 = [[word2id[word] if word in word2id else feature_pad for word in line] for line in test_feature1_line]
test_feature2 = [[word2id[word] if word in word2id else feature_pad for word in line] for line in test_feature2_line]

embedding_matrix = w2v_model.wv.vectors
input_size, hidden_size = embedding_matrix.shape[0], embedding_matrix.shape[1]
loss_func = torch.nn.CrossEntropyLoss()
net = model.ESIM(input_size, hidden_size, 4, embedding_matrix).to(device)

sentence1_field = Field(sequential=True, use_vocab = False, batch_first=True, fix_length = 50, pad_token = feature_pad)
sentence2_field = Field(sequential=True, use_vocab = False, batch_first=True, fix_length = 50, pad_token = feature_pad)
fields = [('sentence1', sentence1_field), ('sentence2', sentence2_field)]
#获得测试集的Iterator
test_examples = []
for index in range(len(test_feature_line)):
    test_examples.append(Example.fromlist([test_feature1[index], test_feature2[index]], fields))
test_set = Dataset(test_examples, fields)
test_iter = Iterator(test_set, batch_size=32, device=device, train=False, shuffle=False, sort=False)

net = model.ESIM(input_size, hidden_size, 4, embedding_matrix).to(device)
net.load_state_dict(torch.load('ckp/model_3.pt'))
net.eval()
out_epoch = []
with torch.no_grad():
    for batch in test_iter:
        seq1 = batch.sentence1
        seq2 = batch.sentence2
        mask1 = (seq1 == 1)
        mask2 = (seq2 == 1)
        out = net(seq1.to(device),seq2.to(device), mask1.to(device), mask2.to(device))
        prediction = out.argmax(dim=1).data.cpu().numpy().tolist()
        out_epoch.extend(prediction)

result = [id2label[line] for line in out_epoch]
with open('result.txt', 'w', encoding='utf-8') as f_result:
    f_result.write('\n'.join(result))
