#encoding: utf-8
import torch
from gensim.models import Word2Vec, KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from torchtext.data import Field, Example, Dataset, BucketIterator, Iterator

def data(device, batch_size):

    # ======================================= 加载数据集并处理 ======================================= #

    #读入训练集
    with open('./snli_1.0/train_sentence1_split.txt', 'r', encoding='utf-8') as ftrain_feature1:
        train_feature1_line = [line.strip() for line in ftrain_feature1.readlines()]
    with open('./snli_1.0/train_sentence2_split.txt', 'r', encoding='utf-8') as ftrain_feature2:
        train_feature2_line = [line.strip() for line in ftrain_feature2.readlines()]
    with open('./snli_1.0/train_gold_label.txt', 'r', encoding='utf-8') as ftrain_label:
        train_label_line = [line.strip() for line in ftrain_label.readlines()]
    #读入验证集
    with open('./snli_1.0/dev_sentence1_split.txt', 'r', encoding='utf-8') as fdev_feature1:
        dev_feature1_line = [line.strip() for line in fdev_feature1.readlines()]
    with open('./snli_1.0/dev_sentence2_split.txt', 'r', encoding='utf-8') as fdev_feature2:
        dev_feature2_line = [line.strip() for line in fdev_feature2.readlines()]
    with open('./snli_1.0/dev_gold_label.txt', 'r', encoding='utf-8') as fdev_label:
        dev_label_line = [line.strip() for line in fdev_label.readlines()]
    #读入测试集
    with open('./snli_1.0/test_sentence1_split.txt', 'r', encoding='utf-8') as ftest_feature1:
        test_feature1_line = [line.strip() for line in ftest_feature1.readlines()]
    with open('./snli_1.0/test_sentence2_split.txt', 'r', encoding='utf-8') as ftest_feature2:
        test_feature2_line = [line.strip() for line in ftest_feature2.readlines()]
    with open('./snli_1.0/test_gold_label.txt', 'r', encoding='utf-8') as ftest_label:
        test_label_line = [line.strip() for line in ftest_label.readlines()]

    #用split分隔开存入列表
    train_feature1_line = [line.split(" ") for line in train_feature1_line]
    train_feature2_line = [line.split(" ") for line in train_feature2_line]
    dev_feature1_line = [line.split(" ") for line in dev_feature1_line]
    dev_feature2_line = [line.split(" ") for line in dev_feature2_line]
    test_feature1_line = [line.split(" ") for line in test_feature1_line]
    test_feature2_line = [line.split(" ") for line in test_feature2_line]

    #获得单词字典
    #model_word2vec = Word2Vec(train_feature1_line+train_feature2_line, sg=1, min_count=1, size=128, window=5)
    #model_word2vec.save('word2vec_model.txt')
    #glove_file = 'glove.840B.300d.txt'
    #tmp_file = 'w2v_model/word2vec_model_1.txt'
    #_ = glove2word2vec(glove_file, tmp_file)
    #w2v_model = Word2Vec.load('w2v_model/word2vec_model_1.txt')
    w2v_model = KeyedVectors.load_word2vec_format('w2v_model/word2vec_model_1.txt',binary=False, encoding='utf-8')
    print('loading word2vec_model......')
    word2id = dict(zip(w2v_model.wv.index2word,range(len(w2v_model.wv.index2word))))                              # word -> id
    id2word = {idx:word for idx,word in enumerate(w2v_model.wv.index2word)}                                       # id -> word
    feature_pad = 0    #PAD:填充词
    label2id = {'neutral':0, 'entailment':1, 'contradiction':2, '-':3}
    label_pad = 0

    #获得数据和标签序列
    train_feature1 = [[word2id[word] if word in word2id else feature_pad for word in line] for line in train_feature1_line]
    train_feature2 = [[word2id[word] if word in word2id else feature_pad for word in line] for line in train_feature2_line]
    train_label = [[label2id[word] if word in label2id else label_pad] for word in train_label_line]
    dev_feature1 = [[word2id[word] if word in word2id else feature_pad for word in line] for line in dev_feature1_line]
    dev_feature2 = [[word2id[word] if word in word2id else feature_pad for word in line] for line in dev_feature2_line]
    dev_label = [[label2id[word] if word in label2id else label_pad] for word in dev_label_line]
    test_feature1 = [[word2id[word] if word in word2id else feature_pad for word in line] for line in test_feature1_line]
    test_feature2 = [[word2id[word] if word in word2id else feature_pad for word in line] for line in test_feature2_line]
    test_label = [[label2id[word] if word in label2id else label_pad] for word in test_label_line]

    #print(train_feature1[0:2]) #[[0, 41, 6, 0, 194, 201, 76, 0, 1145, 37, 800, 1], [0, 41, 6, 0, 194, 201, 76, 0, 1145, 37, 800, 1]]
    #print(train_feature2[0:2]) #[[0, 41, 4, 1152, 19, 194, 36, 0, 439, 1], [0, 41, 4, 16, 0, 2374, 14, 2321, 18, 24459, 1]]
    #print(train_label[0:2])    #[[0], [2]]

    #转成Tensor的形式
    """ train_feature1 = [torch.Tensor(line).long() for line in train_feature1]
    train_feature2 = [torch.Tensor(line).long() for line in train_feature2]
    train_label = [torch.Tensor(line).long() for line in train_label]
    test_feature1 = [torch.Tensor(line).long() for line in test_feature1]
    test_feature2 = [torch.Tensor(line).long() for line in test_feature2]
    test_label = [torch.Tensor(line).long() for line in test_label] """
    
    sentence1_field = Field(sequential=True, use_vocab = False, batch_first=True, fix_length = 50, pad_token = feature_pad)
    sentence2_field = Field(sequential=True, use_vocab = False, batch_first=True, fix_length = 50, pad_token = feature_pad)
    label_field = Field(sequential=False, use_vocab=False)
    fields = [('sentence1', sentence1_field), ('sentence2', sentence2_field), ('label', label_field)]
    
    #获得训练集的Iterator
    train_examples = []
    for index in range(len(train_label)):
        train_examples.append(Example.fromlist([train_feature1[index], train_feature2[index], train_label[index]], fields))
    train_set = Dataset(train_examples, fields)
    train_iter = BucketIterator(train_set, batch_size=batch_size, device=device, shuffle=True)

    #获得验证集的Iterator
    dev_examples = []
    for index in range(len(dev_label)):
        dev_examples.append(Example.fromlist([dev_feature1[index], dev_feature2[index], dev_label[index]], fields))
    dev_set = Dataset(dev_examples, fields)
    dev_iter = Iterator(dev_set, batch_size=batch_size, device=device, train=False, shuffle=False, sort=False)

    #获得测试集的Iterator
    test_examples = []
    for index in range(len(test_label)):
        test_examples.append(Example.fromlist([test_feature1[index], test_feature2[index], test_label[index]], fields))
    test_set = Dataset(test_examples, fields)
    test_iter = Iterator(test_set, batch_size=batch_size, device=device, train=False, shuffle=False, sort=False)

    return w2v_model, train_iter, dev_iter, test_iter