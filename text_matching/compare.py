
import jsonlines
label2id = {'neutral':0, 'entailment':1, 'contradiction':2, '-':3}
dev_gold_label, dev_sentence1,  dev_sentence2 = [], [], []
with open('snli_1.0/snli_1.0_dev.jsonl', 'r', encoding='utf-8') as f:
    for item in jsonlines.Reader(f):
        dev_gold_label.append(item['gold_label'].lower())
        temp = item['sentence1'].replace('.', ' .')
        temp = temp.replace(',', ' ,')
        dev_sentence1.append(temp.lower())
        temp = item['sentence2'].replace('.', ' .')
        temp = temp.replace(',', ' ,')
        dev_sentence2.append(temp.lower())
with open('snli_1.0/dev_gold_label.txt', 'w', encoding='utf-8') as f:
    for line in dev_gold_label:
        f.write(line + '\n')
with open('snli_1.0/dev_sentence1_split.txt', 'w', encoding='utf-8') as f:
    for line in dev_sentence1:
        f.write(line + '\n')
with open('snli_1.0/dev_sentence2_split.txt', 'w', encoding='utf-8') as f:
    for line in dev_sentence2:
        f.write(line + '\n')
num = 0
with open('snli_1.0/dev_gold_label.txt', 'r', encoding='utf-8') as f:
    for line in f:
        num += 1
print(num)
num = 0
with open('snli_1.0/dev_sentence1_split.txt', 'r', encoding='utf-8') as f:
    for line in f:
        num += 1
print(num)
num = 0
with open('snli_1.0/dev_sentence2_split.txt', 'r', encoding='utf-8') as f:
    for line in f:
        num += 1
print(num)
exit()

num = 0 
total_len = 0
with open('./snli_1.0/train_sentence1_split.txt', 'r', encoding='utf-8') as ftrain_feature1:
   for line in ftrain_feature1:
       total_len += len(line)
       num += 1
print(total_len/num)

with open('result.txt', 'r', encoding='utf-8') as f:
    result = [line.strip() for line in f.readlines()]

with open('result_1.txt', 'r', encoding='utf-8') as f1:
    result1 = [line.strip() for line in f1.readlines()]

num = 0
for i in range(len(result)):
    if result[i] != result1[i]:
        num += 1

print(num)