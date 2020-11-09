import model
import torch
import datetime
from sklearn.metrics import accuracy_score

def dev_evaluate(device, net, dev_iter, max_acc, ckp):

    dev_l, n = 0.0, 0
    out_epoch, label_epoch = [], []
    loss_func = torch.nn.CrossEntropyLoss()
    net.eval()
    with torch.no_grad():
        for batch in dev_iter:
            
            seq1 = batch.sentence1
            seq2 = batch.sentence2
            label = batch.label
            mask1 = (seq1 == 1)
            mask2 = (seq2 == 1)
            out = net(seq1.to(device),seq2.to(device), mask1.to(device), mask2.to(device))

            loss = loss_func(out, label.squeeze(-1))

            prediction = out.argmax(dim=1).data.cpu().numpy().tolist()
            label = label.view(1,-1).squeeze().data.cpu().numpy().tolist()

            #测试集评价指标
            out_epoch.extend(prediction)
            label_epoch.extend(label)
            dev_l += loss.item()
            n += 1

        acc = accuracy_score(label_epoch, out_epoch)
        if acc > max_acc : 
            max_acc = acc
            torch.save(net.state_dict(), ckp)
            print("save model......")

    return dev_l/n, acc, max_acc

def test_evaluate(device, net, test_iter):

    test_l, n = 0.0, 0
    out_epoch, label_epoch = [], []
    loss_func = torch.nn.CrossEntropyLoss()
    net.eval()
    with torch.no_grad():
        for batch in test_iter:
            
            seq1 = batch.sentence1
            seq2 = batch.sentence2
            label = batch.label
            mask1 = (seq1 == 1)
            mask2 = (seq2 == 1)
            out = net(seq1.to(device),seq2.to(device), mask1.to(device), mask2.to(device))

            loss = loss_func(out, label.squeeze(-1))

            prediction = out.argmax(dim=1).data.cpu().numpy().tolist()
            label = label.view(1,-1).squeeze().data.cpu().numpy().tolist()

            #测试集评价指标
            out_epoch.extend(prediction)
            label_epoch.extend(label)
            test_l += loss.item()
            n += 1

        acc = accuracy_score(label_epoch, out_epoch)

    return test_l/n, acc

def training(device, w2v_model, train_iter, dev_iter, test_iter, batch_size, num_epoch, lr, weight_decay, ckp, max_acc):

    embedding_matrix = w2v_model.wv.vectors
    input_size, hidden_size = embedding_matrix.shape[0], embedding_matrix.shape[1]
    loss_func = torch.nn.CrossEntropyLoss()
    net = model.ESIM(input_size, hidden_size, 4, embedding_matrix).to(device)
    #net.load_state_dict(torch.load(ckp, map_location='cpu'))
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(num_epoch):
        net.train()
        train_l, n = 0.0, 0
        start = datetime.datetime.now()
        out_epoch, label_epoch = [], []
        for batch in train_iter:
            
            seq1 = batch.sentence1
            seq2 = batch.sentence2
            label = batch.label
            mask1 = (seq1 == 1)
            mask2 = (seq2 == 1)
            out = net(seq1.to(device),seq2.to(device), mask1.to(device), mask2.to(device))
            
            loss = loss_func(out, label.squeeze(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            prediction = out.argmax(dim=1).data.cpu().numpy().tolist()
            label = label.view(1,-1).squeeze().data.cpu().numpy().tolist()

            out_epoch.extend(prediction)
            label_epoch.extend(label)

            train_l += loss.item()
            n += 1

        train_acc = accuracy_score(label_epoch, out_epoch)

        dev_loss, dev_acc, max_acc = dev_evaluate(device, net, dev_iter, max_acc, ckp)
        test_loss, test_acc = test_evaluate(device, net, test_iter)
        end = datetime.datetime.now()

        print('epoch %d, train_acc %f, dev_acc %f, test_acc %f, max_acc %f, time %s'% (epoch+1, train_acc, dev_acc, test_acc, max_acc, end-start))