import torch
import argparse
import data_proc
import train

parser = argparse.ArgumentParser()
parser.add_argument('--cuda', type=str, default='cuda:6')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--num_epoch', type=int, default=50)
parser.add_argument('--lr', type=float, default=0.0004)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--ckp', type=str, default='ckp/model_3.pt')
parser.add_argument('--max_acc', type=float, default=0.5)
args = parser.parse_args()

if torch.cuda.is_available():
    print("using cuda......")
    device = torch.device(args.cuda)

w2v_model, train_iter, dev_iter, test_iter = data_proc.data(device, args.batch_size)
train.training(device, w2v_model, train_iter, dev_iter, test_iter, args.batch_size, args.num_epoch, args.lr, args.weight_decay, args.ckp, args.max_acc)
