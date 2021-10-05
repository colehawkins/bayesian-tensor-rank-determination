from functools import reduce
import argparse
import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import tensor_layers
from torchtext.legacy import data
from torchtext.legacy import datasets
import torch.optim as optim
from models import LSTM_Classifier
from utils import train, evaluate
import random
import math
import sys
import os
import time
sys.path.insert(0, '..')


parser = argparse.ArgumentParser()
parser.add_argument(
    '--embedding',
    default='full',
    choices=['CP', 'TensorTrain', 'TensorTrainMatrix','Tucker','full'],
    type=str)
parser.add_argument('--rank-loss', type=bool, default=False)
parser.add_argument('--kl-multiplier', type=float, default=1.0) #account for the batch size,dataset size, and renormalize
parser.add_argument('--no-kl-epochs', type=int, default=20)
parser.add_argument('--warmup-epochs', type=int, default=50)
parser.add_argument('--rank', type=int, default=8)
parser.add_argument('--prior-type', type=str, default='log_uniform')
parser.add_argument('--eta', type=float, default=1.0)
parser.add_argument('--embed-dim', default=256, type=int)
parser.add_argument('--voc_dim', default=25000, type=int)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--hidden-dim', default=128, type=int)
parser.add_argument('--n_epochs',  default=100, type=int)
parser.add_argument('--batch-size',  default=256, type=int)
parser.add_argument('--dropout', default=0.5, type=float)
args = parser.parse_args()
model_name = args.embedding

BATCH_SIZE = args.batch_size

if args.embedding=='full':
    pass
else:
    if args.embedding=="TensorTrainMatrix":
        tensor_dims = [[5,5,5,5,6,8],[2,2,2,2,4,4]]
        args.voc_dim = reduce(lambda x,y:x*y,tensor_dims[0])
    elif args.embedding=="Tucker":
        tensor_dims = [[25,25,40],[16,16]]
        args.voc_dim = 25*25*40
    elif args.embedding in ["CP","TensorTrain"]:
        tensor_dims = [[5,8,25,25],[4,8,8]]
        args.voc_dim = 5*5*5*5*5*8

    target_stddev = math.sqrt(2/(args.voc_dim+256))

print(args)


random.seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


TEXT = data.Field(tokenize='spacy', fix_length=1000)
LABEL = data.LabelField(dtype=torch.float)

print('Building dataset...')
OUTPUT_DIM = 1
train_data, test_ = datasets.IMDB.splits(TEXT, LABEL)
test_list = list(test_)
random.shuffle(test_list)
test_data_ = test_list[:12500]
val_data_ = test_list[12500:]
valid_data = data.dataset.Dataset(
    val_data_, fields=[('text', TEXT), ('label', LABEL)])
test_data = data.dataset.Dataset(
    test_data_, fields=[('text', TEXT), ('label', LABEL)])

print('Done')


def sort_key(ex):
    return len(ex.text)

TEXT.build_vocab(train_data, max_size=args.voc_dim - 2)
LABEL.build_vocab(train_data)


train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size=BATCH_SIZE,
    device=device)

valid_iterator.sort_key = sort_key
test_iterator.sort_key = sort_key

INPUT_DIM = len(TEXT.vocab)
EMBEDDING_DIM = args.embed_dim
HIDDEN_DIM = args.hidden_dim
N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = args.dropout



lstm_model = LSTM_Classifier(embedding_dim=EMBEDDING_DIM,
                             hidden_dim=HIDDEN_DIM,
                             output_dim=OUTPUT_DIM,
                             n_layers=N_LAYERS,
                             bidirectional=BIDIRECTIONAL,
                             dropout=DROPOUT)

if args.embedding == 'full':
    embed_model = nn.Embedding(
        num_embeddings=INPUT_DIM,
        embedding_dim=EMBEDDING_DIM
    )
    compression_rate = 1.0
else:
    embed_model = tensor_layers.TensorizedEmbedding(
        shape=tensor_dims,
        tensor_type=args.embedding,
        max_rank=args.rank,
        prior_type=args.prior_type,
        eta=args.eta,
        em_stepsize=0.1,
    )
    compression_rate = 1e10


def cross_entropy_loss(logits, target):
    labels = target.type(torch.LongTensor).to(logits.device)
    return nn.CrossEntropyLoss()(logits, labels)


model = nn.Sequential(embed_model, lstm_model)
"""
from utils import get_kl_loss
for fake_epoch in range(100):
    print("Epoch ",fake_epoch)
    get_kl_loss(model,args,fake_epoch)
"""

n_all_param = sum([p.nelement() for p in model.parameters()])

criterion = nn.BCEWithLogitsLoss()

model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=args.lr)
print(model)
N_EPOCHS = args.n_epochs

log = {
    'compression_rate':compression_rate,
    'train_loss':[], 'test_loss':[], 'valid_loss':[],
    'train_acc':[], 'test_acc':[], 'valid_acc':[]}
best_result = {
    "epoch": 0, "train_acc": 0, "valid_acc": 0, "train_acc": 0}

for epoch in range(N_EPOCHS):
    t = time.time()
    train_loss, train_acc = train(model, train_iterator, optimizer, criterion, args, epoch)
    test_loss, test_acc = evaluate(model, test_iterator, criterion)
    valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)

    log['train_loss'].append(train_loss)
    log['test_loss'].append(test_loss)
    log['train_acc'].append(train_acc)
    log['test_acc'].append(test_acc)
    log['valid_acc'].append(valid_acc)
    log['valid_loss'].append(valid_loss)

    if best_result["valid_acc"] < valid_acc:
        best_result["epoch"] = epoch
        best_result["train_acc"] = train_acc
        best_result["valid_acc"] = valid_acc
        best_result["test_acc"] = test_acc

    print(f'| Epoch: {epoch+1:02} | Train Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}% | Val. Loss: {valid_loss:.3f} | Val. Acc: {valid_acc*100:.2f}% | Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}% |')
    print ("TEST ACCURACY:", np.round(best_result["test_acc"] * 100, 2))
    print ("Epoch time :",time.time()-t)

    #terminate low-performing run early
    if epoch>40 and valid_acc<0.6:
        break

    if hasattr(embed_model,'tensor'):
        print("Ranks ",embed_model.tensor.estimate_rank(threshold=1e-8))
        param_savings = embed_model.tensor.get_parameter_savings(threshold=1e-8)
        full_params = 25000*256
        print(param_savings)
        print("Savings {} ratio {}".format(param_savings,full_params/(full_params-sum(param_savings))))

    """
    if epoch == 0 or epoch == N_EPOCHS-1:
        print('Compression rate:', compression_rate)        print('#params = {}'.format(n_all_param))
    """
