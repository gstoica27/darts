import argparse
import os, sys, glob
import time
import math
import numpy as np
import torch
import logging
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from architect import Architect
from tacred_data import DataLoader
from tacred_utils import constant, vocab, scorer

import gc

import data
import model_search as model

from utils import batchify, get_batch, repackage_hidden, create_exp_dir, save_checkpoint

cwd = "/Volumes/External HDD/"
parser = argparse.ArgumentParser(description='PyTorch PennTreeBank/WikiText2 Language Model')
parser.add_argument('--data_dir', type=str, default='/home/scratch/gis/datasets/tacred/data/json',
                    help='location of the data corpus')
parser.add_argument('--vocab_dir', type=str, default='/home/scratch/gis/datasets/vocab')
parser.add_argument('--emsize', type=int, default=300,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=300,
                    help='number of hidden units per layer')
parser.add_argument('--nhidlast', type=int, default=300,
                    help='number of hidden units for the last rnn layer')
parser.add_argument('--lr', type=float, default=20,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=50,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=35,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.75,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--dropouth', type=float, default=0.25,
                    help='dropout for hidden nodes in rnn layers (0 = no dropout)')
parser.add_argument('--dropoutx', type=float, default=0.75,
                    help='dropout for input nodes in rnn layers (0 = no dropout)')
parser.add_argument('--dropouti', type=float, default=0.2,
                    help='dropout for input embedding layers (0 = no dropout)')
parser.add_argument('--dropoute', type=float, default=0,
                    help='dropout to remove words from embedding layer (0 = no dropout)')
parser.add_argument('--seed', type=int, default=3,
                    help='random seed')
parser.add_argument('--nonmono', type=int, default=5,
                    help='random seed')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str,  default='EXP',
                    help='path to save the final model')
parser.add_argument('--alpha', type=float, default=0,
                    help='alpha L2 regularization on RNN activation (alpha = 0 means no regularization)')
parser.add_argument('--beta', type=float, default=1e-3,
                    help='beta slowness regularization applied on RNN activiation (beta = 0 means no regularization)')
parser.add_argument('--wdecay', type=float, default=5e-7,
                    help='weight decay applied to all weights')
parser.add_argument('--continue_train', action='store_true',
                    help='continue train from a checkpoint')
parser.add_argument('--small_batch_size', type=int, default=-1,
                    help='the batch size for computation. batch_size should be divisible by small_batch_size.\
                     In our implementation, we compute gradients with small_batch_size multiple times, and accumulate the gradients\
                     until batch_size is reached. An update step is then performed.')
parser.add_argument('--max_seq_len_delta', type=int, default=20,
                    help='max sequence length')
parser.add_argument('--single_gpu', default=True, action='store_false', 
                    help='use single GPU')
parser.add_argument('--gpu', type=int, default=0, help='GPU device to use')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--arch_wdecay', type=float, default=1e-3,
                    help='weight decay for the architecture encoding alpha')
parser.add_argument('--arch_lr', type=float, default=3e-3,
                    help='learning rate for the architecture encoding alpha')
# TACRED Hyperparams
parser.add_argument('--ner_dim', type=int, default=30, help='NER embedding dimension.')
parser.add_argument('--pos_dim', type=int, default=30, help='POS embedding dimension.')
parser.add_argument('--lower', dest='lower', action='store_true', help='Lowercase all words.')
parser.add_argument('--no-lower', dest='lower', action='store_false')
parser.add_argument('--pe_dim', type=int, default=30, help='Position encoding dimension.')
parser.add_argument('--token_emb_path', type=str, default='/home/scratch/gis/datasets/vocab/embedding.npy')
parser.add_argument('--word_dropout', type=float, default=0.04, help='The rate at which randomly set a word to UNK.')
args = parser.parse_args()

if args.nhidlast < 0:
    args.nhidlast = args.emsize
if args.small_batch_size < 0:
    args.small_batch_size = args.batch_size

if not args.continue_train:
    args.save = '/home/scratch/gis/datasets/search-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
    create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

# Set the random seed manually for reproducibility.
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.set_device(args.gpu)
        cudnn.benchmark = True
        cudnn.enabled=True
        torch.cuda.manual_seed_all(args.seed)

# make opt
opt = vars(args)
opt['num_class'] = len(constant.LABEL_TO_ID)
args.num_class = len(constant.LABEL_TO_ID)

# load vocab
vocab_file = opt['vocab_dir'] + '/vocab.pkl'
vocab = vocab.Vocab(vocab_file, load=True)
opt['vocab_size'] = vocab.size
emb_file = opt['vocab_dir'] + '/embedding.npy'
emb_matrix = np.load(emb_file)
assert emb_matrix.shape[0] == vocab.size
assert emb_matrix.shape[1] == opt['emsize']


# corpus = data.Corpus(args.data)

# train_data = batchify(corpus.train, args.batch_size, args)
# search_data = batchify(corpus.valid, args.batch_size, args)
# val_data = batchify(corpus.valid, eval_batch_size, args)
# test_data = batchify(corpus.test, test_batch_size, args)

eval_batch_size = 10
test_batch_size = 1

# ntokens = len(corpus.dictionary)

ntokens = len(vocab.id2word)

if args.continue_train:
    model = torch.load(os.path.join(args.save, 'model.pt'))
else:
    model = model.RNNModelSearch(ntokens, args.emsize, args.nhid, args.nhidlast, 
                       args.dropout, args.dropouth, args.dropoutx, args.dropouti, args.dropoute,
                                 args.ner_dim, args.pos_dim, args.token_emb_path, len(constant.LABEL_TO_ID))

size = 0
for p in model.parameters():
    size += p.nelement()
logging.info('param size: {}'.format(size))
logging.info('initial genotype:')
logging.info(model.genotype())

if args.cuda:
    if args.single_gpu:
        parallel_model = model.cuda()
    else:
        parallel_model = nn.DataParallel(model, dim=1).cuda()
else:
    parallel_model = model
architect = Architect(parallel_model, args)

total_params = sum(x.data.nelement() for x in model.parameters())
logging.info('Args: {}'.format(args))
logging.info('Model total parameters: {}'.format(total_params))

id2label = dict([(v,k) for k,v in constant.LABEL_TO_ID.items()])

def evaluate(data_source, batch_size=10, data_name='dev'):
    print('Evaluating Model!')
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0
    # ntokens = len(corpus.dictionary)
    # ntokens = len(vocab.word2id)
    # for i in range(0, data_source.size(0) - 1, args.bptt):
    predictions = []
    for i in range(len(data_source)):
        batch = data_source.next_batch()
        batch_size = len(batch['relation'])
        hidden = model.init_hidden(batch_size)[0]
        # data, targets = get_batch(data_source, i, args, evaluation=True)
        data = batch
        targets = batch['relation']
        targets = targets.view(-1)
        print('tokens: {} | hidden: {}'.format(batch['tokens'].shape, hidden.shape))
        log_prob, hidden = parallel_model(data, hidden)
        loss = nn.functional.nll_loss(log_prob, targets).data  # log_prob.view(-1, log_prob.size(2))

        total_loss += loss * len(data)

        batch_predictions = torch.argmax(log_prob, dim=-1).cpu().data.numpy()
        batch_predictions = [id2label[prediction] for prediction in batch_predictions]
        predictions += batch_predictions

        # hidden = repackage_hidden(hidden)

    precision, recall, f1 = scorer.score(dev_data.gold(), predictions)
    logging.info('{} set | Precision: {} | Recall: {} | F1: {}'.format(
        data_name, precision, recall, f1
    ))
    return total_loss[0] / len(data_source)


def train(train_data, dev_data):

    assert args.batch_size % args.small_batch_size == 0, 'batch_size must be divisible by small_batch_size'
    ntokens = len(vocab.word2id)
    # Turn on training mode which enables dropout.
    total_loss = 0
    start_time = time.time()
    # ntokens = len(corpus.dictionary)

    # batch, i = 0, 0
    for batch in range(len(train_data)):
        train_batch = train_data.next_batch()
        dev_batch = dev_data.next_batch()
        # for batch, (train_batch, dev_batch) in enumerate(zip(train_data, dev_data)):
        # hidden = [model.init_hidden(args.small_batch_size) for _ in range(args.batch_size // args.small_batch_size)]
        # hidden_valid = [model.init_hidden(args.small_batch_size) for _ in
        #                 range(args.batch_size // args.small_batch_size)]

        
        #print('hidden shape: {} | hidden valid: {} |'.format(hidden.shape, hidden_valid.shape))
        # while i < train_data.size(0) - 1 - 1:
        bptt = args.bptt if np.random.random() < 0.95 else args.bptt / 2.
        # Prevent excessively small or negative sequence lengths
        # seq_len = max(5, int(np.random.normal(bptt, 5)))
        # # There's a very small chance that it could select a very long sequence length resulting in OOM
        # seq_len = min(seq_len, args.bptt + args.max_seq_len_delta)
        # seq_len = int(bptt)

        lr2 = optimizer.param_groups[0]['lr']
        optimizer.param_groups[0]['lr'] = lr2  #* seq_len / args.bptt
        model.train()

        # data_valid, targets_valid = get_batch(search_data, i % (search_data.size(0) - 1), args)
        # data, targets = get_batch(train_data, i, args, seq_len=seq_len)

        optimizer.zero_grad()

        # start, end, s_id = 0, args.small_batch_size, 0
        cur_data = train_batch
        cur_targets = train_batch['relation']

        cur_data_valid = dev_batch
        cur_targets_valid = dev_batch['relation']

        hidden = model.init_hidden(len(train_batch['relation']))[0]
        hidden_valid = model.init_hidden(len(dev_batch['relation']))[0]
        # print('Train Batch Shapes: | Hidden: {} | Tokens: {} |'.format(hidden.shape, cur_data['tokens'].shape))
        # print('Dev Batch Shapes: | Hidden: {} | Tokens: {} |'.format(hidden_valid.shape, cur_data_valid['tokens'].shape))
        assert hidden.shape[1] == cur_data['tokens'].shape[0], 'Hidden shape: {} | tokens shape: {}'.format(
            hidden.shape, cur_data['tokens'].shape
        )
        assert hidden_valid.shape[1] == cur_data_valid['tokens'].shape[0], 'Hidden shape: {} | tokens shape: {}'.format(
            hidden_valid.shape, cur_data_valid['tokens'].shape
        )

        # while start < args.batch_size:
        #     cur_data, cur_targets = data[:, start: end], targets[:, start: end].contiguous().view(-1)
        #     cur_data_valid, cur_targets_valid = data_valid[:, start: end], targets_valid[:, start: end].contiguous().view(-1)

        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        # hidden[s_id] = repackage_hidden(hidden[s_id])
        # hidden_valid[s_id] = repackage_hidden(hidden_valid[s_id])
        #print(hidden.shape)
        #hidden = repackage_hidden(hidden)
        #hidden_valid = repackage_hidden(hidden_valid)

        # hidden_valid[s_id], grad_norm = architect.step(
        #         hidden[s_id], cur_data, cur_targets,
        #         hidden_valid[s_id], cur_data_valid, cur_targets_valid,
        #         optimizer,
        #         args.unrolled)
        hidden_valid, grad_norm = architect.step(
            hidden, cur_data, cur_targets,
            hidden_valid, cur_data_valid, cur_targets_valid,
            optimizer,
            args.unrolled)
        # print('Finished architect step...')
        # assuming small_batch_size = batch_size so we don't accumulate gradients
        optimizer.zero_grad()
        # hidden[s_id] = repackage_hidden(hidden[s_id])
        #hidden = repackage_hidden(hidden)

        # log_prob, hidden[s_id], rnn_hs, dropped_rnn_hs = parallel_model(cur_data, hidden[s_id], return_h=True)
        # print('Entering model training...')
        hidden = torch.autograd.Variable(hidden.data)
        log_prob, hidden, rnn_hs, dropped_rnn_hs = parallel_model(cur_data, hidden, return_h=True)
        # print('received predictions')
        raw_loss = nn.functional.nll_loss(log_prob, cur_targets)
        # print('received loss' )

        loss = raw_loss
        # Activiation Regularization
        if args.alpha > 0:
            loss = loss + sum(args.alpha * dropped_rnn_h.pow(2).mean() for dropped_rnn_h in dropped_rnn_hs[-1:])
        # Temporal Activation Regularization (slowness)
        loss = loss + sum(args.beta * (rnn_h[1:] - rnn_h[:-1]).pow(2).mean() for rnn_h in rnn_hs[-1:])
        # loss *= args.small_batch_size / args.batch_size
        total_loss += raw_loss.data  # * args.small_batch_size / args.batch_size
        loss.backward()

        # s_id += 1
        # start = end
        # end = start + args.small_batch_size
        # print('backpropogated...')
        gc.collect()
        # print('garbage collected...')

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs.
        torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
        # print('clipped gradients...')
        optimizer.step()
        # print('updated gradients...')
        # total_loss += raw_loss.data
        optimizer.param_groups[0]['lr'] = lr2
        if batch % args.log_interval == 0: # and batch > 0:
            logging.info(parallel_model.genotype())
            print(F.softmax(parallel_model.weights, dim=-1))
            #print('total loss: {}'.format(type(total_loss)))
            #print('total loss: {}'.format(total_loss))
            #print('total loss: {}'.format(total_loss.shape))
            #cur_loss = total_loss[0] / args.log_interval
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            logging.info('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_data), optimizer.param_groups[0]['lr'],
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()
        # print('on to next batch...')
        # batch += 1
        # i += seq_len
    print('Reached end of epoch training!')

# Loop over epochs.
lr = args.lr
best_val_loss = []
stored_loss = 100000000

if args.continue_train:
    optimizer_state = torch.load(os.path.join(args.save, 'optimizer.pt'))
    if 't0' in optimizer_state['param_groups'][0]:
        optimizer = torch.optim.ASGD(model.parameters(), lr=args.lr, t0=0, lambd=0., weight_decay=args.wdecay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wdecay)
    optimizer.load_state_dict(optimizer_state)
else:
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wdecay)

train_data = DataLoader(args.data_dir + '/train.json', args.batch_size, opt, vocab, evaluation=False)
dev_data = DataLoader(args.data_dir + '/dev.json', args.batch_size, opt, vocab, evaluation=True)
test_data = DataLoader(args.data_dir + '/test.json', args.batch_size, opt, vocab, evaluation=True)

for epoch in range(1, args.epochs+1):
    epoch_start_time = time.time()
    train(train_data, dev_data)

    val_loss = evaluate(dev_data, eval_batch_size, data_name='dev')
    logging.info('-' * 89)
    logging.info('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
            'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                       val_loss, math.exp(val_loss)))
    logging.info('-' * 89)

    if val_loss < stored_loss:
        save_checkpoint(model, optimizer, epoch, args.save)
        logging.info('Saving Normal!')
        stored_loss = val_loss

    best_val_loss.append(val_loss)
