# coding: utf-8
import argparse
import math
import os
import time

import torch
import torch.nn as nn
import torch.onnx

import data
import model


SEP_THIN = '-' * 89
SEP_THICK = '=' * 89

parser = argparse.ArgumentParser(description="PyTorch Wikitext-2 RNN/LSTM Language Model",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data', type=str, default='./data/wikitext-2',
                    help="location of the data corpus")
parser.add_argument('--model', type=str, default='LSTM',
                    help="type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)")
parser.add_argument('--emsize', type=int, default=200,
                    help="word embeddings' dimension")
parser.add_argument('--nhid', type=int, default=200,
                    help="number of hidden units per layer")
parser.add_argument('--nlayers', type=int, default=2,
                    help="number of layers")
parser.add_argument('--lr', type=float, default=20,
                    help="initial learning rate")
parser.add_argument('--clip', type=float, default=0.25,
                    help="gradient clipping")
parser.add_argument('--epochs', type=int, default=40,
                    help="upper epoch limit")
parser.add_argument('--batch-size', type=int, default=20, metavar='N',
                    help="batch size")
parser.add_argument('--bptt', type=int, default=35,  # BPTT: Backprop. thru time?
                    help="sequence length")
parser.add_argument('--dropout', type=float, default=0.2,
                    help="dropout applied to layers (0 = no dropout)")
parser.add_argument('--tied', action='store_true',
                    help="tie the word embedding and softmax weights")
parser.add_argument('--seed', type=int, default=1111,
                    help="random seed")
parser.add_argument('--cuda', action='store_true',
                    help="use CUDA")
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help="report interval")
parser.add_argument('--save', type=str, default='./models/model.pt',
                    help="path to save the final model")
parser.add_argument('--onnx-export', type=str, default='',
                    help="path to export the final model in onnx format")
args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

device = torch.device("cuda" if args.cuda else "cpu")   # pylint: disable=no-member

###############################################################################
# Load data
###############################################################################

corpus = data.Corpus(args.data)

# Starting from sequential data, batchify arranges the dataset into columns.
# For instance, with the alphabet as the sequence and batch size 4, we'd get
# ┌ a g m s ┐
# │ b h n t │
# │ c i o u │
# │ d j p v │
# │ e k q w │
# └ f l r x ┘.
# These columns are treated as independent by the model, which means that the
# dependence of e. g. 'g' on 'f' can not be learned, but allows more efficient
# batch processing.

def batchify(data_in, size):
    '''Work out how to cleanly divide the dataset into batches.'''
    batchs_n = data_in.size(0) // size
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data_cln = data_in.narrow(0, 0, batchs_n * size)
    # Evenly divide the data across the size batches.
    data_batches = data_cln.view(size, -1).t().contiguous()
    return data_batches.to(device)

eval_batch_size = 10
train_data = batchify(corpus.train, args.batch_size)
val_data = batchify(corpus.valid, eval_batch_size)
test_data = batchify(corpus.test, eval_batch_size)

# print(corpus.valid.shape)
# print(val_data.shape)

###############################################################################
# Build the model
###############################################################################

model = model.RNNModel(rnn_type=args.model, ntoken=len(corpus.dictionary),
                       ninp=args.emsize, nhid=args.nhid, nlayers=args.nlayers,
                       dropout=args.dropout, tie_weights=args.tied).to(device)

# AttributeError: 'DataParallel' object has no attribute 'init_hidden'
# model = torch.nn.DataParallel(model)  # **MEV**

criterion = nn.CrossEntropyLoss()

###############################################################################
# Training code
###############################################################################

def repackage_hidden(h):
    '''Wraps hidden states in new Tensors, to detach them from their history.'''
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


# get_batch subdivides the source data into chunks of length args.bptt.
# If source is equal to the example output of the batchify function, with
# a bptt-limit of 2, we'd get the following two Variables for i = 0:
# ┌ a g m s ┐ ┌ b h n t ┐
# └ b h n t ┘ └ c i o u ┘
# Note that despite the name of the function, the subdivision of data is not
# done along the batch dimension (i.e., dimension 1), since that was handled
# by the batchify function. The chunks are along dimension 0, corresponding
# to the seq_len dimension in the LSTM.

def get_batch(data_batches, i):
    seq_len = min(args.bptt, len(data_batches) - 1 - i)
    data_seq = data_batches[i: i+seq_len]
    target = data_batches[i+1 : i+1+seq_len].view(-1)
    return data_seq, target


def evaluate(data_batches):
    model.eval()    # Turn on evaluation mode, which disables dropout.
    total_loss = 0.
    hidden = model.init_hidden(eval_batch_size)
    with torch.no_grad():   # no tracking history
        for i in range(0, data_batches.size(0) - 1, args.bptt):
            data_seq, targets = get_batch(data_batches, i)
            output, hidden = model(data_seq, hidden)
            output_flat = output.view(-1, len(corpus.dictionary))
            total_loss += len(data_seq) * criterion(output_flat, targets).item()
            hidden = repackage_hidden(hidden)
    return total_loss / len(data_batches)


def train():
    model.train()  # Turn on training mode, which enables dropout.
    total_loss = 0.
    start_time = time.time()
    hidden = model.init_hidden(args.batch_size)
    for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
        data_seq, targets = get_batch(train_data, i)
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        hidden = repackage_hidden(hidden)
        model.zero_grad()
        output, hidden = model(data_seq, hidden)
        loss = criterion(output.view(-1, len(corpus.dictionary)), targets)
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        for p in model.parameters():
            p.data.add_(-lr, p.grad.data)

        total_loss += loss.item()

        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            print(f'| epoch {epoch:3d} | {batch:5d}/{len(train_data) // args.bptt:5d} batches '
                  f'| lr {lr:02.2f} | ms/batch {elapsed * 1000 / args.log_interval:5.2f} '
                  f'| loss {cur_loss:5.2f} | ppl {math.exp(cur_loss):8.2f}')
            total_loss = 0
            start_time = time.time()


def export_onnx(path, batch_size, seq_len):
    print('The model is also exported in ONNX format at {}'.
          format(os.path.realpath(args.onnx_export)))
    model.eval()
    dummy_input = (torch.LongTensor(seq_len * batch_size)           # pylint: disable=no-member
                        .zero_().view(-1, batch_size).to(device))   # pylint: disable=bad-continuation
    hidden = model.init_hidden(batch_size)
    torch.onnx.export(model, (dummy_input, hidden), path)


# Loop over epochs.
lr = args.lr
best_val_loss = None

# At any point you can hit Ctrl + C to break out of training early.
try:
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        train()
        val_loss = evaluate(val_data)
        print(SEP_THIN)
        print(f'| end of epoch {epoch:3d} | time: {time.time() - epoch_start_time:5.2f}s'
              f'| valid loss {val_loss:5.2f} | valid ppl {math.exp(val_loss):8.2f}')
        print(SEP_THIN)
        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            with open(args.save, 'wb') as f:
                torch.save(model, f)
            best_val_loss = val_loss
        else:
            # Anneal the learning rate if no improvement has been seen in the validation dataset.
            lr /= 4.0
except KeyboardInterrupt:
    print(SEP_THIN)
    print('Exiting from training early')

# Load the best saved model.
with open(args.save, 'rb') as f:
    model = torch.load(f)
    # after load the rnn params are not a continuous chunk of memory
    # this makes them a continuous chunk, and will speed up forward pass
    model.rnn.flatten_parameters()

# Run on test data.
test_loss = evaluate(test_data)
print(SEP_THICK)
print(f'| End of training | test loss {test_loss:5.2f} | test ppl {math.exp(test_loss):8.2f}')
print(SEP_THICK)

if args.onnx_export:
    # Export the model in ONNX format.
    export_onnx(args.onnx_export, batch_size=1, seq_len=args.bptt)
