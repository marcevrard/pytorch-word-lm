###############################################################################
# Language Modeling on Penn Tree Bank
#
# This file generates new sentences sampled from the language model
#
###############################################################################

import argparse

import torch
# from torch.autograd import Variable

import data


def torch_device(args):
    torch.manual_seed(args.seed)

    # if torch.cuda.is_available():
    #     if not args.cuda:
    #         print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    return torch.device("cuda" if args.cuda else "cpu") # pylint: disable=no-member


def batchify(device, data_in, size):
    '''Work out how to cleanly divide the dataset into batches.'''
    batchs_n = data_in.size(0) // size
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data_cln = data_in.narrow(0, 0, batchs_n * size)
    # Evenly divide the data across the size batches.
    data_batches = data_cln.view(size, -1).t().contiguous()
    return data_batches.to(device)


def get_batch(bptt, data_batches, i):
    seq_len = min(bptt, len(data_batches) - 1 - i)
    data_seq = data_batches[i: i+seq_len]
    # target = source[i+1 : i+1+seq_len].view(-1)
    return data_seq


def generate(model, data_batches, args):
    model.eval()    # Turn on evaluation mode which disables dropout.
    hidden = model.init_hidden(args.batch_size)
    with torch.no_grad():   # no tracking history
        for i in range(0, data_batches.size(0) - 1, args.bptt):
            data_seq = get_batch(args.bptt, data_batches, i)
            output, hidden = model(data_seq, hidden)
            word_weights = output.squeeze().div(args.temperature).exp().cpu()
            print(output.shape)
            yield torch.multinomial(word_weights, 1)[0]  # pylint: disable=no-member


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 Language Model',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--data', type=str, default='./data/wikitext-2',
                        help="location of the data corpus")
    parser.add_argument('--checkpoint', type=str, default='./models/model.pt',
                        help="model checkpoint to use")
    parser.add_argument('--out-fpath', type=str, default='./output/generated.txt',
                        help="output file for generated text")
    parser.add_argument('--words-n', type=int, default='1000',
                        help="number of words to generate")
    parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                        help="batch size")
    parser.add_argument('--bptt', type=int, default=35,  # BPTT: Backprop. thru time?
                        help="sequence length")
    parser.add_argument('--seed', type=int, default=1111,
                        help="random seed")
    parser.add_argument('--cuda', action='store_true',
                        help="use CUDA")
    parser.add_argument('--temperature', type=float, default=1.0,
                        help="temperature - higher will increase diversity")
    parser.add_argument('--log-interval', type=int, default=100,
                        help="reporting interval")

    args = parser.parse_args()

    if args.temperature < 1e-3:
        parser.error("--temperature has to be >= 1e-3")

    return args


def main(args):

    device = torch_device(args)

    with open(args.checkpoint, 'rb') as f:
        model = torch.load(f).to(device)

    corpus = data.Corpus(args.data)
    input_txt = 'I love that I'.split()

    data_in = torch.tensor([corpus.dictionary.word2idx[w] for w in input_txt],   # pylint: disable=not-callable
                           dtype=torch.long).to(device)                          # pylint: disable=no-member

    data_batches = batchify(device, data_in, size=args.batch_size)
    print(input_txt)
    print(data_in)
    print('data_batches.shape:', data_batches.shape)
    print('dictionary size:', len(corpus.dictionary))

    for word_idx in generate(model, data_batches, args):
        print(corpus.dictionary.idx2word[word_idx])

    # with torch.no_grad():  # no tracking history
    #     with open(args.out_fpath, 'w') as f_out:
    #         for i in range(args.words_n):
    #             output, hidden = model(input_, hidden)
    #             word_weights = output.squeeze().div(args.temperature).exp().cpu()
    #             word_idx = torch.multinomial(word_weights, 1)[0]    # pylint: disable=no-member
    #             input_.fill_(word_idx)
    #             word = corpus.dictionary.idx2word[word_idx]

    #             f_out.write(word + ('\n' if i % 20 == 19 else ' '))

    #             i += 1  # Increase i by 1 for display
    #             if i % args.log_interval == 0 or i == args.words_n:
    #                 print('> Generated {}/{} words'.format(i, args.words_n))

if __name__ == '__main__':
    main(parse_args())
