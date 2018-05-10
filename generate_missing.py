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

    return torch.device("cuda" if args.cuda else "cpu")   # pylint: disable=no-member


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 Language Model',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--data', type=str, default='./data/wikitext-2',
                        help='location of the data corpus')
    parser.add_argument('--checkpoint', type=str, default='./models/model.pt',
                        help='model checkpoint to use')
    parser.add_argument('--out_fpath', type=str, default='./output/generated.txt',
                        help='output file for generated text')
    parser.add_argument('--words', type=int, default='1000',
                        help='number of words to generate')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--cuda', action='store_true',
                        help='use CUDA')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='temperature - higher will increase diversity')
    parser.add_argument('--log-interval', type=int, default=100,
                        help='reporting interval')

    args = parser.parse_args()

    if args.temperature < 1e-3:
        parser.error("--temperature has to be >= 1e-3")

    return args


def main(args):

    device = torch_device(args)

    with open(args.checkpoint, 'rb') as f:
        model = torch.load(f).to(device)

    model.eval()

    corpus = data.Corpus(args.data)
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(1)
    input_ = torch.randint(ntokens, (1, 1), dtype=torch.long).to(device)    # pylint: disable=no-member

    with open(args.out_fpath, 'w') as f_out:
        with torch.no_grad():  # no tracking history
            for i in range(args.words):
                output, hidden = model(input_, hidden)
                word_weights = output.squeeze().div(args.temperature).exp().cpu()
                word_idx = torch.multinomial(word_weights, 1)[0]    # pylint: disable=no-member
                input_.fill_(word_idx)
                word = corpus.dictionary.idx2word[word_idx]

                f_out.write(word + ('\n' if i % 20 == 19 else ' '))

                if i % args.log_interval == 0:
                    print('| Generated {}/{} words'.format(i, args.words))

if __name__ == '__main__':
    main(parse_args())
