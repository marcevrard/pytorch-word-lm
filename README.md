# Word-level language modeling RNN

*Project forked from <https://github.com/pytorch/examples/tree/master/word_language_model>.*

This example trains a multi-layer RNN (Elman, GRU, or LSTM) on a language modeling task.
By default, the training script uses the `Wikitext-2` dataset, provided.
The trained model can then be used by the generate script to generate new text.

```bash
python main.py --cuda --epochs 6        # Train a LSTM on Wikitext-2 with CUDA, reaching perplexity of 117.61
python main.py --cuda --epochs 6 --tied # Train a tied LSTM on Wikitext-2 with CUDA, reaching perplexity of 110.44
python main.py --cuda --tied            # Train a tied LSTM on Wikitext-2 with CUDA for 40 epochs, reaching perplexity of 87.17
python generate.py                      # Generate samples from the trained LSTM model.
```

The model uses the `nn.RNN` module (and its sister modules `nn.GRU` and `nn.LSTM`)
which will automatically use the cuDNN backend if run on CUDA with cuDNN installed.

During training, if a keyboard interrupt (Ctrl-C) is received,
training is stopped and the current model is evaluated against the test dataset.

The `main.py` script documentation:

```bash
usage: main.py [-h] [--data DATA] [--model MODEL] [--emsize EMSIZE]
               [--nhid NHID] [--nlayers NLAYERS] [--lr LR] [--clip CLIP]
               [--epochs EPOCHS] [--batch_size N] [--bptt BPTT]
               [--dropout DROPOUT] [--tied] [--seed SEED] [--cuda]
               [--log-interval N] [--save SAVE] [--onnx-export ONNX_EXPORT]

PyTorch Wikitext-2 RNN/LSTM Language Model

optional arguments:
  -h, --help            show this help message and exit
  --data DATA           location of the data corpus (default:
                        ./data/wikitext-2)
  --model MODEL         type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)
                        (default: LSTM)
  --emsize EMSIZE       size of word embeddings (default: 200)
  --nhid NHID           number of hidden units per layer (default: 200)
  --nlayers NLAYERS     number of layers (default: 2)
  --lr LR               initial learning rate (default: 20)
  --clip CLIP           gradient clipping (default: 0.25)
  --epochs EPOCHS       upper epoch limit (default: 40)
  --batch_size N        batch size (default: 20)
  --bptt BPTT           sequence length (default: 35)
  --dropout DROPOUT     dropout applied to layers (0 = no dropout) (default:
                        0.2)
  --tied                tie the word embedding and softmax weights (default:
                        False)
  --seed SEED           random seed (default: 1111)
  --cuda                use CUDA (default: False)
  --log-interval N      report interval (default: 200)
  --save SAVE           path to save the final model (default: models/model.pt)
  --onnx-export ONNX_EXPORT
                        path to export the final model in onnx format
                        (default: )
```

With these arguments, a variety of models can be tested.
As an example, the following arguments produce slower but better models:

```bash
python main.py --cuda --emsize 650 --nhid 650 --dropout 0.5 --epochs 40           # Test perplexity of 80.97
python main.py --cuda --emsize 650 --nhid 650 --dropout 0.5 --epochs 40 --tied    # Test perplexity of 75.96
python main.py --cuda --emsize 1500 --nhid 1500 --dropout 0.65 --epochs 40        # Test perplexity of 77.42
python main.py --cuda --emsize 1500 --nhid 1500 --dropout 0.65 --epochs 40 --tied # Test perplexity of 72.30
```

Perplexities on PTB are equal or better than
[Recurrent Neural Network Regularization (Zaremba et al. 2014)](https://arxiv.org/pdf/1409.2329.pdf)
and are similar to [Using the Output Embedding to Improve Language Models (Press & Wolf 2016](https://arxiv.org/abs/1608.05859) and [Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling (Inan et al. 2016)](https://arxiv.org/pdf/1611.01462.pdf), though both of these papers have improved perplexities by using a form of recurrent dropout [(variational dropout)](http://papers.nips.cc/paper/6241-a-theoretically-grounded-application-of-dropout-in-recurrent-neural-networks).
