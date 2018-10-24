# POS Tagging using LSTM

Part-of-Speech Tagging using a LSTM Deep Learning Neural Network

## Usage

```bash
python3 posTagging.py -h
```

```text
usage: posTagging.py [-h] [-d DATA_PREPROC] train validation test

Part-of-Speech Tagging using a LSTM Deep Learning Neural Network.

positional arguments:
  train                 Train input file path
  validation            Validation input file path
  test                  Test input file path

optional arguments:
  -h, --help            show this help message and exit
  -d DATA_PREPROC, --data_preproc DATA_PREPROC
                        Do preprocessing of corpus? (0 = False / 1 = True)
                        (Default: 1)
```
