import argparse

def parser():
    parser = argparse.ArgumentParser(description='Part-of-Speech Tagging using a LSTM Deep Learning Neural Network.')

    # Required arguments
    parser.add_argument('train', type=str, help='Train input file path')
    parser.add_argument('validation', type=str, help='Validation input file path')
    parser.add_argument('test', type=str, help='Test input file path')

    # Optional arguments
    parser.add_argument('-d', '--data_preproc', type=int, default=False, help='Do preprocessing of corpus? (Default: False)')

    args = parser.parse_args()

    return args
