import argparse


def parser():
    parser = argparse.ArgumentParser(description='Part-of-Speech Tagging using a LSTM Deep Learning Neural Network.')

    # Required arguments
    parser.add_argument('train', type=str, help='Train input file path')
    parser.add_argument('validation', type=str, help='Validation input file path')
    parser.add_argument('test', type=str, help='Test input file path')

    return parser.parse_args()
