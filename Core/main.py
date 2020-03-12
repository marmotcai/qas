import argparse

from data import *
from model import *

import sys
sys.path.append("..")


default_data_dir = "../data/"
default_code = "300096"

def parse_args():
    """Parse arguments."""
    # Parameters settings
    parser = argparse.ArgumentParser(prog="python main.py", description="-- QAS 量化分析内核 --", epilog="---------------------")

    parser.add_argument('-d', '--download', action='store_true', default=False, help='下载数据')

    parser.add_argument('-t', '--train', action='store_true', default=False, help='训练数据')

    # Dataset setting
    parser.add_argument('--dir', type=str, default=default_data_dir, help='下载数据存放路径')
    parser.add_argument('--code', type=str, default=default_code, help='数据代码')

    parser.add_argument('--dataroot', type=str, default=default_data_dir + "default.csv", help='path to dataset')
    parser.add_argument('--batchsize', type=int, default=128, help='input batch size [128]')

    # Encoder / Decoder parameters setting
    parser.add_argument('--nhidden_encoder', type=int, default=128, help='size of hidden states for the encoder m [64, 128]')
    parser.add_argument('--nhidden_decoder', type=int, default=128, help='size of hidden states for the decoder p [64, 128]')
    parser.add_argument('--ntimestep', type=int, default=10, help='the number of time steps in the window T [10]')

    # Training parameters setting
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train [10, 200, 500]')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate [0.001] reduced by 0.1 after each 10000 iterations')

    # parse the arguments
    args = parser.parse_args()

    return args


def main():
    print("==> qas core start ...")

    args = parse_args()
    print(args)

    if args.download:
        down_obj = download(args.dir)
        down_obj.download_by_code(args.code)
        # down_data(args.dir, args.code)
        # args.dataroot = default_data_dir + default_code + ".csv"

    if args.train:
        print("==> Load dataset: ", args.dataroot, "..")

        down_obj = download(args.dir)
        X, y = down_obj.read_by_file(default_data_dir + args.dataroot, debug = False)

        # X, y = read_data(default_data_dir + args.dataroot, debug = False)

        # Initialize model
        print("==> Initialize DA-RNN model ...")
        model = DA_RNN(
            X,
            y,
            args.ntimestep,
            args.nhidden_encoder,
            args.nhidden_decoder,
            args.batchsize,
            args.lr,
            args.epochs
        )
        print(model)

        # Train
        print("==> Start training ...")
        model.train()

    print("==> Close  ...")

if __name__ == '__main__':
    main()