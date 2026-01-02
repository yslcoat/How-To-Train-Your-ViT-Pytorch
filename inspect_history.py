import argparse
import pandas as pd
import matplotlib.pyplot as plt

import torch


def main(args):
    print("Loading checkpoint")
    checkpoint = torch.load(args.training_artifact, weights_only=False)
    print("loading compelte")
    # batch_history = checkpoint["batch_history"]
    epoch_history = checkpoint["epoch_history"]

    df = pd.DataFrame(epoch_history)
    train_history = df[df["mode"] == "train"]
    val_history = df[df["mode"] == "validate"]

    plt.plot(train_history["loss"].to_list(), label="Train") # to_list() to get correct values on the x-axis
    plt.plot(val_history["loss"].to_list(), label="Val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curves")
    plt.legend()
    plt.grid(True) 
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-ta",
        "--training_artifact", 
        required=True, 
        help="Path to artifact created by train.py, f.ex checkpoint.pth.tar "
    )
    args = parser.parse_args()

    main(args)
