#
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>,
# Apoorv Vyas <avyas@idiap.ch>
#

import argparse
from itertools import chain
import math
import time

import torch
import numpy as np
import matplotlib.pyplot as plt
from imageio import imwrite

from fast_transformers.builders import RecurrentEncoderBuilder

from main import ImageGenerator
from image_datasets import add_dataset_arguments, get_dataset
from utils import load_model, sample_mol, load_model_pytorch, \
    add_transformer_arguments, print_transformer_arguments


class Timer(object):
    def __init__(self, force_cpu=False):
        self.force_cpu = force_cpu
        if not self.force_cpu and torch.cuda.is_available():
            self._start = torch.cuda.Event(enable_timing=True)
            self._end = torch.cuda.Event(enable_timing=True)
            self._start.record()
        else:
            self._start = time.time()

    def measure(self):
        if not self.force_cpu and torch.cuda.is_available():
            self._end.record()
            torch.cuda.synchronize()
            return self._start.elapsed_time(self._end)/1000
        else:
            return time.time()-self._start


class RecurrentGenerator(torch.nn.Module):
    class PositionalEncoding(torch.nn.Module):
        def __init__(self, d_model, dropout=0.0, max_len=5000):
            super(RecurrentGenerator.PositionalEncoding, self).__init__()
            self.dropout = torch.nn.Dropout(p=dropout)
            self.d_model = d_model
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)
            self.register_buffer('pe', pe)

        def forward(self, x, i):
            pos_embedding =  self.pe[0, i:i+1]
            x = torch.cat(
                [x, pos_embedding.expand_as(x)],
                dim=1
            )
            return self.dropout(x)

    def __init__(self, d_model, sequence_length, mixtures,
                 attention_type="full", n_layers=4, n_heads=4,
                 d_query=32, dropout=0.1, softmax_temp=None,
                 attention_dropout=0.1,
                 bits=32, rounds=4,
                 chunk_size=32, masked=True):
        super(RecurrentGenerator, self).__init__()

        self.pos_embedding = self.PositionalEncoding(
            d_model//2,
            max_len=sequence_length
        )
        self.value_embedding = torch.nn.Embedding(
            256,
            d_model//2
        )
        self.transformer = RecurrentEncoderBuilder.from_kwargs(
            attention_type=attention_type,
            n_layers=n_layers,
            n_heads=n_heads,
            feed_forward_dimensions=n_heads*d_query*4,
            query_dimensions=d_query,
            value_dimensions=d_query,
            dropout=dropout,
            softmax_temp=softmax_temp,
            attention_dropout=attention_dropout,
            #bits=bits,
            #rounds=rounds,
            #chunk_size=chunk_size,
            #masked=masked
        ).get()
        self.predictor = torch.nn.Linear(
            d_model,
            mixtures * 3
        )

    def forward(self, x, i=0, memory=None):
        x = x.view(x.shape[0])
        x = self.value_embedding(x)
        x = self.pos_embedding(x, i)
        y_hat, memory = self.transformer(x, memory)
        y_hat = self.predictor(y_hat)

        return y_hat, memory


def predict_with_recurrent(model, images, n):
    memory = None
    y_hat = []
    x_hat = []

    with torch.no_grad():
        for i in range(n):
            x_hat.append(images[:, i:i+1])
            yi, memory = model(x_hat[-1], i=i, memory=memory)
            y_hat.append(yi)

        for i in range(n, images.shape[1]):
            x_hat.append(sample_mol(y_hat[-1], 256))
            yi, memory = model(x_hat[-1], i=i, memory=memory)
            y_hat.append(yi)

        x_hat.append(sample_mol(y_hat[-1], 256))
        x_hat = torch.stack(x_hat, dim=1)

    return x_hat


def predict(model, images, n):
    N, L = images.shape
    x_hat = images.new_zeros(N, L+1, dtype=torch.long)
    x_hat[:, :n] = images[:, :n]
    with torch.no_grad():
        for i in range(n, L):
            y_hat = model(x_hat[:, :i])
            x_hat[:, i:i+1] = sample_mol(y_hat[:,-1,:], 256)
        x_hat[:, -1:] = sample_mol(y_hat[:,-1,:], 256)
    return x_hat


def index_type(x):
    if "," in x:
        return chain(*[index_type(xi) for xi in x.split(",")])

    if "-" in x:
        start, stop = x.split("-")
        return [i for i in range(int(start), int(stop))]

    if "*" in x:
        idx, mul = x.split("*")
        return [int(idx)]*int(mul)

    return [int(x)]


def collect_batch(dset, indices, device):
    images = torch.stack([dset[i][0] for i in indices], dim=0)
    return images.to(device)


def main(argv):
    parser = argparse.ArgumentParser(
        description="Generate an image from a pretrained model"
    )

    add_dataset_arguments(parser)
    add_transformer_arguments(parser)

    parser.add_argument(
        "model",
        help="The path to the model (give '-' for random intialization)"
    )

    parser.add_argument(
        "--mixtures",
        type=int,
        default=10,
        help="How many logistics to use to model the output"
    )

    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot the generated image"
    )
    parser.add_argument(
        "--save_image",
        help="Path to save an image to"
    )
    parser.add_argument(
        "--image_shape",
        type=lambda x: tuple(int(xi) for xi in x.split(",")),
        default=(28, 28),
        help="Reshape the prediction to plot it"
    )
    parser.add_argument(
        "--index",
        type=index_type,
        default=[0],
        help="Choose the index from the dataset"
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=300,
        help="Choose the offset in the image"
    )
    parser.add_argument(
        "--training_set",
        action="store_true",
        help="Predict from the training set"
    )

    parser.add_argument(
        "--load_pytorch",
        action="store_true",
        help="Load old pytorch model"
    )

    parser.add_argument(
        "--force_cpu",
        action="store_true",
        help="Set the device to cpu"
    )
    parser.add_argument(
        "--recurrent",
        action="store_true",
        help="Use a recurrent model for inference"
    )

    args = parser.parse_args(argv)
    print_transformer_arguments(args)

    # Choose a device to run on
    device = (
        "cuda" if torch.cuda.is_available() and not args.force_cpu
        else "cpu"
    )
    # Get the dataset and load the model
    train_set, test_set = get_dataset(args)
    if args.recurrent:
        model = RecurrentGenerator(
            args.d_query*args.n_heads,
            train_set.sequence_length, args.mixtures,
            attention_type=args.attention_type,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            d_query=args.d_query,
            dropout=args.dropout,
            softmax_temp=None,
            attention_dropout=args.attention_dropout,
            bits=args.bits,
            rounds=args.rounds, chunk_size=args.chunk_size,
            masked=True
        )
    else:
        model = ImageGenerator(
            args.d_query*args.n_heads,
            train_set.sequence_length, args.mixtures,
            attention_type=args.attention_type,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            d_query=args.d_query,
            dropout=args.dropout,
            softmax_temp=None,
            attention_dropout=args.attention_dropout,
            bits=args.bits,
            rounds=args.rounds, chunk_size=args.chunk_size,
            masked=True
        )

    # Gather the images
    images = collect_batch(
        train_set if args.training_set else test_set,
        args.index,
        device
    )

    # Load the model
    if args.model != "-":
        if args.load_pytorch:
            load_model_pytorch(args.model, model, None, device)
        else:
            load_model(args.model, model, None, device)

    model.to(device)
    model.eval()

    # Do the predictions
    if args.recurrent:
        timer = Timer()
        pred_images = predict_with_recurrent(
            model,
            images,
            args.offset
        )
        print("Elapsed time:", timer.measure())
    else:
        timer = Timer()
        pred_images = predict(
            model,
            images,
            args.offset
        )
        print("Elapsed time:", timer.measure())
     
    # Plot or save the images
    if args.plot:
        print(pred_images)
        pred_images = pred_images.cpu()
        images = images.cpu()
        plt.figure()
        plt.imshow(pred_images[0].reshape(*args.image_shape))
        plt.figure()
        plt.imshow(np.hstack([images[0], 0]).reshape(*args.image_shape))
        plt.show()

    if args.save_image:
        pred_images = pred_images.cpu()
        images = images.cpu()
        for i in range(len(images)):
            imwrite(
                args.save_image.format("pred", i),
                pred_images[i].reshape(*args.image_shape)
            )
            imwrite(
                args.save_image.format("real", i),
                np.hstack([images[i], 0]).reshape(*args.image_shape)
            )


if __name__ == "__main__":
    main(None)
