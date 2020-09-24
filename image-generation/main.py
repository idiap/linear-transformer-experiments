#
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>,
# Apoorv Vyas <avyas@idiap.ch>
#

import argparse
import math
import sys
from collections import OrderedDict

import torch
from torch.utils.data import DataLoader

from fast_transformers.masking import LengthMask, TriangularCausalMask
from fast_transformers.builders import TransformerEncoderBuilder

from image_datasets import add_dataset_arguments, get_dataset
from utils import add_optimizer_arguments, get_optimizer, \
    add_transformer_arguments, print_transformer_arguments, \
    EpochStats, load_model, save_model, \
    discretized_mix_logistic_loss as dmll


class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, dropout=0.0, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)
        self.d_model = d_model
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        pos_embedding =  self.pe[:, :x.size(1), :]
        pos_embedding = torch.repeat_interleave(pos_embedding, x.shape[0], dim=0)
        x =  torch.cat([x, pos_embedding], dim=2)
        return self.dropout(x)


class ImageGenerator(torch.nn.Module):
    def __init__(self, d_model, sequence_length, mixtures,
                 attention_type="full", n_layers=4, n_heads=4,
                 d_query=32, dropout=0.1, softmax_temp=None,
                 attention_dropout=0.1,
                 bits=32, rounds=4,
                 chunk_size=32, masked=True):
        super(ImageGenerator, self).__init__()

        self.pos_embedding = PositionalEncoding(
            d_model//2,
            max_len=sequence_length
        )
        self.value_embedding = torch.nn.Embedding(
            256,
            d_model//2
        )

        self.transformer = TransformerEncoderBuilder.from_kwargs(
            attention_type=attention_type,
            n_layers=n_layers,
            n_heads=n_heads,
            feed_forward_dimensions=n_heads*d_query*4,
            query_dimensions=d_query,
            value_dimensions=d_query,
            dropout=dropout,
            softmax_temp=softmax_temp,
            attention_dropout=attention_dropout,
            bits=bits,
            rounds=rounds,
            chunk_size=chunk_size,
            masked=masked
        ).get()

        hidden_size = n_heads*d_query
        self.predictor = torch.nn.Linear(
            hidden_size,
            mixtures * 3
        )

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.value_embedding(x)
        x = self.pos_embedding(x)
        triangular_mask = TriangularCausalMask(x.shape[1], device=x.device)
        y_hat = self.transformer(x, attn_mask=triangular_mask)
        y_hat = self.predictor(y_hat)

        return y_hat


def loss(y, y_hat):
    log2 = 0.6931471805599453
    y_hat = y_hat.permute(0, 2, 1).contiguous()
    N, C, L = y_hat.shape
    l = dmll(y_hat, y.view(N, L, 1))
    bpd = l.item() / log2

    return l, bpd


def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0
    total_bpd = 0
    total_samples = 0
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            y_hat = model(x)
            l, bpd = loss(y, y_hat)
            total_loss += x.shape[0] * l.item()
            total_bpd += x.shape[0] * bpd
            total_samples += x.shape[0]
    if sys.stdout.isatty():
        print()
    print(
        "Testing =>",
        "Loss:",
        total_loss/total_samples,
        "bpd:",
        total_bpd/total_samples
    )

    return total_loss/total_samples


def yield_batches(dataloader):
    while True:
        for batch in dataloader:
            yield batch


def train(model, optimizer, dataloader, iteration, total_iterations, callback,
          device):
    model.train()
    stats = EpochStats(["bpd", "lr"])
    batches = yield_batches(dataloader)
    stop = False
    while not stop and (iteration < total_iterations):
        x, y = next(batches)
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        y_hat = model(x)
        l, bpd = loss(y, y_hat)
        l.backward()
        optimizer.step()
        lr = optimizer.get_lr()[0]
        stats.update(x.shape[0], l.item(), [bpd, lr])
        stats.progress()
        iteration += 1
        stop = callback(iteration)
    stats.finalize()
    return stop


def saver(save_freq, save_to, model, optimizer):
    def inner(iteration):
        if (iteration % save_freq) == 0 and save_to:
            save_model(save_to, model, optimizer, iteration)
    return inner


def evaluator(evaluate_freq, model, dataloader, device):
    def inner(iteration):
        if (iteration % evaluate_freq) == 0:
            evaluate(model, dataloader, device)
    return inner


def stopper(yield_freq):
    def inner(iteration):
        return (iteration % yield_freq) == 0
    return inner


def callback_chain(*callbacks):
    def inner(iteration):
        ret = False
        for cb in callbacks:
            ret = ret or cb(iteration)
        return ret
    return inner


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Train a transformer to generate images"
    )

    add_transformer_arguments(parser)
    add_optimizer_arguments(parser)
    add_dataset_arguments(parser)

    parser.add_argument(
        "--mixtures",
        type=int,
        default=10,
        help="How many logistics to use to model the output"
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="How many iterations to train for"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="How many samples to use together"
    )

    parser.add_argument(
        "--save_to",
        default=None,
        help="Set a file to save the models to."
    )
    parser.add_argument(
        "--continue_from",
        default=None,
        help="Load the model from a file"
    )
    parser.add_argument(
        "--save_frequency",
        default=3000,
        type=int,
        help="Save every that many steps"
    )
    parser.add_argument(
        "--evaluate_frequency",
        default=3000,
        type=int,
        help="Evaluate on the test set after that many iterations"
    )
    parser.add_argument(
        "--yield_frequency",
        default=10**9,
        type=int,
        help="Stop after that many iterations so that other jobs can run"
    )

    args = parser.parse_args(argv)
    print_transformer_arguments(args)

    # Make the dataset and the model
    train_set, test_set = get_dataset(args)

    model = ImageGenerator(
        args.d_query*args.n_heads, train_set.sequence_length, args.mixtures,
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

    # Choose a device and move everything there
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Running on {}".format(device))
    model.to(device)

    # Start training
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        pin_memory=device=="cuda"
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        pin_memory=device=="cuda"
    )
    optimizer = get_optimizer(model.parameters(), args)
    iteration = 0
    if args.continue_from:
        iteration = load_model(
            args.continue_from,
            model,
            optimizer,
            device
        )
    optimizer.set_lr(args.lr)

    callbacks = callback_chain(
        saver(args.save_frequency, args.save_to, model, optimizer),
        evaluator(args.evaluate_frequency, model, test_loader, device),
        stopper(args.yield_frequency)
    )
    yielded = train(
        model,
        optimizer,
        train_loader,
        iteration,
        args.iterations,
        callbacks,
        device
    )

    # Non-zero exit code to notify the process watcher that we yielded
    if yielded:
        sys.exit(1)


if __name__ == "__main__":
    main()
