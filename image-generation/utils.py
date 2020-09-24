#
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>,
# Apoorv Vyas <avyas@idiap.ch>
#

from glob import iglob
from os import path, rename
import time
from collections import namedtuple
import sys

import numpy as np
import torch
import torch.nn.functional as F

from radam import RAdam

from fast_transformers.weight_mapper import PytorchMapper


def log_sum_exp(x):
    """ numerically stable log_sum_exp implementation that prevents overflow """
    # TF ordering
    axis = len(x.size()) - 1
    m, _ = torch.max(x, dim=axis)
    m2, _ = torch.max(x, dim=axis, keepdim=True)
    return m + torch.log(torch.sum(torch.exp(x - m2), dim=axis))


def discretized_mix_logistic_loss(y_hat, y, num_classes=256,
                                  log_scale_min=-7.0, reduce=True):
    """Discretized mixture of logistic distributions loss
    Note that it is assumed that input is scaled to [-1, 1].
    Args:
        y_hat (Tensor): Predicted output (B x C x T)
        y (Tensor): Target (B x T x 1).
        num_classes (int): Number of classes
        log_scale_min (float): Log scale minimum value
        reduce (bool): If True, the losses are averaged or summed for each
          minibatch.
    Returns
        Tensor: loss
    """
    assert y_hat.dim() == 3
    assert y_hat.size(1) % 3 == 0
    nr_mix = y_hat.size(1) // 3

    # (B x T x C)
    y_hat = y_hat.transpose(1, 2)

    # unpack parameters. (B, T, num_mixtures) x 3
    logit_probs = y_hat[:, :, :nr_mix]
    means = y_hat[:, :, nr_mix:2 * nr_mix]
    log_scales = torch.clamp(y_hat[:, :, 2 * nr_mix:3 * nr_mix], min=log_scale_min)

    # B x T x 1 -> B x T x num_mixtures
    y = y.expand_as(means)

    centered_y = y - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_y + 1. / (num_classes - 1))
    cdf_plus = torch.sigmoid(plus_in)
    min_in = inv_stdv * (centered_y - 1. / (num_classes - 1))
    cdf_min = torch.sigmoid(min_in)

    # log probability for edge case of 0 (before scaling)
    # equivalent: torch.log(torch.sigmoid(plus_in))
    log_cdf_plus = plus_in - F.softplus(plus_in)

    # log probability for edge case of 255 (before scaling)
    # equivalent: (1 - torch.sigmoid(min_in)).log()
    log_one_minus_cdf_min = -F.softplus(min_in)

    # probability for all other cases
    cdf_delta = cdf_plus - cdf_min

    mid_in = inv_stdv * centered_y
    # log probability in the center of the bin, to be used in extreme cases
    # (not actually used in our code)
    log_pdf_mid = mid_in - log_scales - 2. * F.softplus(mid_in)

    # tf equivalent
    """
    log_probs = tf.where(x < -0.999, log_cdf_plus,
                         tf.where(x > 0.999, log_one_minus_cdf_min,
                                  tf.where(cdf_delta > 1e-5,
                                           tf.log(tf.maximum(cdf_delta, 1e-12)),
                                           log_pdf_mid - np.log(127.5))))
    """
    # TODO: cdf_delta <= 1e-5 actually can happen. How can we choose the value
    # for num_classes=65536 case? 1e-7? not sure..
    inner_inner_cond = (cdf_delta > 1e-5).float()

    inner_inner_out = inner_inner_cond * \
        torch.log(torch.clamp(cdf_delta, min=1e-12)) + \
        (1. - inner_inner_cond) * (log_pdf_mid - np.log((num_classes - 1) / 2))
    inner_cond = (y > 0.999).float()
    inner_out = inner_cond * log_one_minus_cdf_min + (1. - inner_cond) * inner_inner_out
    cond = (y < -0.999).float()
    log_probs = cond * log_cdf_plus + (1. - cond) * inner_out

    log_probs = log_probs + F.log_softmax(logit_probs, -1)

    if reduce:
        return -torch.mean(log_sum_exp(log_probs))
    else:
        return -log_sum_exp(log_probs).unsqueeze(-1)


# Simpler equivalent of the code above but since people are using the above
# code that is ported around from tf that's what we will be using. In all our
# tests the code below works as well.
#
# def dmll(y_hat, y, num_classes=256):
#     nr_mix = y_hat.size(1) // 3
#     y_hat = y_hat.transpose(1, 2)
#
#     probs = torch.softmax(y_hat[:, :, :nr_mix], dim=-1)
#     means = y_hat[:, :, nr_mix:2 * nr_mix]
#     scales = torch.nn.functional.elu(y_hat[:, :, 2*nr_mix:3*nr_mix]) + 1.0001
#
#     nonzero_mask = (y > -0.999).float()
#     nonmax_mask = (y < 0.999).float()
#     ycentered = y - means
#     step = 1/(num_classes-1)
#     yneg = torch.sigmoid((ycentered - step)/scales) * nonzero_mask
#     ypos = torch.sigmoid((ycentered + step)/scales) * nonmax_mask + (1-nonmax_mask)
#
#     prob_of_data = (probs*(ypos - yneg)).sum(-1)
#
#     return -torch.log(prob_of_data).mean()


def sample_mol(y_hat, num_classes=256):
    """Sample from mixture of logistics.

    y_hat: NxC where C is 3*number of logistics
    """
    assert len(y_hat.shape) == 2

    N = y_hat.size(0)
    nr_mix = y_hat.size(1) // 3

    probs = torch.softmax(y_hat[:, :nr_mix], dim=-1)
    means = y_hat[:, nr_mix:2 * nr_mix]
    scales = torch.nn.functional.elu(y_hat[:, 2*nr_mix:3*nr_mix]) + 1.0001

    indices = torch.multinomial(probs, 1).squeeze()
    batch_indices = torch.arange(N, device=probs.device)
    mu = means[batch_indices, indices]
    s = scales[batch_indices, indices]
    u = torch.rand(N, device=probs.device)
    preds = mu + s*(torch.log(u) - torch.log(1-u))

    return torch.min(
        torch.max(
            torch.round((preds+1)/2*(num_classes-1)),
            preds.new_zeros(1),
        ),
        preds.new_ones(1)*(num_classes-1)
    ).long().view(N, 1)


def add_optimizer_arguments(parser):
    parser.add_argument(
        "--optimizer",
        choices=["radam", "adam"],
        default="radam",
        help="Choose the optimizer"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Set the learning rate"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Set the weight decay"
    )


def get_optimizer(params, args):
    if args.optimizer == "adam":
        return torch.optim.Adam(params, lr=args.lr)
    elif args.optimizer == "radam":
        return RAdam(params, lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise RuntimeError("Optimizer {} not available".format(args.optimizer))


def add_transformer_arguments(parser):
    parser.add_argument(
        "--attention_type",
        type=str,
        choices=["full", "causal-linear", "reformer"],
        default="causal-linear",
        help="Attention model to be used"
    )
    parser.add_argument(
        "--n_layers",
        type=int,
        default=4,
        help="Number of self-attention layers"
    )
    parser.add_argument(
        "--n_heads",
        type=int,
        default=6,
        help="Number of attention heads"
    )
    parser.add_argument(
        "--d_query",
        type=int,
        default=32,
        help="Dimension of the query, key, and value embedding"
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout to be used for transformer layers"
    )
    parser.add_argument(
        "--softmax_temp",
        type=float,
        default=None,
        help=("Softmax temperature to be used for training "
              "(default: 1/sqrt(d_query))")
    )
    parser.add_argument(
        "--attention_dropout",
        type=float,
        default=0.1,
        help="Dropout to be used for attention layers"
    )
    parser.add_argument(
        "--bits",
        type=int,
        default=32,
        help="Number of planes to use for hashing for reformer"
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=32,
        help="Number of queries in each block for reformer"
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=4,
        help="Number of rounds of hashing for reformer"
    )
    parser.add_argument(
        "--unmasked_reformer",
        action="store_false",
        dest="masked",
        help="If set the query can attend to itsself for reformer"
    )

    return parser


def print_transformer_arguments(args):
    print((
        "Transformer Config:\n"
        "    Attention type: {attention_type}\n"
        "    Number of layers: {n_layers}\n"
        "    Number of heads: {n_heads}\n"
        "    Key/Query/Value dimension: {d_query}\n"
        "    Transformer layer dropout: {dropout}\n"
        "    Softmax temperature: {softmax_temp}\n"
        "    Attention dropout: {attention_dropout}\n"
        "    Number of hashing planes: {bits}\n"
        "    Chunk Size: {chunk_size}\n"
        "    Rounds: {rounds}\n"
        "    Masked: {masked}"
    ).format(**vars(args)))


class EpochStats(object):
    def __init__(self, metric_names=[], freq=1, out=sys.stdout):
        self._start = time.time()
        self._samples = 0
        self._loss = 0
        self._metrics = [0]*len(metric_names)
        self._metric_names = metric_names
        self._out = out
        self._freq = freq
        self._max_line = 0

    def update(self, n_samples, loss, metrics=[]):
        self._samples += n_samples
        self._loss += loss*n_samples
        for i, m in enumerate(metrics):
            self._metrics[i] += m*n_samples

    def _get_progress_text(self):
        time_per_sample = (time.time()-self._start) / self._samples
        loss = self._loss / self._samples
        metrics = [
            m/self._samples
            for m in self._metrics
        ]
        text = "Loss: {} ".format(loss, self._samples)
        text += " ".join(
            "{}: {}".format(mn, m)
            for mn, m in zip(self._metric_names, metrics)
        )
        if self._out.isatty():
            to_add = " [{} sec/sample]".format(time_per_sample)
            if len(text) + len(to_add) > self._max_line:
                self._max_line = len(text) + len(to_add)
            text += " " * (self._max_line-len(text)-len(to_add)) + to_add
        else:
            text += " time: {}".format(time_per_sample)
        return text

    def progress(self):
        if self._samples < self._freq:
            return
        text = self._get_progress_text()
        if self._out.isatty():
            print("\r" + text, end="", file=self._out)
        else:
            print(text, file=self._out, flush=True)
        self._loss = 0
        self._samples = 0
        self._last_progress = 0
        for i in range(len(self._metrics)):
            self._metrics[i] = 0
        self._start = time.time()

    def finalize(self):
        self._freq = 1
        self.progress()
        if self._out.isatty():
            print("\n", file=self._out)


def normalize_checkpoint_file(filename):
    def replace_formats(x):
        chars = []
        pending = []
        for c in x:
            if not pending:
                if c == "{":
                    pending.append(c)
                else:
                    chars.append(c)
            else:
                if c == "{":
                    chars.extend(pending)
                    pending.clear()
                    pending.append("{")
                elif c == "}":
                    chars.append("*")
                    pending.clear()
                else:
                    pending.append(c)
        chars.extend(pending)
        return "".join(chars)

    if path.exists(filename):
        return filename
    if "{" in filename:
        filename = replace_formats(filename)
        try:
            return max(iglob(filename), key=path.getmtime)
        except ValueError as e:
            raise ValueError("No checkpoint found") from e


def load_model(saved_file, model, optimizer, device):
    try:
        saved_file = normalize_checkpoint_file(saved_file)
        data = torch.load(saved_file, map_location=device)
        print("Loading from", saved_file)
        if model is not None:
            model.load_state_dict(data["model_state"], strict=False)
        if optimizer is not None:
            optimizer.load_state_dict(data["optimizer_state"])
        return data["iteration"]
    except (ValueError, KeyError):
        return 0


def load_model_pytorch(saved_file, model, optimizer, device):
    try:
        saved_file = normalize_checkpoint_file(saved_file)
        data = torch.load(saved_file, map_location=device)
        print("Loading from", saved_file)
        if model is not None:
            previous_state_dict = data["model_state"]
            previous_state_dict['pos_embedding.pe'] = \
                model.state_dict()['pos_embedding.pe']
            model.load_state_dict(
                PytorchMapper().map(previous_state_dict), strict=False
            )
        if optimizer is not None:
            optimizer.load_state_dict(data["optimizer_state"])
        return data["iteration"]
    except (ValueError, KeyError):
        return 0


def save_model(save_file, model, optimizer, iteration):
    final_path = save_file.format(iteration)
    tmp_path = final_path + ".tmp"
    torch.save(
        dict(
            model_state=model.state_dict(),
            optimizer_state=optimizer.state_dict(),
            iteration=iteration
        ),
        tmp_path
    )
    # Renaming is an atomic operation so we never end up with half-written
    # files (all or nothing)
    rename(tmp_path, final_path)
