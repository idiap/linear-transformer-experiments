Image Generation with Transformers
==================================

In this task, the transformer learns to model an image one byte at a time.
Namely, given a sequence of bytes,

    r1 g1 b1 r2 g2 b2 ... rn gn

the transformer must predict the next value, in this case bn.

Requirements
------------

The installation requirements to run the training script (`main.py`) are 

* torch
* torchvision
* pytorch-fast-transformers

They can be installed in most systems via

    pip install torch torchvision pytorch-fast-transformers

However, in order to run the image generation script (`prediction.py`), you
need also

* matplotlib
* imageio

Running the code
----------------

### Training

The training is done using `main.py`. The script comes with thorough command
line help which is shown using the `--help` command line argument.

The following two commands train a linear transformer for MNIST and CIFAR-10
respectively.

    python main.py --dataset mnist --attention_type causal-linear \
        --d_query 32 --n_heads 8 --n_layers 8 --mixtures 10 \
        --batch_size 16 --iterations 937500 --evaluate_every 10000 \
        --save_to /path/to/weights.{}.pth \
        --continue_from /path/to/weights.{}.pth

    python main.py --dataset cifar10 --attention_type causal-linear \
        --d_query 32 --n_heads 8 --n_layers 16 --mixtures 10 --lr 2e-4 \
        --batch_size 10 --iterations 500000 --evaluate_every 10000 \
        --save_to /path/to/weights.{}.pth \
        --continue_from /path/to/weights.{}.pth

### Prediction

After training your model or downloading a model you can generate images using
the `prediction.py` script.

The following code generates images from MNIST after training for a few epochs.

    python prediction.py --dataset mnist --attention_type causal-linear \
        --d_query 32 --n_heads 8 --n_layers 8 --mixtures 10 \
        --index 0-10 --offset 1 --recurrent --force_cpu \
        --save_image /path/to/images/mnist.{}.{}.pth \
        /path/to/weights.XX.pth

The following arguments affect the generation process

1. `--index` Selects the images from the test set that will be used to
   condition the generation on
2. `--offset` Selects how many pixels to condition on (setting it to 1 amounts
   to unconditional generation)
3. `--recurrent` Chooses to use a recurrent model which is optimised for
   inference, instead of a batch model which is optimised for training
