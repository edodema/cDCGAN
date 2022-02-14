# cDCGAN
A simple playground for *conditional Deep Convolutional Generative Adversarial Network* (cDCGAN)

## Repo structure
```
.
├── checkpoints
├── data
│   ├── cifar-100-python
│   ├── cifar-10-batches-py
│   └── MNIST
├── docs
│   ├── report
│   │   └── images
│   └── slides
│       └── images
├── notebooks
├── outputs
│   ├── cifar10
│   ├── fashion_mnist
│   └── mnist
├── scripts
└── src

```

## Usage
First of all remember to place yourself in the project root and activate the python environment.

**conda**

Easier way, just run
```
$ conda env create -f environment.yml
$ conda activate cDCGAN
```
or you can create it from scratch
```
$ conda create -n cDCGAN
$ conda activate cDCGAN
$ conda install -c anaconda pip
$ pip install -r requirements.txt
```

**venv**

For a more isolated approach you can use
```
$ python -m venv env
$ source env/bin/activate
$ pip install -r requirements.txt
```

### Scripts
**N.B. remember to give the scripts execution rights**.
```
$ ./scripts/create.sh
```
or
```
$ ./scripts/train.sh
```

### Examples
To generate a nice matrix of examples, one for each class, for the MNIST dataset execute
```
PYTHONPATH=. python src/main.py --dataset "mnist" --channels 1 --ncol 10 --figsize 16
```
of if you want to generate an example for a specific class select a label
```
PYTHONPATH=. python src/main.py --dataset "mnist" --channels 1 --ncol 10 --figsize 16 --label 0
```
For any doubt just ask the script for help
```
$ PYTHONPATH=. python src/main.py -h
```