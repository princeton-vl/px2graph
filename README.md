# Pixels to Graphs by Associative Embedding

Tensorflow training pipeline for:

**Pixels to Graphs by Associative Embedding.** [Alejandro Newell](http://www-personal.umich.edu/~alnewell/) and [Jia Deng](http://web.eecs.umich.edu/~jiadeng/). Neural Information Processing Systems (NIPS), 2017.
[arXiv:1706.07365](https://arxiv.org/abs/1706.07365)

## Getting started

#### Python 3 package requirements: 

(code tested on Python 3.5 and 3.6)

- tensorflow_gpu (1.3.0)
- numpy
- h5py
- scipy
- scikit-image
- tqdm
- easydict
- graphviz
- simplejson (for Visual Genome driver)

Make sure to add the parent directory of px2graph to your PYTHONPATH.

#### Set up munkres-tensorflow:

- Clone repo from [here](https://github.com/mbaradad/munkres-tensorflow)
- Follow build instructions (make sure to use g++ 4.X)
- Place 'hungarian.so' in util/

#### Download data

- Download images from [Visual Genome](http://visualgenome.org/api/v0/api_home.html) (parts 1 and 2)
- Place all images into data/genome/VG_100K/
- Download VG-SGG.h5 and proposals.h5 from [here](https://github.com/danfeiX/scene-graph-TF-release/tree/master/data_tools) and place them in data/genome/ (credit to Xu et al. 'Scene Graph Generation by Iterative Message Passing' for preprocessing the Visual Genome annotations)

#### Pretrained models

Pretrained models will be available soon.

## Using the code

To train a network, call:

```python main.py -e [experiment name] --sg_task [PR|CL|SG]```

The different task settings are defined as follows:

- PR: Object boxes and classes provided, predict relationships
- CL: Object boxes provided, classify objects and their relationships
- SG: Full task, nothing but image required as input (an additional argument ```--use_rpn [0|1]``` determines whether or not to include box proposals from a RPN)

To train with multiple GPUs use a command like:

```python main.py --batchsize 32 --gpu_choice '0,1,2,3'```

To continue an experiment or branch off to a new directory (to test different hyperparameters for example), call:

```python main.py -e test_exp --continue 1``` or ```python main.py --branch test_exp -e test_exp_2 --learning_rate 1e-5```

In general, any binary options you wish to change via the command line must be set explicitly with either a 0 or 1. Use ```--help``` to get a list of available options to set.

#### Evaluating a model

The recall logged during training should give a good sense of the performance of the model. After training, to generate a proper set of predictions and evaluate them, call:

```
python main.py --branch test_exp -e my_results --predict valid
python eval.py my_results/valid_preds
```

The ```--predict``` argument accepts ```train|valid|test``` as options, and will generate predictions on the first ```opt.valid_iters * opt.batchsize``` samples of the specified subset.
