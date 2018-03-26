# Pixels to Graphs by Associative Embedding

Tensorflow training pipeline for:

**Pixels to Graphs by Associative Embedding.** [Alejandro Newell](http://www-personal.umich.edu/~alnewell/) and [Jia Deng](http://web.eecs.umich.edu/~jiadeng/). Neural Information Processing Systems (NIPS), 2017.
[arXiv:1706.07365](https://arxiv.org/abs/1706.07365)

**Update (March 25, 2018):** Changed default optimization to deal with diverging models, so code should no longer freeze in the middle of training. Also, finally added pretrained models!

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
- For newer versions of TensorFlow it was a bit difficult to get this working, here is the command that finally worked for me:
```
TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
TF_LIB=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')

g++-4.9 -std=c++11 -shared hungarian.cc -o hungarian.so -fPIC -I$TF_INC -I$TF_INC/external/nsync/public -L$TF_LIB -ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=0
```
- Place 'hungarian.so' in util/

#### Download data

- Download images from [Visual Genome](http://visualgenome.org/api/v0/api_home.html) (parts 1 and 2)
- Place all images into data/genome/VG_100K/
- Download VG-SGG.h5 and proposals.h5 from [here](https://github.com/danfeiX/scene-graph-TF-release/tree/master/data_tools) and place them in data/genome/ (credit to Xu et al. 'Scene Graph Generation by Iterative Message Passing' for preprocessing the Visual Genome annotations)

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

*Important:* Regarding optimization, some feature of the code can lead to spontaneous divergence in the midst of training


#### Evaluating a model

The recall logged during training should give a good sense of the performance of the model. After training, to generate a proper set of predictions and evaluate them, call:

```
python main.py --branch test_exp -e my_results --predict valid
python eval.py my_results/valid_preds
```

The ```--predict``` argument accepts ```train|valid|test``` as options, and will generate predictions on the first ```opt.valid_iters * opt.batchsize``` samples of the specified subset.


#### Pretrained models

| Task Setting  | R @ 50 | R @ 100 |
| ------------ | :--------------: | :--------------: |
| [SGGen (SG) (no RPN)](https://umich.box.com/s/ryl3raq5c3qg9udhdyfx4ipkosg1ybn3) | 15.5 | 18.8 |
| [SGCls (CL)](https://umich.box.com/s/0g9pzom76i9ujpsi0fg79nzbltilj2wb) | 35.7 | 38.4 |
| [PredCls (PR)](https://umich.box.com/s/10pylql82xwa2ctomk7y868yweky9q5w) | 82.0 | 86.4 |

Note, performance is higher than previously reported in our paper. This is not due to any changes in the algorithm, but instead is the result of training for longer and tuning some postprocessing parameters.

Another important detail regarding our evaluation is that our numbers are reported with _unconstrained_ graphs meaning that when proposing the top 50 or 100 predicates, multiple edges may be proposed between any two nodes. This leads to a notable performance difference when comparing to numbers in a constrained setting particularly on the SGCls and PredCls tasks where nodes are provided ahead of time.

To get final test set predictions from the pretrained models, put them in the exp/ directory and call:

```
python main.py --branch pretrained_sg -e sg_results --predict test
```
