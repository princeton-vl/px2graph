import tensorflow as tf
import importlib

from px2graph.util.loader import Loader

def init_task(opt, train_flag):
    # Initialize dataset
    print("Initializing dataset and task...")
    ds = importlib.import_module('px2graph.data.%s.ref' % opt.dataset)
    ds.initialize(opt) # Set up training/val split in ds.initialize

    # Setup task
    task = importlib.import_module('px2graph.task.' + opt.task)
    task = task.Task(opt, ds)

    # Data loader
    loader = Loader(opt, task, ds, train_flag)
    new_sample = loader.get_sample
    inp = new_sample[:task.num_inputs]
    label = new_sample[task.num_inputs:-1]
    sample_idx = new_sample[-1]

    return task, loader, inp, label, sample_idx


def average_gradients(tower_grads):
    average_grads = []
    for grads_and_vars in zip(*tower_grads):
        grads = [g for g,_ in grads_and_vars]
        if grads[0] is not None:
            grad = tf.reduce_mean(tf.stack(grads, 0), 0)
            v = grads_and_vars[0][1]
            grad_and_var = (grad, v)
        else:
            grad_and_var = grads_and_vars[0]
        average_grads += [grad_and_var]
    return average_grads


def merge_gpu_results(*args, concat_axis=0, do_mean=False):
    if None in args:
        # Check for None in args
        print("None?")
        return None
    if type(args[0]) == list:
        ans = []
        for i in range(len(args[0])):
            ans += [tf.concat([j[i] for j in args], axis=concat_axis)]
    elif type(args[0]) == dict:
        ans = {}
        for i in args[0].keys():
            if do_mean:
                ans[i] = tf.reduce_mean(tf.stack([j[i] for j in args]))
            else:
                ans[i] = tf.concat([j[i] for j in args], axis=concat_axis)
    else:
        ans = tf.concat(args, axis=concat_axis)
    return ans


def init_model(opt, task, inp, label, sample_idx, train_flag):
    # Load model and set up optimization
    print("Defining computation graph...")

    global_step = tf.get_variable('global_step', [], tf.int32, trainable=False)
    model = importlib.import_module('px2graph.models.' + opt.model)
    opt_fn = tf.train.__dict__[opt.optimizer]
    opt_args = []
    lr = tf.placeholder(tf.float32, [])

    if opt.optimizer == 'MomentumOptimizer': opt_args = [.9]

    optim = opt_fn(lr, *opt_args)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    def setup_network(inp, label):
        net = model.initialize(inp, opt, train_flag)
        loss = task.loss(net, label, sample_idx)
        pred, accuracy = task.postprocess(net, label, sample_idx)
        return net, loss, pred, accuracy

    def setup_grad(optim, loss):
        with tf.control_dependencies(update_ops):
            update_vars = None
            if opt.to_train is not None:
                update_vars = [tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=s)
                               for s in opt.to_train]
            return optim.compute_gradients(sum(loss), var_list=update_vars)
        
    if opt.gpu_num == 1:
        # Single GPU
        net, loss, pred, accuracy = setup_network(inp, label)
        grads = setup_grad(optim, loss)

    else:
        # Multiple GPUs
        tmp, l = [], 0
        for i in range(opt.gpu_num):
            r = min(l + opt.batch_split, opt.batchsize)
            do_reuse = i > 0
            with tf.device('/gpu:%d' % i), \
                 tf.variable_scope(tf.get_variable_scope(), reuse=do_reuse):

                print("Setting up on GPU", i)
                inp_ = [tf.identity(tmp_inp[l:r]) for tmp_inp in inp]
                label_ = [tf.identity(tmp_lbl[l:r]) for tmp_lbl in label]
                for j, val in enumerate(setup_network(inp_, label_)):
                    if i == 0: tmp += [[]]
                    tmp[j] += [val]
                if i == 0: tmp += [[]]
                tmp[-1] += [setup_grad(optim, tmp[1][-1])]

            l = r

        grads = average_gradients(tmp[-1])
        vals = [merge_gpu_results(*tmp[0], concat_axis=0),
                tf.reduce_mean(tf.stack(tmp[1],0),0),
                merge_gpu_results(*tmp[2], concat_axis=0),
                merge_gpu_results(*tmp[3], do_mean=True)]
        net, loss, pred, accuracy = vals

    with tf.control_dependencies(update_ops):
        if opt.clip_grad: grads = [(tf.clip_by_value(tmp_[0], -opt.clip_grad, opt.clip_grad), tmp_[1]) \
                                    if tmp_[0] is not None else tmp_ for tmp_ in grads]
        optim = optim.apply_gradients(grads, global_step=global_step)

    return net, loss, pred, accuracy, optim, lr
