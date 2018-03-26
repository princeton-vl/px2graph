import argparse
import sys
import pickle
import os, os.path

def mkdir_p(dirname):
    try: os.makedirs(dirname)
    except FileExistsError: pass

def parse_command_line():

    ### ======================================================================
    ### Command line options
    ### ======================================================================

    parser = argparse.ArgumentParser()

    parser.add_argument('-t','--task', type=str, default='graph')
    parser.add_argument('-m','--model', type=str, default='px2graph')
    parser.add_argument('-d','--dataset', type=str, default='genome')

    # Experiment
    parser.add_argument('-e','--exp_id', type=str, default='default')
    g = parser.add_mutually_exclusive_group()
    g.add_argument('-c','--continue_exp', type=int, default=0,
        help='Continue an experiment (keep in same experiment directory)')
    g.add_argument('--branch', default='',
        help='Branch from experiment (start a new experiment directory)')
    parser.add_argument('--predict', type=str, default='',
        help='Generate a final set of predictions (train | valid | test)')
    parser.add_argument('--num_rounds', type=int, default=1000,
        help='Number of training rounds')
    parser.add_argument('--train_iters', type=int, default=2000,
        help='Number of training iterations per round')
    parser.add_argument('--valid_iters', type=int, default=100,
        help='Number of validation iterations per round')
    parser.add_argument('--gpu_choice', type=str, default='0',
        help='Specify which GPU amongst visible devices to use (can be comma separated for multiple GPUs eg \'0,1\')')
    parser.add_argument('--suppress_output', type=int, default=0,
        help='Suppress print statements used to monitor training progress')

    # Hyperparameters
    parser.add_argument('-l','--learning_rate', type=float, default=3e-3)
    parser.add_argument('--batchsize', type=int, default=8)
    parser.add_argument('--optimizer', type=str, default='MomentumOptimizer')
    parser.add_argument('--drop_rate', type=float, default=0.)
    parser.add_argument('--clip_grad', type=float, default=1.)
    parser.add_argument('--batchnorm', type=int, default=0)

    # Data
    parser.add_argument('--input_res', type=int, default=512)
    parser.add_argument('--output_res', type=int, default=64)
    parser.add_argument('--num_data_threads', type=int, default=4)
    parser.add_argument('--max_queue_size', type=int, default=256)

    # Model
    parser.add_argument('--num_feats', type=int, default=256)
    parser.add_argument('--num_stack', type=int, default=4)
    parser.add_argument('--pretrained', type=str, default='')
    parser.add_argument('--trainable', type=str, default='')
    parser.add_argument('--scale_feats', type=str, default='1-2-1-1')

    # Scene graph options
    parser.add_argument('--sg_task', type=str, default='SG',
        help='Scene graph training/evaluation setting (PR | CL | SG)')
    parser.add_argument('--use_rpn', type=int, default=0,
        help='Set to 1 to use RPN proposals (only applies in SG setting)')
    parser.add_argument('--obj_slots', type=int, default=3)
    parser.add_argument('--rel_slots', type=int, default=6)
    parser.add_argument('--obj_thr', type=str, default='0.15-.06-.15')
    parser.add_argument('--rel_thr', type=str, default='0.03-0.01-0.03')
    parser.add_argument('--class_thr', type=str, default='.5-.5-.15-15')

    # sg_task:CL defaults - obj_thr: .15-.15-.2, rel_thr: .005-.01-.001

    parser.add_argument('--obj_box_nms', type=float, default=.5)
    parser.add_argument('--obj_hm_nms', type=float, default=0)
    parser.add_argument('--rel_hm_nms', type=float, default=0)

    parser.add_argument('--rel_top_k', type=int, default=10)

    # Tag options
    parser.add_argument('--tag_dim', type=int, default=8)
    parser.add_argument('--tag_margin', type=int, default=8)


    flags, unparsed = parser.parse_known_args()
    flags.restore_session = None
    last_round = 0

    ### ======================================================================
    ### Process and load/save options
    ### ======================================================================

    # Check if we need to load up a previous set of options
    if flags.continue_exp or len(flags.branch) > 0:
        if flags.continue_exp: tmp_exp_dir = flags.exp_id
        else: tmp_exp_dir = flags.branch

        # Check which flags have been manually set and keep them
        short_ref = {'-d':'dataset', '-e':'exp_id', '-c':'continue',
                     '-t':'task', '-m':'model', '-l':'learning_rate'}
        opts = {}
        for val in sys.argv:
            if val[0] == '-':
                if val in short_ref: tmp_arg = short_ref[val]
                else: tmp_arg = val[2:]
                if tmp_arg in flags.__dict__:
                    opts[tmp_arg] = flags.__dict__[tmp_arg]

        # Load old flags
        with open('exp/'+tmp_exp_dir+'/opts.p','rb') as f:
            old_flags = pickle.load(f)

        # Check if optimizer has been changed
        old_flags.new_optim = False
        if 'optimizer' in opts:
            if old_flags.optimizer != opts['optimizer']:
                print("Changing optimizer from %s to %s" %
                      (old_flags.optimizer, opts['optimizer']))
                old_flags.new_optim = True
        elif old_flags.optimizer == 'None':
            old_flags.optimizer = 'MomentumOptimizer'
            old_flags.new_optim = True

        # Replace values that have been manually set
        for opt in opts.keys():
            old_flags.__dict__[opt] = opts[opt]

        # Fill in any unset values new options
        for k in flags.__dict__.keys():
            if not k in old_flags.__dict__:
                print("adding default for",k,":",flags.__dict__[k])
                old_flags.__dict__[k] = flags.__dict__[k]

        flags = old_flags

        # Mark flag to load an old session
        flags.restore_session = tmp_exp_dir

        try:
            with open('exp/'+tmp_exp_dir+'/last_round','r') as f:
                last_round = int(f.readline())
        except:
            pass

    flags.last_round = last_round + flags.num_rounds

    # Save options
    mkdir_p('exp/'+flags.exp_id)
    with open('exp/'+flags.exp_id+'/opts.p','wb') as f: 
        pickle.dump(flags, f)
    
    # Process restoring and locking of variable scopes
    load_from = {}
    if len(flags.pretrained) > 0:
        tmp = flags.pretrained.split(';')
        assert len(tmp) % 2 == 0, "Format for loading pretrained weights: --pretrained \"model_1;scope_1,scope_2;model_2;scope_1,scope_3;model_3;scope_2\""
        for i in range(0,len(tmp),2):
            load_from[tmp[i]] = tmp[i+1].split(',')

    flags.load_from = load_from
    flags.to_train = None if len(flags.trainable) == 0 else flags.trainable.split(',')

    flags.iters = {'train':flags.train_iters, 'valid':flags.valid_iters}

    disp_opts = [['Experiment ID', flags.exp_id],
                 ['Network model', flags.model],
                 ['Dataset', flags.dataset],
                 ['Task', flags.task],
                 ['Optimizer', flags.optimizer],
                 ['Batchsize', flags.batchsize],
                 ['Learning rate', flags.learning_rate]]
    if flags.restore_session: disp_opts += [['Restoring from', flags.restore_session]]

    flags.gpu_num = len(flags.gpu_choice.split(','))
    if flags.gpu_num > 1:
        disp_opts += [['# of GPUS', flags.gpu_num]]
        flags.batch_split = (flags.batchsize + flags.gpu_num - 1) // flags.gpu_num
        disp_opts += [['Batchsize/GPU', flags.batch_split]]
    else:
        flags.batch_split = flags.batchsize

    print("---------------------------------------------")
    for tmp_opt in disp_opts: print("{:15s}: {}".format(*tmp_opt))
    print("---------------------------------------------")

    # Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = flags.gpu_choice
    # Suppress further output
    if flags.suppress_output:
        f = open(os.devnull,'w')
        sys.stdout = f
        sys.stderr = f

    # Split strings into lists
    split_str = lambda x: list(map(float, x.split('-')))
    flags.scale_feats = split_str(flags.scale_feats)
    flags.obj_thr = split_str(flags.obj_thr)
    flags.rel_thr = split_str(flags.rel_thr)
    flags.class_thr = split_str(flags.class_thr)

    return flags

if __name__ == '__main__':
    f = parse_command_line()
    print(f)
