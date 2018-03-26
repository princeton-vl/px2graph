import tensorflow as tf
import numpy as np
import h5py
from tqdm import tqdm

from px2graph.util import setup
from px2graph.opts import parse_command_line

def main():

    # Initial setup
    opt = parse_command_line()
    train_flag = tf.placeholder(tf.bool, [])
    task, loader, inp, label, sample_idx = setup.init_task(opt, train_flag)
    net, loss, pred, accuracy, optim, lr = setup.init_model(opt, task, inp, label,
                                                            sample_idx, train_flag)

    # Prepare TF session
    summaries, image_summaries = task.setup_summaries(net, inp, label, loss, pred, accuracy)
    saver = tf.train.Saver()
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    sess.run(tf.global_variables_initializer())

    writer = tf.summary.FileWriter('exp/'+opt.exp_id, sess.graph)

    # Start data loading threads
    loader.start_threads(sess)

    # Restore previous session if continuing experiment
    if opt.restore_session is not None:
        print("Restoring previous session...",'(exp/' + opt.restore_session + '/snapshot)')

        if opt.new_optim:
          # Optimizer changed, don't load values associated with old optimizer
          tmp_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
          tmp_saver = tf.train.Saver(tmp_vars)
          tmp_saver.restore(sess, 'exp/' + opt.restore_session + '/snapshot')
        else:
          saver.restore(sess, 'exp/' + opt.restore_session + '/snapshot')

    # Load pretrained weights
    for tmp_model,scopes in opt.load_from.items():
        for tmp_scope in scopes:
            print("Loading weights from: %s, scope: %s" % (tmp_model, tmp_scope))
            tmp_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=tmp_scope)
            tmp_saver = tf.train.Saver(tmp_vars)
            tmp_saver.restore(sess, 'exp/%s/snapshot' % tmp_model)

    if opt.predict == '':

        splits = [s for s in ['train', 'valid'] if opt.iters[s] > 0]
        start_round = opt.last_round - opt.num_rounds

        # Main training loop
        for round_idx in range(start_round, opt.last_round):
            for split in splits:

                print("Round %d: %s" % (round_idx, split))
                loader.start_epoch(sess, split, train_flag, opt.iters[split] * opt.batchsize)

                flag_val = split == 'train'

                for step in tqdm(range(opt.iters[split]), ascii=True):
                    global_step = step + round_idx * opt.iters[split]
                    to_run = [sample_idx, summaries[split], loss, accuracy]
                    if split == 'train': to_run += [optim]

                    # Do image summaries at the end of each round
                    do_image_summary = step == opt.iters[split] - 1
                    if do_image_summary: to_run[1] = image_summaries[split]

                    # Start with lower learning rate to prevent early divergence
                    t = 1/(1+np.exp(-(global_step-5000)/1000))
                    lr_start = opt.learning_rate / 15
                    lr_end = opt.learning_rate
                    tmp_lr = (1-t) * lr_start + t * lr_end

                    # Run computation graph
                    result = sess.run(to_run, feed_dict={train_flag:flag_val, lr:tmp_lr})

                    out_loss = result[2]
                    if sum(out_loss) > 1e5:
                        print("Loss diverging...exiting before code freezes due to NaN values.")
                        print("If this continues you may need to try a lower learning rate, a")
                        print("different optimizer, or a larger batch size.")
                        return

                    # Log data
                    if split == 'valid' or (split == 'train' and step % 20 == 0) or do_image_summary:
                        writer.add_summary(result[1], global_step)
                        writer.flush()

            # Save training snapshot
            saver.save(sess, 'exp/' + opt.exp_id + '/snapshot')
            with open('exp/' + opt.exp_id + '/last_round', 'w') as f:
                f.write('%d\n' % round_idx)

    else:

        # Generate predictions
        num_samples = opt.iters['valid'] * opt.batchsize
        split = opt.predict
        idxs = opt.idx_ref[split]
        num_samples = idxs.shape[0]

        pred_dims = {k:[int(d) for d in pred[k].shape[1:]] for k in pred}
        final_preds = {k:np.zeros((num_samples, *pred_dims[k])) for k in pred}
        idx_ref = np.zeros(num_samples)
        flag_val = False

        print("Generating predictions...")
        loader.start_epoch(sess, split, train_flag, num_samples, flag_val=flag_val, in_order=True)

        for step in tqdm(range(num_samples // opt.batchsize), ascii='True'):
            tmp_idx, tmp_pred = sess.run([sample_idx, pred], feed_dict={train_flag:flag_val})
            i_ = [(step + i)*opt.batchsize for i in range(2)]
            
            idx_ref[i_[0]:i_[1]] = tmp_idx.flatten()
            for k,v in tmp_pred.items(): final_preds[k][i_[0]:i_[1]] = v

        with h5py.File('exp/%s/%s_preds.h5' % (opt.exp_id, split), 'w') as f:
            f['idx'] = idx_ref.astype(int)
            for k,v in final_preds.items(): f[k] = v

if __name__ == '__main__':
    main()
