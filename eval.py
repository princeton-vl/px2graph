import numpy as np
import h5py
import sys, os
from easydict import EasyDict as edict
from tqdm import tqdm
from graphviz import Digraph

import px2graph
from px2graph.util import calc, img
import px2graph.data.genome.ref as ds

exp_dir = os.path.join(os.path.dirname(px2graph.__file__),'exp')

def load_predictions(pred_file):
    pred_file = os.path.join(exp_dir, pred_file + '.h5')
    print("Loading predictions from:", pred_file)

    # Load raw data
    with h5py.File(pred_file,'r') as f:
        idxs = f['idx'][:].astype(int)
        objs_ = f['objs'][:]
        rels_ = f['rels'][:]
        n_objs = f['n_objs'][:].astype(int)
        n_rels = f['n_rels'][:].astype(int)

    objs = edict({'score':objs_[:,:,0],
                  'class_idx':objs_[:,:,1:6].astype(int),
                  'class_score':objs_[:,:,6:11],
                  'bbox':objs_[:,:,11:15],
                  'tag':objs_[:,:,15:-1],
                  'num':n_objs})
    rels = edict({'score':objs_[:,:,0],
                  'tuple':rels_[:,:,1:4].astype(int),
                  'num':n_rels})

    # Transform boxes from heatmap to image coordinates
    print("Processing bounding boxes...")
    for i in tqdm(range(idxs.shape[0])):
        tmp_idx = idxs[i]
        ht,wd = ds.image_dims(tmp_idx)
        c,s = [wd/2, ht/2], max(ht, wd) / 200
        for j in range(objs.num[i]):
            ul_ = objs.bbox[i,j,:2]
            br_ = objs.bbox[i,j,2:]
            ul = img.transform(ul_, c, s, [512]*2, invert=1)
            br = img.transform(br_, c, s, [512]*2, invert=1)
            objs.bbox[i,j,:2] = ul
            objs.bbox[i,j,2:] = br

    return edict({'idxs':idxs, 'objs':objs, 'rels':rels})

def process_pred(preds, idx, n=100):
    objs, rels = preds.objs, preds.rels
    result = None

    if rels.num[idx] > 0:
        n_rels = min(n, rels.num[idx])
        obj_class = objs.class_idx[idx,:objs.num[idx],0]
        obj_bbox = objs.bbox[idx,:objs.num[idx]]

        rel_scores = rels.score[idx, :n_rels]
        rel_tuples = rels.tuple[idx, :n_rels]

        s = obj_class[rel_tuples[:,0]]
        p = rel_tuples[:,1]
        o = obj_class[rel_tuples[:,2]]
        triplets = np.stack([s,p,o],1)

        s_bb = obj_bbox[rel_tuples[:,0]]
        o_bb = obj_bbox[rel_tuples[:,2]]
        triplet_boxes = np.concatenate([s_bb, o_bb], 1)
        
        result = edict({
            'objs':{'classes':obj_class, 'bbox':obj_bbox},
            'rels':{'scores':rel_scores, 'tuples':rel_tuples},
            'triplets':triplets, 'triplet_boxes':triplet_boxes
        })

    return result

def evaluate_pred(preds, idx, iou_thresh=.5, rel_filter=None):
    idxs, objs, rels = preds.idxs, preds.objs, preds.rels
    tmp_idx = idxs[idx]
    try:
        gt_triplet, gt_boxes = ds.get_gt_triplets(tmp_idx)
        if gt_triplet.shape[0] > 0:
            pred = process_pred(preds, idx)
            if pred is None:
                return 0, 0
            else:
                ar50 = calc._relation_recall(gt_triplet, pred.triplets[:50],
                                             gt_boxes, pred.triplet_boxes[:50], iou_thresh, rel_filter=rel_filter)
                ar100 = calc._relation_recall(gt_triplet, pred.triplets,
                                              gt_boxes, pred.triplet_boxes, iou_thresh, rel_filter=rel_filter)
                return ar50, ar100
        else:
            return -1, -1
    except:
        return -1, -1

def main():
    try:
        pred_dir = sys.argv[1]
    except:
        print("Please provide a prediction filename (ex: test_exp_001/train_preds)")
        print("File location is assumed to be in px2graph/exp/, no .h5 extension necessary")
        return

    preds = load_predictions(pred_dir)
    results = [[],[]]

    n_exs = len(preds.idxs)
    ar_50, ar_100 = np.zeros(n_exs), np.zeros(n_exs)

    print("Running evaluation...")
    for i in tqdm(range(n_exs)):
        ar_50[i], ar_100[i] = evaluate_pred(preds, i)
    avg_ar_50 = ar_50[ar_50 > -1].mean()
    avg_ar_100 = ar_100[ar_100 > -1].mean()

    print("R@50:", avg_ar_50)
    print("R@100:", avg_ar_100)

    tmp_pred_dir = '/'.join(pred_dir.split('/')[:-1])
    with open('exp/' + tmp_pred_dir + '/result.txt','w') as f:
        f.write('%.4f %.4f\n' % (avg_ar_50, avg_ar_100))

if __name__ == '__main__':
    main()
