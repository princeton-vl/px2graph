import numpy as np
import scipy.misc
import h5py
import os

from px2graph.data.genome.driver import local as vg

data_dir = os.path.dirname(__file__)
class_labels, relationships = [], []
with open(data_dir + '/sorted_objects.txt', 'r') as f:
    for line in f: class_labels += [line[:-1]]
with open(data_dir + '/sorted_predicates.txt', 'r') as f:
    for line in f: relationships += [line[:-1]]

# Flip reference - this is used to swap potential relationships that would be
# affected by flipping an image (e.g. on left of) but this isn't necessary
# with the current set of predicates that we are training on.
rel_flip_ref = np.arange(len(relationships))

# Load image info
img_info = vg.GetAllImageData()
# Remove images that are corrupted? according to scene-graph-TF-release
bad_idxs = [4616, 4615, 1721, 1591]
for idx in bad_idxs: del img_info[idx]

max_objs = 50
max_rels = 50           # (arbitrary) max number of rels for one object
max_total_rels = 200    # max total number of rels in single example
num_cats = len(class_labels)
num_rels = len(relationships)
num_examples = len(img_info)

# Load preprocessed data
with h5py.File(data_dir + '/VG-SGG.h5','r') as f:
    bb1024 = f['boxes_1024'][:]
    obj_idxs = [f['img_to_first_box'][:], f['img_to_last_box'][:]]
    rel_idxs = [f['img_to_first_rel'][:], f['img_to_last_rel'][:]]
    obj_labels = f['labels'][:]
    rel_labels = f['predicates'][:]
    rel_sbj_obj = f['relationships'][:]
    train_val_split = f['split'][:]

# Convert from center, width, height to x1, y1, x2, y2
bb1024[:,:2] = bb1024[:,:2] - bb1024[:,2:] / 2
bb1024[:,2:] = bb1024[:,:2] + bb1024[:,2:]

# RPN proposal info (only loaded if opt.use_rpn is set)
im_scales, im_to_roi_idx, num_rois, rpn_rois, rpn_scores = [None]*5

def setup_val_split(opt):
    ref_idxs = np.arange(num_examples)

    valid_idxs = np.loadtxt(data_dir + '/valid_ids.txt', dtype=int)
    valid_mask = np.ones(num_examples, bool)
    valid_mask[valid_idxs] = 0

    train_idxs = ref_idxs[(train_val_split == 0) * valid_mask]
    test_idxs = ref_idxs[train_val_split == 2]

    return train_idxs, valid_idxs, test_idxs

def initialize(opt):
    global im_scales, im_to_roi_idx, num_rois, rpn_rois, rpn_scores
    opt.idx_ref = {}
    opt.idx_ref['train'], opt.idx_ref['valid'], opt.idx_ref['test'] = setup_val_split(opt)

    if opt.use_rpn:
        print("Loading region proposals")
        with h5py.File(data_dir + '/proposals.h5', 'r') as roi:
            im_scales = roi['im_scales'][:]
            im_to_roi_idx = roi['im_to_roi_idx'][:]
            num_rois = roi['num_rois'][:]
            rpn_rois = roi['rpn_rois'][:]
            rpn_scores = roi['rpn_scores'][:]

def get_id(idx):
    return img_info[idx].id

def load_image(idx):
    return scipy.misc.imread(data_dir + '/VG_100K/%d.jpg' % get_id(idx), mode='RGB')

def image_dims(idx):
    return img_info[idx].height, img_info[idx].width

def get_rpn_proposals(idx):
    im_h, im_w = image_dims(idx)
    tmp_idx = im_to_roi_idx[idx]
    tmp_rois = rpn_rois[tmp_idx:tmp_idx+num_rois[idx]] * max(im_w, im_h) / 1024
    tmp_scores = rpn_scores[tmp_idx:tmp_idx+num_rois[idx]]
    return tmp_rois, tmp_scores

def get_graph(idx):
    sg = {'obj_class':[], 'obj_bbox':None, 'rel_class':[], 'rel_sbj':None, 'rel_obj':None}
    o_idx = [obj_idxs[0][idx], obj_idxs[1][idx] + 1]
    r_idx = [rel_idxs[0][idx], rel_idxs[1][idx] + 1]
    im_h, im_w = image_dims(idx)
    if o_idx[0] != -1:
        sg['obj_class'] = (obj_labels[o_idx[0]:o_idx[1]] - 1).flatten().tolist()
        sg['obj_bbox'] = bb1024[o_idx[0]:o_idx[1]].reshape((-1,2,2))
        sg['obj_bbox'] = sg['obj_bbox'] * max(im_w, im_h) / 1024
    if r_idx[0] != -1:
        sg['rel_class'] = (rel_labels[r_idx[0]:r_idx[1]] - 1).flatten().tolist()
        tmp_so = rel_sbj_obj[r_idx[0]:r_idx[1]]
        sg['rel_sbj'] = tmp_so[:,0] - o_idx[0]
        sg['rel_obj'] = tmp_so[:,1] - o_idx[0]
    return sg

def preload_sample_info(idx):
    return get_graph(idx)

def num_objects(idx, sg=None):
    if sg is None: sg = get_graph(idx)
    return len(sg['obj_class'])

def get_obj_classes(idx, sg=None):
    if sg is None: sg = get_graph(idx)
    return sg['obj_class']

def get_bboxes(idx, sg=None):
    # Return num_objs x 2 x 2 tensor
    # objs x (upper left, bottom right) x (x, y)
    if sg is None: sg = get_graph(idx)
    if num_objects(idx, sg) > 0: return sg['obj_bbox']
    else: np.zeros((1,2,2))

def get_rels(idx, sg=None):
    if sg is None: sg = get_graph(idx)
    num_objs = num_objects(idx, sg)
    if num_objs == 0: return []
    # Loop through and get tuples
    total_rel_count = 0
    all_rels = [[] for _ in range(num_objs)]
    for i,p_idx in enumerate(sg['rel_class']):
        s_idx = sg['rel_sbj'][i]
        o_idx = sg['rel_obj'][i]
        if (not [o_idx, p_idx] in all_rels[s_idx]
            and len(all_rels[s_idx]) < max_rels
            and total_rel_count < max_total_rels):
            all_rels[s_idx] += [[o_idx, p_idx]]
            total_rel_count += 1
    return all_rels
    
def get_gt_triplets(idx, sg=None):
    if sg is None: sg = get_graph(idx)
    obj_bboxes = get_bboxes(idx, sg).reshape((-1,4))
    obj_classes = get_obj_classes(idx, sg)
    num_rels = len(sg['rel_class'])
    gt_triplets = np.zeros((num_rels,3))
    gt_triplet_boxes = np.zeros((num_rels,8))

    for i in range(num_rels):
        s_, o_ = sg['rel_sbj'][i], sg['rel_obj'][i]
        gt_triplets[i] = [obj_classes[s_], sg['rel_class'][i], obj_classes[o_]]
        gt_triplet_boxes[i,:4] = obj_bboxes[s_]
        gt_triplet_boxes[i,4:] = obj_bboxes[o_]

    return gt_triplets, gt_triplet_boxes
