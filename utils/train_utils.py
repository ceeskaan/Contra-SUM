import h5py
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import warnings

from utils.knapsack import knapsack_ortools
from utils.spearman import spearman_evaluation

# Weight initialization that authors used for VASNet
def weights_init(m):
    classname = m.__class__.__name__
    if classname == 'Linear':
        init.xavier_uniform_(m.weight, gain=1)
        if m.bias is not None:
            init.constant_(m.bias, 0.1)

# Training simple/supervised video summarization model
def train(model, optimizer, loader, device):
    criterion = nn.MSELoss().to(device)
    model.train()
    avg_loss = []

    for input, target, _, _ in loader:
        input, target = input.to(device), target.to(device)
        scores = model(input)
        loss = criterion(scores, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_loss.append(float(loss))

    avg_loss = np.array(avg_loss)
    return avg_loss

# Evaluation function for all methods
def eval(model, loader, device, dataset):
    model.eval()

    metric = dataset.split('_')[0]
    eval_metric = 'avg' if metric == 'tvsum' else 'max'

    if metric == 'tvsum':
            f = h5py.File('summarizer_dataset_tvsum_google_pool5 (1).h5', 'r')

    fms = []
    spearman_corr = []
    video_scores = []
    with torch.no_grad():
        for key_idx, (input, target, info, key) in enumerate(loader):
            input = input.to(device)
            scores = model(input)
            scores = scores[0].detach().cpu().numpy()
            
            cps = info['change_points'][...].cpu().detach().numpy()[0]
            num_frames = info['n_frames'][...].cpu().detach().numpy()[0]
            nfps = info['n_frame_per_seg'][...].cpu().detach().numpy().tolist()[0]
            positions = info['picks'][...].cpu().detach().numpy()[0]
            user_summary = info['user_summary'][...].cpu().detach().numpy()[0]

            machine_summary = generate_summary(scores, cps, num_frames, nfps, positions)
            fm, _, _ = evaluate_summary(machine_summary, user_summary, eval_metric)
            fms.append(fm)

            if metric == 'tvsum':
                spear = spearman_evaluation(scores, f[key[0]]['user_scores'][...], num_frames, positions)
                spearman_corr.append(spear)

            # Reporting & logging
            video_scores.append([key_idx + 1, key, "{:.1%}".format(fm)])
           
    mean_fm = np.mean(fms)
    return mean_fm, np.mean(spearman_corr), video_scores, spearman_corr


"""
Courtesy of KaiyangZhou: https://github.com/KaiyangZhou/pytorch-vsumm-reinforce
And additional modifications by Jiri Fajtl: https://github.com/ok1zjf/VASNet
"""

def generate_summary(ypred, cps, n_frames, nfps, positions, proportion=0.15, method='knapsack'):
    """Generate keyshot-based video summary i.e. a binary vector.
    Args:
    ---------------------------------------------
    - ypred: predicted importance scores.
    - cps: change points, 2D matrix, each row contains a segment.
    - n_frames: original number of frames.
    - nfps: number of frames per segment.
    - positions: positions of subsampled frames in the original video.
    - proportion: length of video summary (compared to original video length).
    - method: defines how shots are selected, ['knapsack', 'rank'].
    """
    n_segs = cps.shape[0]
    frame_scores = np.zeros((n_frames), dtype=np.float32)
    if positions.dtype != int:
        positions = positions.astype(np.int32)
    if positions[-1] != n_frames:
        positions = np.concatenate([positions, [n_frames]])
    for i in range(len(positions) - 1):
        pos_left, pos_right = positions[i], positions[i+1]
        if i == len(ypred):
            frame_scores[pos_left:pos_right] = 0
        else:
            frame_scores[pos_left:pos_right] = ypred[i]

    seg_score = []
    for seg_idx in range(n_segs):
        start, end = int(cps[seg_idx,0]), int(cps[seg_idx,1]+1)
        scores = frame_scores[start:end]
        seg_score.append(float(scores.mean()))

    limits = int(math.floor(n_frames * proportion))

    if method == 'knapsack':
        picks = knapsack_ortools(seg_score, nfps, n_segs, limits)
    elif method == 'rank':
        order = np.argsort(seg_score)[::-1].tolist()
        picks = []
        total_len = 0
        for i in order:
            if total_len + nfps[i] < limits:
                picks.append(i)
                total_len += nfps[i]
    else:
        raise KeyError("Unknown method {}".format(method))

    summary = np.zeros((1), dtype=np.float32) # this element should be deleted
    for seg_idx in range(n_segs):
        nf = nfps[seg_idx]
        if seg_idx in picks:
            tmp = np.ones((nf), dtype=np.float32)
        else:
            tmp = np.zeros((nf), dtype=np.float32)
        summary = np.concatenate((summary, tmp))

    summary = np.delete(summary, 0) # delete the first element
    
    return summary

def evaluate_summary(machine_summary, user_summary, eval_metric='avg'):
    """Compare machine summary with user summary (keyshot-based).
    Args:
    --------------------------------
    machine_summary and user_summary should be binary vectors of ndarray type.
    eval_metric = {'avg', 'max'}
    'avg' averages results of comparing multiple human summaries.
    'max' takes the maximum (best) out of multiple comparisons.
    """
    machine_summary = machine_summary.astype(np.float32)
    user_summary = user_summary.astype(np.float32)
    n_users, n_frames = user_summary.shape

    # binarization
    machine_summary[machine_summary > 0] = 1
    user_summary[user_summary > 0] = 1

    if len(machine_summary) > n_frames:
        machine_summary = machine_summary[:n_frames]
    elif len(machine_summary) < n_frames:
        zero_padding = np.zeros((n_frames - len(machine_summary)))
        machine_summary = np.concatenate([machine_summary, zero_padding])

    f_scores = []
    prec_arr = []
    rec_arr = []

    for user_idx in range(n_users):
        gt_summary = user_summary[user_idx, :]
        overlap_duration = (machine_summary * gt_summary).sum()
        precision = overlap_duration / (machine_summary.sum() + 1e-8)
        recall = overlap_duration / (gt_summary.sum() + 1e-8)
        if precision == 0 and recall == 0:
            f_score = 0.
        else:
            f_score = (2 * precision * recall) / (precision + recall)
        f_scores.append(f_score)
        prec_arr.append(precision)
        rec_arr.append(recall)

    if eval_metric == 'avg':
        final_f_score = np.mean(f_scores)
        final_prec = np.mean(prec_arr)
        final_rec = np.mean(rec_arr)
    elif eval_metric == 'max':
        final_f_score = np.max(f_scores)
        max_idx = np.argmax(f_scores)
        final_prec = prec_arr[max_idx]
        final_rec = rec_arr[max_idx]

    return final_f_score, final_prec, final_rec
