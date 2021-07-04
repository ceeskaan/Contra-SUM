import torch

def naive_topk(frames, scores, k):
    top_scores, top_indices = torch.topk(scores, k)
    out_frames = frames[0][top_indices] * top_scores.T
    return out_frames


