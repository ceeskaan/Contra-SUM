import numpy as np

from scipy import stats


def upsample(scores, n_frames, positions):
    """Upsample scores vector to the original number of frames.
    Input
      scores: (n_steps,)
      n_frames: (1,)
      positions: (n_steps, 1)
    Output
      frame_scores: (n_frames,)
    """
    frame_scores = np.zeros((n_frames), dtype=np.float32)
    if positions.dtype != int:
        positions = positions.astype(np.int32)
    if positions[-1] != n_frames:
        positions = np.concatenate([positions, [n_frames]])
    for i in range(len(positions) - 1):
        pos_left, pos_right = positions[i], positions[i+1]
        if i == len(scores):
            frame_scores[pos_left:pos_right] = 0
        else:
            frame_scores[pos_left:pos_right] = scores[i]
    return frame_scores

def generate_scores(probs, n_frames, positions):
    """Set score to every original frame of the video for comparison with annotations.
    Input
      probs: (n_steps,)
      n_frames: (1,)
      positions: (n_steps, 1)
    Output
      machine_scores: (n_frames,)
    """
    machine_scores = upsample(probs, n_frames, positions)
    return machine_scores

def evaluate_scores(machine_scores, user_scores, metric="spearmanr"):
    """Compare machine scores with user scores (keyframe-based).
    Input
      machine_scores: (n_frames,)
      user_scores: (n_users, n_frames)
    Output
      avg_corr, max_corr: (1,)
    """
    n_users, _ = user_scores.shape

    # Ranking correlation metrics
    if metric == "kendalltau":
        f = lambda x, y: stats.kendalltau(stats.rankdata(-x), stats.rankdata(-y))[0]
    elif metric == "spearmanr":
        f = lambda x, y: stats.spearmanr(stats.rankdata(-x), stats.rankdata(-y))[0]
    else:
        raise KeyError(f"Unknown metric {metric}")

    # Compute correlation with each annotator
    corrs = [f(machine_scores, user_scores[i]) for i in range(n_users)]
    
    # Mean over all annotators
    avg_corr = np.mean(corrs)
    return avg_corr


def spearman_evaluation(scores, user_scores, num_frames, positions):
    upsampled_scores = generate_scores(scores,  num_frames, positions)
    spear = evaluate_scores(upsampled_scores, user_scores, metric="spearmanr")
    return spear