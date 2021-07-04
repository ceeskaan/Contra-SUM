import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vasnet.vasnet import vasnet, VASNet_proj
from .bilstm import BiLSTM, bilstm_256, BiLSTM_proj
from .gru import bigru_256
from .top_k import perturbed_topk, halving_topk, naive_topk
from .metric_learner import M


class ScorePredictor(nn.Module):
    def __init__(self, model=bigru_256):
        super(ScorePredictor, self).__init__()
        self.score_predictor = model()

    def forward(self, x):
        return self.score_predictor(x)
    

class TopK(nn.Module):
    def __init__(self, proportion=0.15, topk_method='halving', verbose=False):
        super(TopK, self).__init__()

        self.proportion = proportion
        self.verbose = verbose

        if topk_method == 'halving':
            self.topk = halving_topk
        if topk_method == 'perturbed':
            self.topk = perturbed_topk
        if topk_method == 'naive':
            self.topk = naive_topk


    def forward(self, frames, scores):
        k = round(frames.shape[1] * self.proportion)
        summary = self.topk(frames, scores, k)
        

        if self.verbose:
            m = []
            # Compute top k according to torch.topk()
            for frames, scores in zip(frames, scores):
                values, ind = torch.topk(scores.unsqueeze(0), k, dim=1)
                sorted_emb_idx, sorted_score_idx = torch.sort(ind, dim=1) 
                real_summary = frames[sorted_emb_idx[0].T]
                m.append(real_summary.squeeze(0))

            real_summary = torch.stack(m)

            selected_summary = summary.detach()
            sim = torch.cosine_similarity(real_summary, selected_summary,2)
            print(f"TopK quality: {torch.mean(sim)} ")          

        return summary


class ContraSUM(nn.Module):
    def __init__(self, score_predictor=ScorePredictor(), topK=TopK(), metric_learner=M()):
        super(ContraSUM, self).__init__()
        self.score_predictor = score_predictor
        self.topk = topK
        self.metric_learner = metric_learner

    def forward(self, video, pos_clip, neg_clip, random_clip):
        scores = self.score_predictor(video)
        summary = self.topk(video, scores)
        summary = self.metric_learner(summary)

        pos = self.metric_learner(pos_clip)
        neg = self.metric_learner(neg_clip)
        rand = self.metric_learner(random_clip)

        return summary, pos, neg, rand


def nt_xent_loss(sum, pos, neg, rand, temperature):

    # Dot product of normalized vectors = cosine similarity
    sum, pos, neg, rand = F.normalize(sum), F.normalize(pos), F.normalize(neg), F.normalize(rand)
    samples = torch.cat((sum, pos, neg, rand))
    num_positives = pos.shape[0]

    # Numerator: similarity between positive clips and generated summary
    positives = sum @ pos.T
    positives = torch.cat((positives,positives),dim=1)
    positives = torch.exp(positives/temperature)

    # Denominator: similarity between the positive examples and the rest
    sim = samples @ samples.T
    mask = ~torch.eye(sim.shape[0]).bool()
    negatives = sim.masked_select(mask).view(sim.shape[0],-1)[:num_positives+1]
    negatives = (torch.exp(negatives/temperature)).sum(1)

    pattern = torch.cat((torch.tensor([num_positives]), torch.ones(num_positives))).long()
    negatives = torch.repeat_interleave(negatives, pattern, dim=0)

    # Compute loss
    loss = -torch.log(positives / negatives).mean()

    return  loss 

class ContraSUM2(nn.Module):
    def __init__(self, score_predictor=ScorePredictor(), topK=TopK(), metric_learner=M()):
        super(ContraSUM, self).__init__()

    def forward(self, video, pos_clip, neg_clip, random_clip):
        scores = self.score_predictor(video)
        summary = self.topk(video, scores)
        summary = self.metric_learner(summary)

        pos = self.metric_learner(pos_clip)
        neg = self.metric_learner(neg_clip)
        rand = self.metric_learner(random_clip)

        return summary, pos, neg, rand


'''
def batch_nt_xent_loss(out_1, out_2, temperature, eps=1e-6):
        """
            assume out_1 and out_2 are normalized
            out_1: [batch_size, dim]
            out_2: [batch_size, dim]
        """
        out_1_dist = out_1
        out_2_dist = out_2

        out_1, out_2 = F.normalize(out_1), F.normalize(out_2)
        out_1_dist, out_2_dist = F.normalize(out_1_dist), F.normalize(out_2_dist)
       

        # out: [2 * batch_size, dim]
        # out_dist: [2 * batch_size * world_size, dim]
        out = torch.cat([out_1, out_2], dim=0)
        out_dist = torch.cat([out_1_dist, out_2_dist], dim=0)

        # cov and sim: [2 * batch_size, 2 * batch_size * world_size]
        # neg: [2 * batch_size]
        cov = torch.mm(out, out_dist.t().contiguous())
        sim = torch.exp(cov / temperature)
        neg = sim.sum(dim=-1)

        # from each row, subtract e^(1/temp) to remove similarity measure for x1.x1
        row_sub = torch.Tensor(neg.shape).fill_(math.e**(1 / temperature)).to(neg.device)
        neg = torch.clamp(neg - row_sub, min=eps)  # clamp for numerical stability

        # Positive similarity, pos becomes [2 * batch_size]
        pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
        pos = torch.cat([pos, pos], dim=0)

        loss = -torch.log(pos / (neg + eps)).mean()

        return loss
        '''

import torch.nn.functional as F
import math
def batch_nt_xent_loss(out_1, out_2, temperature, neg_samples=None, eps=1e-6):
        """
            assume out_1 and out_2 are normalized
            out_1: [batch_size, dim]
            out_2: [batch_size, dim]
        """

        out_1, out_2 = F.normalize(out_1), F.normalize(out_2)

        # out: [2 * batch_size, dim]
        # out_dist: [2 * batch_size * world_size, dim]
        if neg_samples == None:
            out = torch.cat([out_1, out_2], dim=0)
            cov = torch.mm(out, out.t().contiguous())
        else:
            out = torch.cat([out_1, out_2, neg_samples], dim=0)
            cov = torch.mm(out, out.t().contiguous())[:-neg_samples.shape[0],:]

        # cov and sim: [2 * batch_size, 2 * batch_size * world_size]
        # neg: [2 * batch_size]
        sim = torch.exp(cov / temperature)
        neg = sim.sum(dim=-1)

        # from each row, subtract e^(1/temp) to remove similarity measure for x1.x1
        row_sub = torch.Tensor(neg.shape).fill_(math.e**(1 / temperature)).to(neg.device)
        neg = torch.clamp(neg - row_sub, min=eps)  # clamp for numerical stability

        # Positive similarity, pos becomes [2 * batch_size]
        pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
        pos = torch.cat([pos, pos], dim=0)

        loss = -torch.log(pos / (neg + eps)).mean()

        return loss