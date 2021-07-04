import torch
import numpy as np
import h5py
from argparse import ArgumentParser
import pickle

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

from utils.datasets import SumMe, SumMe_Batch, TVSum, TVSum_Batch
from utils.train_utils import eval, weights_init
from utils.spearman import upsample, generate_scores, evaluate_scores

from models.gru import gru_256, GRU_proj
from models.contrasum import ScorePredictor, TopK, ContraSUM, batch_nt_xent_loss
from models.projection import M
from models.vasnet.vasnet import vasnet, VASNet_proj

parser = ArgumentParser(description='Train a model for one or multiple folds')
parser.add_argument('--epochs', type=int, help='Maximum amount of Epochs', default=1000)
parser.add_argument('--batch_size', type=int, help='Batch size', default=32)
parser.add_argument('--topk', type=str, help='TopK method', default='halving')
parser.add_argument('--proportion', type=float, help='Proportion of video chosen as top K', default=0.3)
parser.add_argument('--temperature', type=float, help='Temperature in NT-Xent loss', default=0.5)
parser.add_argument('--experiment_name', type=str, help='Name of experiment/model used', default='noname')
parser.add_argument('--fold', nargs='+', type=int, help='Specify a fold (or multiple folds) to train on', default=[0])
parser.add_argument('--dataset', type=str, help='Dataset to be used', default='tvsum')
parser.add_argument('--augment', type=bool, help='Augment the given dataset', default=False)
parser.add_argument('--learning_rate', type=float, help='Training learning rate', default=1e-4)
parser.add_argument('--weight_decay', type=float, help='Weight decay used during training', default=0)
parser.add_argument('--seed', type=float, help='Random seed of experiment', default=12345)
args = parser.parse_args()



if __name__ == "__main__":

    # Load data
    if args.dataset == 'tvsum':
        train_loader = torch.utils.data.DataLoader(dataset=TVSum_Batch(split='train', fold_id=0, augment=False), batch_size=args.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(dataset=TVSum(split='test', augment=False, fold_id=0), shuffle=False)

    if args.dataset == 'summe':
        train_loader = torch.utils.data.DataLoader(dataset=SumMe_Batch(split='train', fold_id=0, augment=False), batch_size=args.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(dataset=SumMe(split='test', augment=False, fold_id=0), shuffle=False)
        
    scorepredictor = gru_256
    metriclearner = GRU_proj()

    
    device = torch.device('cpu')
    model = ContraSUM(score_predictor=ScorePredictor(scorepredictor), topK=TopK(args.proportion, args.topk), metric_learner =metriclearner)

    gpu = torch.cuda.is_available()
    print(gpu)
    device = torch.device('cuda' if gpu else 'cpu')

    if gpu:
        model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    spearman, train_loss, val_fscore = [],[],[]
    for epoch in range(args.epochs):
        avg_loss = []
        for i in train_loader:
            model.train()
            scores = model.score_predictor(i)
            summary = model.topk(i, scores.squeeze(2))

            summary = model.metric_learner(summary)
            full_video = model.metric_learner(i)

            loss = batch_nt_xent_loss(summary, full_video, args.temperature)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            avg_loss.append(float(loss))
        
        val, spear, _ = eval(model.score_predictor, test_loader, device, 'tvsum')

        spearman.append(spear)
        train_loss.append(np.mean(avg_loss))
        val_fscore.append(val)

        print('Loss:', np.mean(avg_loss))
        print('Val:', val) 


     # Save training_info
        mylist = [train_loss, spearman, val_fscore]
        with open(f'{args.experiment_name}.pkl', 'wb') as f:
            pickle.dump(mylist, f)

            