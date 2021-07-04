from argparse import ArgumentParser

import torch
import numpy as np

from utils.train_utils import evaluate_summary, generate_summary
from utils.datasets import TVSum, SumMe
from utils.misc import write_to_results

def eval_random(dataset, augment, fold):
    if dataset == 'tvsum':
                loader = torch.utils.data.DataLoader(dataset=TVSum(split='test', augment=augment, fold_id=fold), shuffle=True)
    if dataset == 'summe':        
                loader = torch.utils.data.DataLoader(dataset=SumMe(split='test', augment=augment, fold_id=fold), shuffle=True)

    metric = dataset.split('_')[0]
    eval_metric = 'max' #if metric == 'tvsum' else 'max'

    fms = []
    video_scores = []

    for key_idx, (input, target, info, key) in enumerate(loader):
        scores = np.random.random(len(target[0])) # Random importance score generator

        cps = info['change_points'][...].cpu().detach().numpy()[0]
        num_frames = info['n_frames'][...].cpu().detach().numpy()[0]
        nfps = info['n_frame_per_seg'][...].cpu().detach().numpy().tolist()[0]
        positions = info['picks'][...].cpu().detach().numpy()[0]
        user_summary = info['user_summary'][...].cpu().detach().numpy()[0]

        machine_summary = generate_summary(scores, cps, num_frames, nfps, positions)
        fm, _, _ = evaluate_summary(machine_summary, user_summary, eval_metric)
        fms.append(fm)

        # Reporting & logging
        video_scores.append([key_idx + 1, key, "{:.1%}".format(fm)])
            
    mean_fm = np.mean(fms)
    return mean_fm, video_scores

parser = ArgumentParser(description='run experiment with randomized scores')
parser.add_argument('--experiment_name', type=str, help = 'Name of experiment/model used', default='noname')
parser.add_argument('--fold', nargs='+', type=int, help = 'Specify a fold (or multiple folds) to train on', default = [0,1,2,3,4])
parser.add_argument('--dataset', type=str, help='Dataset to be used', default='tvsum')
parser.add_argument('--augment', type=bool, help='Augment the given dataset', default=False)
args = parser.parse_args()


if __name__ == "__main__":
    fold_f_scores = []
    for i in args.fold:
        val_fscore, video_scores = eval_random(args.dataset, args.augment, i)
        fold_f_scores.append(val_fscore)

    results = {
        'name': args.experiment_name, 
        'dataset': args.dataset, 
        'mean_f_score': sum(fold_f_scores)/len(args.fold), 
        'epochs': '-', 
        'augmented': args.augment, 
        'fold_f_scores' : fold_f_scores,
        'elapsed_time' : '-',
    }
    
    print(results)
    write_to_results(results, 'results.json')




