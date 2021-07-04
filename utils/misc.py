import json
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

from sklearn.manifold import TSNE

def write_to_results(dict, result_file):
    '''
    Adds line to results json file with experiment results
    '''
    with open(result_file) as f:
        data = json.load(f)
        data.append(dict)

    with open(result_file,'w') as f:
        f.write('[\n')
        for d in data:
            f.write(json.dumps(d)) 
            if d == data[-1]:
                f.write("\n")
                break
            f.write(",\n")

        f.write(']')

def get_clip(input, length):
    idx = np.random.randint(0, len(input[0]) - length)
    input = input[0][idx:idx + length]
    return input

def mscatter(x,y, ax=None, m=None, **kw):
    import matplotlib.markers as mmarkers
    ax = ax or plt.gca()
    sc = ax.scatter(x,y,**kw)
    if (m is not None) and (len(m)==len(x)):
        paths = []
        for marker in m:
            if isinstance(marker, mmarkers.MarkerStyle):
                marker_obj = marker
            else:
                marker_obj = mmarkers.MarkerStyle(marker)
            path = marker_obj.get_path().transformed(
                        marker_obj.get_transform())
            paths.append(path)
        sc.set_paths(paths)
    return sc

def visualize(loader, model):
    figure(figsize=(12, 12), dpi=80)
    all_things, fulls, summaries = [],[],[]
    for id, (i,_,_,key) in enumerate(loader):
        model.eval()
        full = model.metric_learner(i)

        scores = model.score_predictor(i)
        summary = model.topk(i, scores.squeeze(2))
        summary = model.metric_learner(summary)

        fulls.append(full[0].tolist())
        summaries.append(summary[0].tolist())

        all_things.append([full[0].tolist(), id, '+', 200])
        all_things.append([summary[0].tolist(), id, 'x', 200])
        for j in range(50):
            all_things.append([model.metric_learner(get_clip(i, 50).unsqueeze(0))[0].tolist(), id, '.', 50])

    embeds = [i[0] for i in all_things]

    X_embedded = TSNE(n_components=2).fit_transform(embeds)
    x,y = list(zip(*X_embedded))

    labels = [i[1] for i in all_things]
    markers = [i[2] for i in all_things]
    sizes = [i[3] for i in all_things]

    mscatter(x,y, c=labels, m=markers, s=sizes, cmap='Paired')

    return fulls, summaries