import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm
import os

import utils.dist as dist

@torch.no_grad()
def evaluate_piaa(opt, model, dataloader, device, save_name=None):
    model.eval()
    bins = 10
    with torch.no_grad():
        all_preds = []
        all_labels, all_counts = [], []
        img_names = []
        pbar = tqdm(total=len(dataloader), ncols=100, disable=not dist.is_main_process())
        pbar.set_description('Evaluating')
        for i, batch in enumerate(dataloader):
            images = batch["image"].to(device)
            features = None
            input_dict = {'img': images,
                          'features': features}
            label_mos = batch['MOS']

            preds = model(input_dict)

            all_preds.append(preds.detach().cpu().numpy())
            all_labels.append(label_mos.numpy())
            img_names += batch['image_name']
            pbar.update(1)

        all_preds = np.concatenate(all_preds, axis=0) # (N, 5))
        all_labels = np.concatenate(all_labels, axis=0) # (N,)

    assert len(all_preds) == len(all_labels)
    all_preds = np.dot(all_preds, np.arange(1, 11))
    
    plcc, _ = pearsonr(all_preds, all_labels)
    srcc, _ = spearmanr(all_preds, all_labels)

    torch.cuda.synchronize()

    # csv output
    if opt.save_csv and save_name is not None:
        import csv
        with open(os.path.join(opt.output_dir, f'{save_name}.csv'), 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['plcc', plcc, 'srcc', srcc])
            writer.writerow(['img_name', 'score_mean', 'label_mean'])
            for _score_mean, _label_mean, _img_name in zip(all_preds, all_labels, img_names):
                _img_name = _img_name.split('/')[-1]
                writer.writerow([_img_name, _score_mean, _label_mean])

    model.train()
    return {'plcc': plcc, 'srcc': srcc}
