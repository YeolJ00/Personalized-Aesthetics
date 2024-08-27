import gc
import os
import time
import csv
import random
import importlib

import torch
import yaml
from pytorch_lightning import seed_everything
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler
from tqdm import tqdm

import utils.dist as dist
import utils.logger as logger
import utils.parser as parser
from evaluate import evaluate_piaa
from models.iaa import MultiModalIAA
from models.piaa import AlphaWrapper
from utils.losses import MSELoss, RankLoss

def class_from_name(module_name, class_name):
    # load the module, will raise ImportError if module cannot be loaded
    m = importlib.import_module(module_name)
    # get the class, will raise AttributeError if class cannot be found
    c = getattr(m, class_name)
    return c

def main(opt):
    dist.init_distributed_mode(opt)
    device = torch.device(opt.device)

    if not opt.force_resume:
        date = time.localtime(time.time())
        exp = f'{date.tm_mon:0>2}-{date.tm_mday:0>2}-{date.tm_hour:0>2}:{date.tm_min:0>2}-{opt.exp}'
        opt.output_dir = os.path.join(opt.output_dir, exp)
        os.makedirs(opt.output_dir, exist_ok=True)

    seed_everything(opt.seed)

    # Load models
    if opt.clip_model == "ViT-B-16":
        model_path_list = ["./work_dir/clip_B_3fc_ava.pth", "./work_dir/clip_B_3fc_aes.pth", "./work_dir/clip_B_3fc_tad.pth",
                        "./work_dir/clip_B_3fc_para.pth", "./work_dir/clip_B_3fc_koniq.pth", "./work_dir/clip_B_3fc_spaq.pth"]
    elif opt.clip_model == "ViT-L-14":
        model_path_list = ["./work_dir/clip_L_3fc_ava.pth", "./work_dir/clip_L_3fc_aes.pth", "./work_dir/clip_L_3fc_tad.pth",
                        "./work_dir/clip_L_3fc_para.pth", "./work_dir/clip_L_3fc_koniq.pth", "./work_dir/clip_L_3fc_spaq.pth"]
    else:
        raise NotImplementedError

    model_path_list = model_path_list[:opt.num_models]
    evaluate_fn = evaluate_piaa
    model = MultiModalIAA(opt, device)
    model.eval()
    torch.cuda.empty_cache()
    alpha_model = AlphaWrapper(model_path_list, model)

    # create dataset
    print("Creating dataset...")
    with open(os.path.join(opt.root_dir, opt.label_file), "r") as f:
        reader = csv.reader(f)
        lines = [i for i in reader]
    if opt.dataset_name == "REALCURDataset":
        worker = sorted(list(set([line[0] for line in lines[1:]])), 
                        key=lambda x: int(x.split('_')[-1]))[opt.worker_idx]
    else:
        worker = sorted(list(set([line[0] for line in lines[1:]])))[opt.worker_idx]
    data_list = [line for line in lines[1:] if line[0] == worker]
    
    random.shuffle(data_list)
    train_names = [data[1] for data in data_list[:opt.k]]

    dataset_class = class_from_name("dataset", opt.dataset_name)
    train_dataset = dataset_class(root_dir=opt.root_dir, label_file=opt.label_file, image_size=opt.image_size, 
                                split='train', worker=worker, train_list=train_names,)
    test_dataset = dataset_class(root_dir=opt.root_dir, label_file=opt.label_file, image_size=opt.image_size, 
                                split='test', worker=worker, train_list=train_names)

    batch_size = opt.batch_size
    if opt.distributed:
        num_tasks = dist.get_world_size()
        global_rank = dist.get_rank()
        sampler_rank = global_rank

        train_sampler = DistributedSampler(train_dataset, num_replicas=num_tasks, rank=sampler_rank, shuffle=True)

        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size, 
                                    num_workers=16, worker_init_fn=dist.seed_worker)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=0, shuffle=False)

        alpha_model = torch.nn.parallel.DistributedDataParallel(alpha_model, device_ids=[opt.gpu], find_unused_parameters=True)
        model_without_ddp = alpha_model.module
    else:
        num_tasks = 1
        global_rank = 0
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size, 
                                    num_workers=16, worker_init_fn=dist.seed_worker)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=16, 
                                    worker_init_fn=dist.seed_worker, shuffle=False)
        alpha_model = alpha_model.to(device)
        model_without_ddp = alpha_model


    # Test each model zero-shot and initialize alpha
    print("Initializing alpha...")
    srcc = []
    for model_path in model_path_list:
        _model = MultiModalIAA(opt, device='cuda')
        _model.load_state_dict(torch.load(model_path)['model'])
        _model.eval()

        # evaluate piaa on train set to initialize alphas
        print("Evaluating...")
        results = evaluate_piaa(opt, _model, train_dataloader, device='cuda')
        srcc.append(results['srcc'])
    srcc = torch.tensor(srcc)
    srcc = torch.softmax(srcc * opt.softmax_temp, dim=0) # temp temperature
    alpha_model.set_alpha(srcc)

    # free memory
    del _model

    # create optimizer
    optimizer = torch.optim.AdamW(alpha_model.parameters(), lr=opt.lr)

    # create criterion
    criterion = RankLoss(num_bins=10, device=device)
    
    # create logger
    if global_rank == 0:
        log_writer = logger.TensorboardLogger(log_dir=os.path.join(opt.output_dir, f'log_{opt.seed:02d}'))
    else:
        log_writer = None

    if dist.is_main_process():
        with open(os.path.join(opt.output_dir, "configs.yaml"), "w") as f:
            yaml.dump(parser.dict_to_nested_dict(vars(opt)), f)
    print(f"Steps: [{steps:06d}/{total_steps:06d}]")

    # Start training
    total_steps, steps, iters = opt.num_steps, 0, 0
    plcc_list, srcc_list, alpha_list = [], [], []

    alpha_model.train()
    pbar = tqdm(total=total_steps, ncols=100, disable=not dist.is_main_process())
    pbar.update(steps)
    pbar.set_description(f"[{steps:06d}/{total_steps:06d}]")
    metrics = evaluate_fn(opt, model_without_ddp.return_current_model(), test_dataloader, device, save_name=None)
    
    while steps < total_steps:
        for i, batch in enumerate(train_dataloader):
            is_update_step = (iters + 1) % opt.accumulation_steps == 0

            # Logging
            if log_writer is not None and is_update_step:
                log_writer.set_step(step=steps)

            # Forward
            images = batch["image"].to(device)
            labels = batch["MOS"].to(device)
            input_dict = {'img': images, }

            preds = alpha_model(input_dict)
            loss = criterion(preds, labels)
            loss_value = loss.item()


            loss.backward()
            if is_update_step:
                optimizer.step()
                optimizer.zero_grad()

            torch.cuda.synchronize()

            if log_writer is not None and is_update_step:
                log_writer.update(loss=loss_value, head="loss")
                
            # Save and Evaluate
            if steps % opt.save_every == 0 and is_update_step:
                metrics = evaluate_fn(opt, model_without_ddp.return_current_model(), test_dataloader, device, save_name=None)
                if log_writer is not None:
                    plcc_list.append(metrics['plcc'])
                    srcc_list.append(metrics['srcc'])
                    alpha_list.append(alpha_model.alpha_raw)
                    for k, v in metrics.items():
                        log_writer.update(**{k: v}, head="test")
                metrics = evaluate_fn(opt, model_without_ddp.return_current_model(), train_dataloader, device, save_name=None)
                if log_writer is not None:
                    for k, v in metrics.items():
                        log_writer.update(**{k: v}, head="train")
                gc.collect()

            iters += 1
            if is_update_step:
                steps += 1
                pbar.update(1)
                pbar.set_description(f"[{steps:06d}/{total_steps:06d}]")

            torch.cuda.synchronize()

            if steps >= total_steps:
                break

    # Evaluate and Save
    if dist.is_main_process():
        metrics = evaluate_fn(opt, alpha_model.return_current_model(), test_dataloader, device, save_name=f'restuls_{opt.seed:02d}')
        plcc_list.append(metrics['plcc'])
        srcc_list.append(metrics['srcc'])
        alpha_list.append(alpha_model.alpha_raw)
        if log_writer is not None:
            for k, v in metrics.items():
                log_writer.update(**{k: v}, head="test")
        metrics = evaluate_fn(opt, model_without_ddp.return_current_model(), train_dataloader, device, save_name=None)
        if log_writer is not None:
            for k, v in metrics.items():
                log_writer.update(**{k: v}, head="train")
        print(torch.mean(alpha_model.alpha_raw, dim=0).detach().cpu())
        gc.collect()

    # save average and std of metrics
    if dist.is_main_process():
        plcc_list = torch.tensor(plcc_list)
        srcc_list = torch.tensor(srcc_list)
        alpha_list.append(alpha_model.alpha_raw)
        # print max scores
        with open(os.path.join(opt.output_dir, "results.txt"), "w") as f:
            f.write(f"plcc: {torch.max(plcc_list).item()}\n")
            f.write(f"srcc: {torch.max(srcc_list).item()}\n")
        # save alpha of the best model
        best_idx = torch.argmax(srcc_list)
        best_alpha = alpha_list[best_idx]
        best_alpha_torch = torch.stack([torch.concatenate([a.view(-1) for name, a in zip(alpha_model.names, best_alpha[:, i]) if 'visual' in name]) 
                                    for i in range(6)])
        torch.save(best_alpha_torch, os.path.join(opt.output_dir, "best_alpha.pt"))


if __name__ == "__main__":
    opt = parser.parse_args()
    main(opt)

