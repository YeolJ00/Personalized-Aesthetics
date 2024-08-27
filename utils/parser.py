import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    # Data parameters
    parser.add_argument("--dataset_name", type=str, help="name of dataset", 
                        default="AVADataset")
    parser.add_argument("--root_dir", type=str, help="root dir of dataset",
                        default="./Datasets/AVA_dataset",)
    parser.add_argument("--label_file", type=str, help="label file of dataset",
                        default="AVA_data_official.csv",)
    parser.add_argument("--image_size", type=int, help="image size", 
                        default=384)
    parser.add_argument("--split", type=str, help="split of dataset", 
                        default="trainval",)
    
    # Experiment parameters
    parser.add_argument("--exp", type=str, help="experiment name", default="test")
    parser.add_argument('--output_dir', default='./work_dir', help='path where to save models')
    parser.add_argument("--force_resume", action='store_true', help="force resume training")
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument('--no_auto_resume', action='store_false', dest='auto_resume')
    parser.set_defaults(auto_resume=True)
    parser.add_argument("--load_model_only", action='store_true', help="load model only")

    parser.add_argument('--save_every', type=int, default=5000, metavar='N',
                        help='how many steps to wait before saving model')
    parser.add_argument('--evaluate_every', type=int, default=1000, metavar='N',
                        help='how many steps to wait before evaluating model')
    parser.add_argument('--save_csv', action='store_true', help='save csv file')
    
    parser.add_argument("--seed", type=int, default=42, help="the seed (for reproducible sampling)",)

    # Training parameters
    parser.add_argument("--batch_size",type=int,help="how many samples to produce for each given prompt. A.k.a batch size",
                        default=2,)
    parser.add_argument("--num_steps", type=int, help="number of steps to run the model for", 
                        default=100_000)
    parser.add_argument("--accumulation_steps", type=int, help="number of steps to run the model for",
                        default=1)
    
    # Personalization parameters)
    parser.add_argument("--softmax_temp", type=float, help="temperature for softmax", default=10.0)
    parser.add_argument("--num_models", type=int, help="number of models to use",
                        default=1)
    parser.add_argument("--worker_idx", type=int, help="index of worker for personalization",
                        default=0)
    parser.add_argument('--k', type=int, help='number of samples to use for personalization',
                        default=10)
    
    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='weight decay (default: 0.01)')
    parser.add_argument('--weight_decay_end', type=float, default=None, help="""Final value of the
                        weight decay. We use a cosine schedule for WD. 
                        (Set the same value with args.weight_decay to keep weight decay no change)""")
    parser.add_argument('--lr', type=float, default=1.5e-4, metavar='LR',
                        help='learning rate (default: 1.5e-4)')
    parser.add_argument('--lr_schedule', default='cosine', type=str, metavar='SCHEDULE',
                        help='LR schedule')
    parser.add_argument('--warmup_lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min_lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    parser.add_argument('--warmup_steps', type=int, default=1000, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    
    # Model parameters
    parser.add_argument("--prediction_type", type=str, help="classification or regression", default="classification")
    parser.add_argument("--loss_type", type=str, help="type of loss [EMD, MSE, ...]", default="Rank")

    parser.add_argument("--model_type", type=str, help="model type", default="3fc")
    parser.add_argument("--hidden_dim", type=int, help="hidden dimension", default=512)
    parser.add_argument("--num_classes", type=int, help="number of classes", default=14)

    # CLIP features
    parser.add_argument("--clip_model", type=str, help="clip model", default="ViT-L-14")
    parser.add_argument("--clip_pretrained", type=str, help="clip pretrained", default="datacomp_xl_s13b_b90k")


    # Feature Extraction
    parser.add_argument("--feature_dims", nargs='*', type=int, help="feature dimensions of the preprocessed features",
                        default=[512])
    parser.add_argument("--freeze_backbone", action='store_true', help="freeze backbone of the model")


    # Precision
    parser.add_argument("--precision",type=str,help="evaluate at this precision",choices=["full", "autocast"], default="autocast")
    parser.add_argument("--device",type=str,help="Device on which Stable Diffusion will be run",choices=["cpu", "cuda"], default="cuda")


    # Distributed training
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument("--cudnn_benchmark", action="store_true",
                        help="Enable cudnn benchmarking. May improve performance. (may not)")
    parser.add_argument("--ignore_warnings", action="store_true", help="ignore deprecated warnings")

    opt = parser.parse_args()
    if opt.ignore_warnings:
        import warnings
        warnings.filterwarnings("ignore", category=DeprecationWarning) 
        warnings.filterwarnings("ignore", category=UserWarning)

    return opt


def dict_to_nested_dict(opt):
    """Converts a dictionary to a nested dictionary.
    Args:
        opt (dict): Dictionary to be converted.
    Returns:
        dict: Nested dictionary.
    """
    catagory_mapping = {"dataset_name": "DATASET",
                        "root_dir": "DATASET",
                        "label_file": "DATASET",
                        "image_size": "DATASET",
                        "split" : "DATASET",
                        "semantic_tag": "DATASET",
                        "exp": "EXPERIMENT",
                        "log_dir": "EXPERIMENT",
                        "output_dir": "EXPERIMENT",
                        "force_resume" : "EXPERIMENT",
                        "resume": "EXPERIMENT",
                        "auto_resume": "EXPERIMENT",
                        "no_auto_resume": "EXPERIMENT",
                        "load_model_only" : "EXPERIMENT",
                        "save_every": "EXPERIMENT",
                        "evaluate_every": "EXPERIMENT",
                        "save_csv": "EXPERIMENT",
                        "batch_size": "TRAINING",
                        "num_steps": "TRAINING",
                        "accumulation_steps": "TRAINING",
                        "num_models": "PERSONALIZATION",
                        "worker_idx": "PERSONALIZATION",
                        "k": "PERSONALIZATION",
                        "opt": "OPTIMIZER",
                        "opt_eps": "OPTIMIZER",
                        "opt_betas": "OPTIMIZER",
                        "clip_grad": "OPTIMIZER",
                        "momentum": "OPTIMIZER",
                        "weight_decay": "OPTIMIZER",
                        "weight_decay_end": "OPTIMIZER",
                        "lr": "OPTIMIZER",
                        "warmup_lr": "OPTIMIZER",
                        "min_lr": "OPTIMIZER",
                        "warmup_steps": "OPTIMIZER",
                        "lr_schedule": "OPTIMIZER",
                        "prediction_type": "MODEL",
                        "loss_type": "MODEL",
                        "softmax_temp": "MODEL",
                        "model_type": "MODEL",
                        "hidden_dim": "MODEL",
                        "num_classes": "MODEL",
                        "clip_model": "MODEL",
                        "clip_pretrained": "MODEL",
                        "seed": "FEATURE_EXTRACTION",
                        "feature_dims": "FEATURE_EXTRACTION",
                        "freeze_backbone": "FEATURE_EXTRACTION",
                        "precision": "PRECISION",
                        "device": "PRECISION",
                        "world_size": "DISTRIBUTED",
                        "local_rank": "DISTRIBUTED",
                        "dist_on_itp": "DISTRIBUTED",
                        "dist_url": "DISTRIBUTED",
                        "cudnn_benchmark": "PRECISION",
                        "ignore_warnings": "PRECISION",
                        }
    nested_dict = {'MISC':{}}
    for key, value in opt.items():
        if key in catagory_mapping:
            if catagory_mapping[key] not in nested_dict:
                nested_dict[catagory_mapping[key]] = {}
            nested_dict[catagory_mapping[key]][key] = value
        else:
            nested_dict['MISC'][key] = value
    return nested_dict