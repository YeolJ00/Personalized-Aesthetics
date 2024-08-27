import torch
import torch.nn as nn
from models.iaa import MultiModalIAA


class MultiModalPIAA(torch.nn.Module):
    '''
    IAA model with CLIP features
    defines a backbone and a head
    '''
    def __init__(self, opt, device) -> None:
        super().__init__()

        self.iaa_model = MultiModalIAA(opt, device)
        
        if opt.resume:
            print("Loading checkpoint from {}".format(opt.resume))
            state_dict = torch.load(opt.resume, map_location='cpu')['model']
            self.iaa_model.load_state_dict(state_dict=state_dict, strict=False)

        hidden_dim = self.iaa_model.backbone.visual.output_dim
        self.finetune_layer = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim), 
                nn.GELU(), 
                nn.Dropout(p=0.5),
                nn.Linear(hidden_dim, 5),
                nn.Sigmoid(),
            )
        
    def forward(self, x):
        features = self.iaa_model.extract_feature(x)

        pred = self.finetune_layer(features)
        pred = pred / torch.sum(pred, dim=1, keepdim=True)

        return pred

# Utilities to make nn.Module functional
def del_attr(obj, names):
    if len(names) == 1:
        delattr(obj, names[0])
    else:
        del_attr(getattr(obj, names[0]), names[1:])

def set_attr(obj, names, val):
    if len(names) == 1:
        setattr(obj, names[0], val)
    else:
        set_attr(getattr(obj, names[0]), names[1:], val)

def load_weights(mod, names, params):
    for name, p in zip(names, params):
        set_attr(mod, name.split("."), p)

def make_functional(mod):
    orig_params = tuple(mod.parameters())
    # Remove all the parameters in the model
    names = []
    for name, p in list(mod.named_parameters()):
        del_attr(mod, name.split("."))
        names.append(name)
    return orig_params, names

class AlphaWrapper(torch.nn.Module):
    '''
    Wrapper for alpha coefficient mixing
    Parameters are kept in cpu and detached, only during forward pass they are moved to device
    '''
    def __init__(self, model_path_list, model, device='cuda'):
        super(AlphaWrapper, self).__init__()
        self.model_path_list = model_path_list
        self.model = model
        self.device = device
        
        # get names of parameters
        original_params, names = make_functional(self.model)
        original_params = tuple(p.detach().requires_grad_().cpu() for p in original_params)
        self.original_params = original_params
        self.names = names

        ralpha = torch.zeros(len(original_params), len(model_path_list))
        ralpha[:, 0] = 1.0 # initialize with the first model
        
        self.alpha_raw = torch.nn.Parameter(ralpha)

        sds = []
        for i, model_path in enumerate(model_path_list):
            state_dict_i = torch.load(model_path, map_location='cpu')['model']
            sds.append(state_dict_i)
        paramlist = [tuple(v.detach().requires_grad_().cpu() for _, v in sd.items()) for i, sd in enumerate(sds)]

        breakpoint()

        self.task_vectors = []
        for param in paramlist:
            params = [p - p_0 for p, p_0 in zip(param, original_params)]
            self.task_vectors.append(tuple(params))

    def set_alpha(self, alphas):
        '''
        alphas: list of len=len(model_path_list)
        '''
        ralpha = torch.zeros(len(self.alpha_raw[:,0]), len(self.alpha_raw[0, :]))
        for i, alpha in enumerate(alphas):
            ralpha[:, i] = alpha
        self.alpha_raw = torch.nn.Parameter(ralpha)

    def return_current_model(self):
        merged_params = [sum(tuple([self.original_params[i]] 
                                   + [pi * alphai for pi, alphai in zip(ps, self.alpha_raw[i, :].cpu())])) 
                         for i, ps in enumerate(zip(*self.task_vectors))]
        merged_params = tuple(p.to(self.device) for p in merged_params)
        load_weights(self.model, self.names, merged_params)

        return self.model
    
    def forward(self, x):
        merged_params = [sum(tuple([self.original_params[i]] 
                                   + [pi * alphai for pi, alphai in zip(ps, self.alpha_raw[i, :].cpu())])) 
                         for i, ps in enumerate(zip(*self.task_vectors))]
        merged_params = tuple(p.to(self.device) for p in merged_params)
        load_weights(self.model, self.names, merged_params)

        return self.model(x)
    