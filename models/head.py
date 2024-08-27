import torch
import torch.nn as nn

class AestheticScorePredictor(nn.Module):
    def __init__(self, opt, feature_dims, hidden_dim=512, out_channels=10,):
        super().__init__()

        self.hidden_dim = hidden_dim

        self.model_type = opt.model_type
        if self.model_type == '1fc':
            self.feature_dim = feature_dims[0]
            self.projection = nn.Sequential(
                nn.Linear(self.feature_dim, out_channels),
                nn.Dropout(p=0.5),
                nn.Sigmoid(),
            )
        elif self.model_type == '3fc':
            self.feature_dim = feature_dims[0]
            self.projection = nn.Sequential(
                nn.Linear(self.feature_dim, hidden_dim), 
                nn.GELU(), 
                nn.Dropout(p=0.5),
                nn.Linear(hidden_dim, out_channels),
                nn.Sigmoid(),
            )
        elif self.model_type == '5fc':
            self.feature_dim = feature_dims[0]
            self.projection = nn.Sequential(
                nn.Linear(self.feature_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(p=0.5),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(p=0.5),
                nn.Linear(hidden_dim, out_channels),
                nn.Sigmoid(),
            )
        else:
            raise NotImplementedError("Only mlp heads are supported")

    def forward(self, x):

        x = self.projection(x)
        x = x / torch.sum(x, dim=1, keepdim=True)
        return x
