import open_clip
import torch

from models.head import AestheticScorePredictor


class MultiModalIAA(torch.nn.Module):
    '''
    IAA model with CLIP features
    defines a backbone and a head
    '''
    def __init__(self, opt, device) -> None:
        super().__init__()
        self.backbone, _ = get_clip_model(opt.clip_model, pretrained=opt.clip_pretrained)
        feature_dim = [self.backbone.visual.output_dim]
        opt.feature_dims = [feature_dim]

        self.backbone = self.backbone.to(torch.float32)
        self.backbone = self.backbone.to(device)

        self.score_predictor = AestheticScorePredictor(opt, feature_dims=feature_dim, 
                                            hidden_dim=opt.hidden_dim,).to(device)

        if opt.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def extract_feature(self, x):
        '''
        x: dict{'img': torch.Tensor, 'caption': list[str], 'aspect_ratio': torch.Tensor}
        '''
        features = self.backbone.encode_image(x['img'])
        return features


    def forward(self, x):
        '''
        x: dict{'img': torch.Tensor, 'caption': list[str], 'aspect_ratio': torch.Tensor}
        '''
        features = self.extract_feature(x)
        pred = self.score_predictor(features)

        return pred
    

def get_clip_model(clip_model="ViT-L/14", pretrained=None):

    _clip, _, preprocess = open_clip.create_model_and_transforms(clip_model, pretrained=pretrained)
    _clip.eval()

    return _clip, preprocess
