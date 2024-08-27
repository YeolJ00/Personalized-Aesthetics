import torch

class EMDLoss(torch.nn.Module):
    def __init__(self, r=2, weight=None):
        super().__init__()
        self.r = r

    def forward(self, pred, target, ):
        '''
        convert pred, target into a CDF
        compute the distance between two CDFs
        '''
        # pdf to cdf
        pred_cdf = torch.cumsum(pred, dim=1)
        target_cdf = torch.cumsum(target, dim=1)


        cdf_diff = torch.mean(torch.pow(torch.abs(pred_cdf - target_cdf), self.r), dim=1) 
        loss = torch.mean(torch.pow(cdf_diff, 1/self.r))

        return loss
    

class MSELoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        loss = torch.mean(torch.pow(pred - target, 2))

        return loss
    

class PLCCLoss(torch.nn.Module):
    def __init__(self, num_bins=10, device='cuda'):
        super().__init__()
        template = torch.arange(1, num_bins+1).float()
        self.x_template = template.to(device)
        self.y_template = template.to(device)
    
    def forward(self, pred, target):
        '''
        pred: [B, C]
        target: [B, 1]
        '''
        score_means = torch.sum(pred * self.x_template, dim=1)
        label_means = torch.sum(target * self.y_template, dim=1)

        # calculate plcc
        vx = score_means - torch.mean(score_means)
        vy = label_means - torch.mean(label_means)

        plcc = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)) + 1e-8)

        # Normalize by batch size
        batch_size = pred.size(0)
        plcc = plcc / batch_size

        return 1 - plcc
    
class RankLoss(torch.nn.Module):
    def __init__(self, num_bins=10, device='cuda'):
        super().__init__()
        template = torch.arange(1, num_bins+1).float()
        self.x_template = template.to(device)

        self.criteria = torch.nn.CrossEntropyLoss()

    def forward(self, pred, target):
        '''
        Follow the Bradely-Terry model to calculate the probability of each pair
        score_means: [B, 10]
        label_means: [B,]
        '''
        B = pred.shape[0]
        if B % 2 != 0:
            score_means = torch.sum(pred[:-1] * self.x_template, dim=1)
            label_means = target[:-1]
            B = B - 1
        else:
            score_means = torch.sum(pred * self.x_template, dim=1)
            label_means = target

        # pair each sample with each other
        score_means_1 = score_means[:B//2]
        score_means_2 = score_means[B//2:]
    
        label_means_1 = label_means[:B//2]
        label_means_2 = label_means[B//2:]

        # calculate the probability of each pair
        rank_logit = torch.stack([score_means_1, score_means_2], dim=1)

        # make one-hot label
        rank_label = torch.zeros_like(rank_logit)
        rank_label[:, 0] = label_means_1 > label_means_2
        rank_label[:, 1] = label_means_1 < label_means_2

        loss = self.criteria(rank_logit, rank_label)

        return loss

