import math
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import models
import torch.nn.functional as F


class Criterion(nn.Module):

    def __init__(self, opt):
        super(Criterion, self).__init__()
        self.opt = opt

        # criterions
        self.criterion_metric = opt['network']['criterions']['criterion_metric']
        self.criterion_fea = opt['network']['criterions']['criterion_fea']
        #self.criterion_contrastive = opt['network']['criterions']['criterion_contrastive']

        self.lambda_metric = opt['network']['lambdas']['lambda_metric']
        self.lambda_fea = opt['network']['lambdas']['lambda_fea']
        #self.lambda_contrastive = opt['network']['lambdas']['lambda_contrastive']

        self.metric_loss = RateDistortionLoss(lmbda=self.lambda_metric,
                                              criterion=self.criterion_metric)
        if self.criterion_fea:
            self.fea_loss = FeaLoss(lmbda=self.lambda_fea, criterion=self.criterion_fea)

        self.con_loss = ConLoss()

        # if self.criterion_contrastive:
        #     self.con_loss = ConLoss(lmbda=self.lambda_contrastive, criterion=self.criterion_contrastive)

    def forward(self, out_net, gt):
        out = {'loss': 0, 'rd_loss': 0, 'contrastive_loss': 0}

        # bpp loss and metric loss
        out_metric = self.metric_loss(out_net, gt)
        out['loss'] += out_metric['bpp_loss']
        out['rd_loss'] += out_metric['bpp_loss']
        for k, v in out_metric.items():
            out[k] = v
            if 'weighted' in k:
                out['loss'] += v
                out['rd_loss'] += v

        # fea loss
        if self.criterion_fea:
            if 'y_inter' in out_net.keys():
                out_fea = self.fea_loss(out_net['y'], out_net['y_gt'], out_net['y_inter'], out_net['y_inter_gt'])
            else:
                out_fea = self.fea_loss(out_net['y'], out_net['y_gt'])
            for k, v in out_fea.items():
                out[k] = v
                if 'weighted' in k:
                    out['loss'] += v

        # contrastive loss
        if 'y_noisy_project' in out_net.keys() and 'y_ae_project' in out_net.keys():
            out_con = self.con_loss(out_net['y_ae_project'], out_net['y_noisy_project'])
            for k, v in out_con.items():
                out[k] = v
                if 'weighted' in k:
                    out['loss'] += v

        return out


class Criterion_val(nn.Module):

    def __init__(self, opt):
        super(Criterion_val, self).__init__()
        self.opt = opt

        # criterions
        self.criterion_metric = opt['network']['criterions']['criterion_metric']
        self.criterion_fea = opt['network']['criterions']['criterion_fea']

        self.lambda_metric = opt['network']['lambdas']['lambda_metric']
        self.lambda_fea = opt['network']['lambdas']['lambda_fea']

        self.metric_loss = RateDistortionLoss(lmbda=self.lambda_metric,
                                              criterion=self.criterion_metric)
        if self.criterion_fea:
            self.fea_loss = FeaLoss(lmbda=self.lambda_fea, criterion=self.criterion_fea)

    def forward(self, out_net, gt):
        out = {'loss': 0, 'rd_loss': 0}

        # bpp loss and metric loss
        out_metric = self.metric_loss(out_net, gt)
        out['loss'] += out_metric['bpp_loss']
        out['rd_loss'] += out_metric['bpp_loss']
        for k, v in out_metric.items():
            out[k] = v
            if 'weighted' in k:
                out['loss'] += v
                out['rd_loss'] += v

        # fea loss
        if self.criterion_fea:
            if 'y_inter' in out_net.keys():
                out_fea = self.fea_loss(out_net['y'], out_net['y_gt'], out_net['y_inter'], out_net['y_inter_gt'])
            else:
                out_fea = self.fea_loss(out_net['y'], out_net['y_gt'])
            for k, v in out_fea.items():
                out[k] = v
                if 'weighted' in k:
                    out['loss'] += v

        return out


# rate distortion loss
class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=1e-2, criterion='mse'):
        super().__init__()
        self.lmbda = lmbda  ##权重
        self.criterion = criterion
        if self.criterion == 'mse':
            self.loss = nn.MSELoss()
        elif self.criterion == 'ms-ssim':
            from pytorch_msssim import ms_ssim
            self.loss = ms_ssim
        else:
            NotImplementedError('RateDistortionLoss criterion [{:s}] is not recognized.'.format(criterion))

    def forward(self, out_net, target):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W
        # bpp_loss计算
        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in out_net["likelihoods"].values()
        )

        if self.criterion == 'mse':
            out["mse_loss"] = self.loss(out_net["x_hat"], target)
            out["weighted_mse_loss"] = self.lmbda * 255 ** 2 * out["mse_loss"]
        elif self.criterion == 'ms-ssim':
            out["ms_ssim_loss"] = 1 - self.loss(out_net["x_hat"], target, data_range=1.0)
            out["weighted_ms_ssim_loss"] = self.lmbda * out["ms_ssim_loss"]

        return out


# contrastive loss
class ConLoss(nn.Module):
    def __int__(self):
        super().__init__()

    def forward(self, fea_noisy_project, fea_ae_project):
        features = torch.cat([fea_ae_project, fea_noisy_project], dim=0)
        logits, labels = info_nce_loss(features)
        conloss = crosscriterion(logits, labels)

        out = {
            'contrastive_loss': conloss,
            'weighted_con_loss': conloss * 0.5,
        }
        return out

def info_nce_loss(features):
    labels = torch.cat([torch.arange(16) for i in range(2)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    #labels = labels.unsqueeze(2).unsqueeze(3)
    labels = labels.to('cuda')

    #features = F.normalize(features, dim=1)#[32,128,64,128]
    features = F.adaptive_avg_pool2d(features, (1, 1))
    features = features.squeeze()
    similarity_matrix = torch.matmul(features, features.T)

    mask = torch.eye(labels.shape[0], dtype=torch.bool).to('cuda')
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

    # select only the negatives the negatives
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to('cuda')

    logits = logits / 0.07
    return logits, labels

crosscriterion = nn.CrossEntropyLoss()



# fea loss
class FeaLoss(nn.Module):
    def __init__(self, lmbda=1., criterion='l2'):
        super(FeaLoss, self).__init__()
        self.lmbda = lmbda
        self.criterion = criterion
        if self.criterion == 'l2':
            self.loss = nn.MSELoss()
        elif self.criterion == 'l1':
            self.loss = nn.L1Loss()
        else:
            NotImplementedError('FeaLoss criterion [{:s}] is not recognized.'.format(criterion))

    def forward(self, fea, fea_gt, fea_inter=None, fea_inter_gt=None):
        loss = self.loss(fea, fea_gt)
        if fea_inter is not None and fea_inter_gt is not None:
            loss += self.loss(fea_inter, fea_inter_gt)

        out = {
            'fea_loss': loss,
            'weighted_fea_loss': loss * self.lmbda,
        }
        return out
