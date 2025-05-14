import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import openood.utils.comm as comm
from openood.utils import Config
import math
from .base_trainer import BaseTrainer


import torch
import torch.nn.functional as F
import numpy as np

# 全量训练版本
def get_oe_loss(logits_out, T, eps=1e-8):
    probs_orig = F.log_softmax(logits_out, dim=1)
    probs_soft = F.softmax(logits_out / T, dim=1).clamp(min=eps)

    num_classes = logits_out.size(1)
    uniform_dist = torch.full_like(probs_soft, 1.0 / num_classes)

    loss_align = F.kl_div(probs_orig, probs_soft, reduction='batchmean')

    loss_uniform = F.kl_div(probs_soft.log(), uniform_dist, reduction='batchmean')

    loss = loss_align + loss_uniform

    return loss

# Finetune 版本
def get_oe_loss(logits_out, T, eps=1e-8):
    probs_orig = F.log_softmax(logits_out, dim=1)
    probs_soft = F.softmax(logits_out / T, dim=1).clamp(min=eps).detach()
    loss_align = F.kl_div(probs_orig, probs_soft, reduction='batchmean')
    loss = loss_align
    return loss


class OETrainer(BaseTrainer):
    def __init__(
        self,
        net: nn.Module,
        train_loader: DataLoader,
        train_unlabeled_loader: DataLoader,
        config: Config,
    ) -> None:
        super().__init__(net, train_loader, config)
        self.train_unlabeled_loader = train_unlabeled_loader
        self.lambda_oe = config.trainer.lambda_oe
        # self.avg_probs_id = None
        self.alpha = 1

    def train_epoch(self, epoch_idx):
        self.net.train()  # enter train mode

        loss_avg = 0.0
        train_dataiter = iter(self.train_loader)

        if self.train_unlabeled_loader:
            unlabeled_dataiter = iter(self.train_unlabeled_loader)

        for train_step in tqdm(range(1,
                                     len(train_dataiter) + 1),
                               desc='Epoch {:03d}: '.format(epoch_idx),
                               position=0,
                               leave=True,
                               disable=not comm.is_main_process()):
            batch = next(train_dataiter)

            try:
                unlabeled_batch = next(unlabeled_dataiter)
            except StopIteration:
                unlabeled_dataiter = iter(self.train_unlabeled_loader)
                unlabeled_batch = next(unlabeled_dataiter)

            data = torch.cat((batch['data'], unlabeled_batch['data'])).cuda()
            batch_size = batch['data'].size(0)

            # forward
            logits_classifier = self.net(data)
            loss = F.cross_entropy(logits_classifier[:batch_size],
                                   batch['label'].cuda())
            
            
            # loss_oe = get_oe_loss(logits_classifier[batch_size:], 2.5)
            loss_oe = -(
                logits_classifier[batch_size:].mean(1) -
                torch.logsumexp(logits_classifier[batch_size:], dim=1)).mean()
            # alpha = self.alpha * (1.0 - (1 + math.cos(math.pi * (epoch_idx + 1) / 100)) / 2)
            # alpha = self.alpha * min(1.0, (epoch_idx + 1) / 100)
            # alpha = self.alpha * (1 - math.exp(-1/35 * epoch_idx))
            
            loss += self.lambda_oe * loss_oe
            # loss += alpha * loss_oe

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            # exponential moving average, show smooth values
            with torch.no_grad():
                loss_avg =  float(loss) 

        metrics = {}
        metrics['epoch_idx'] = epoch_idx
        metrics['loss'] = self.save_metrics(loss_avg)

        return self.net, metrics
