import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import openood.utils.comm as comm
from openood.utils import Config

from .base_trainer import BaseTrainer
from openood.networks import get_network
import openood.trainers.utils_amp as awp
import copy


def get_oe_loss(logits_out, T, eps=1e-8):
    probs_orig = F.log_softmax(logits_out, dim=1)
    probs_soft = F.softmax(logits_out / T, dim=1).clamp(min=eps)

    num_classes = logits_out.size(1)
    uniform_dist = torch.full_like(probs_soft, 1.0 / num_classes)

    loss_align = F.kl_div(probs_orig, probs_soft, reduction='batchmean')

    loss_uniform = F.kl_div(probs_soft.log(), uniform_dist, reduction='batchmean')

    loss = loss_align + loss_uniform

    return loss




class DOETrainer(BaseTrainer):
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
        
        self.warmup = 5
        self.proxy = get_network(self.config.network).cuda()
        self.proxy_optim = torch.optim.SGD(self.proxy.parameters(), lr=1)
        self.diff = None

    def train_epoch(self, epoch_idx):
        diff = copy.deepcopy(self.diff)
        
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
            
            if epoch_idx >= self.warmup:
                gamma =  torch.Tensor([1e-1,1e-2,1e-3,1e-4])[torch.randperm(4)][0] # 31
                self.proxy.load_state_dict(self.net.state_dict())
                self.proxy.train()
                scale = torch.Tensor([1]).cuda().requires_grad_()
                x = self.proxy(data) * scale
                l_sur = (x[batch_size:].mean(1) - torch.logsumexp(x[batch_size:], dim=1)).mean()
                reg_sur = torch.sum(torch.autograd.grad(l_sur, [scale], create_graph = True)[0] ** 2)
                self.proxy_optim.zero_grad()
                reg_sur.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1)
                self.proxy_optim.step()
                if epoch_idx == self.warmup and train_step == 1:
                    diff = awp.diff_in_weights(self.net, self.proxy)
                else:
                    # diff = awp.diff_in_weights(net, proxy)
                    diff = awp.average_diff(diff, awp.diff_in_weights(self.net, self.proxy), beta = .6)

                awp.add_into_weights(self.net, diff, coeff = gamma)
            
            # forward
            logits_classifier = self.net(data)
            l_ce = F.cross_entropy(logits_classifier[:batch_size],
                                   batch['label'].cuda())

            l_oe = -(
                logits_classifier[batch_size:].mean(1) -
                torch.logsumexp(logits_classifier[batch_size:], dim=1)).mean()


            if epoch_idx >= self.warmup:
                loss = l_ce +  l_oe
            else: 
                loss = l_ce +  l_oe

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1) 
            self.optimizer.step()
            self.scheduler.step()
            
            if epoch_idx >= self.warmup:
                awp.add_into_weights(self.net, diff, coeff = - gamma)
                self.optimizer.zero_grad()
                x = self.net(data)
                l_ce = F.cross_entropy(x[:batch_size], batch['label'].cuda())
                loss = l_ce # + l_kl
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1)
                self.optimizer.step()

            # exponential moving average, show smooth values
            with torch.no_grad():
                loss_avg = loss_avg * 0.8 + float(loss) * 0.2

        metrics = {}
        metrics['epoch_idx'] = epoch_idx
        metrics['loss'] = self.save_metrics(loss_avg)
        
        self.diff = diff

        return self.net, metrics
