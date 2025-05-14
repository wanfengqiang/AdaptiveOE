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
    probs_soft = F.softmax(logits_out / T, dim=1).clamp(min=eps).detach()

    loss_align = F.kl_div(probs_orig, probs_soft, reduction='batchmean')

    loss = loss_align

    return loss

class DALTrainer(BaseTrainer):
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
        self.gamma = 0.01

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
            batch['label'] = batch['label'].cuda()
            batch_size = batch['data'].size(0)

            x, emb = self.net.pred_emb(data)
            l_ce = F.cross_entropy(x[:batch_size], batch['label'])
            l_oe_old = - (x[batch_size:].mean(1) - torch.logsumexp(x[batch_size:], dim=1)).mean()


            emb_oe = emb[batch_size:].detach()
            emb_bias = torch.rand_like(emb_oe) * 0.0001

            iter_step = 10
            strength = 0.01
            beta = 0.5
            rho = 0.01
            args_gamma = 1
            for _ in range(iter_step):
                emb_bias.requires_grad_()
                x_aug = self.net.fc(emb_bias + emb_oe)
                l_sur = - (x_aug.mean(1) - torch.logsumexp(x_aug, dim=1)).mean()
                r_sur = (emb_bias.abs()).mean(-1).mean()
                l_sur = l_sur - r_sur * self.gamma
                grads = torch.autograd.grad(l_sur, [emb_bias])[0]
                grads /= (grads ** 2).sum(-1).sqrt().unsqueeze(1)

                emb_bias = emb_bias.detach() + strength * grads.detach() # + torch.randn_like(grads.detach()) * 0.000001
                self.optimizer.zero_grad()

            self.gamma -= beta * (rho - r_sur.detach())
            self.gamma = self.gamma.clamp(min=0.0, max=args_gamma)


            if epoch_idx >= self.warmup:
                x_oe = self.net.fc(emb[batch_size:] + emb_bias)
            else: 
                x_oe = self.net.fc(emb[batch_size:])

            l_oe = get_oe_loss(x_oe, 5.5)

            # l_oe = - (x_oe.mean(1) - torch.logsumexp(x_oe, dim=1)).mean()
            loss = l_ce + .5 * l_oe

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1) 
            self.optimizer.step()
            self.scheduler.step()
            
            # exponential moving average, show smooth values
            with torch.no_grad():
                loss_avg = loss_avg * 0.8 + float(loss) * 0.2

        metrics = {}
        metrics['epoch_idx'] = epoch_idx
        metrics['loss'] = self.save_metrics(loss_avg)
        
        self.diff = diff

        return self.net, metrics
