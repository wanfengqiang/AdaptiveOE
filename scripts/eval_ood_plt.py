import os, sys
ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
# ROOT_DIR = "/data/xhn/OpenOOD"
sys.path.append(ROOT_DIR)
import numpy as np
import pandas as pd
import argparse
import pickle
import collections
from glob import glob

import torch
import torch.nn as nn
import torch.nn.functional as F

from openood.evaluation_api import Evaluator

from openood.networks import ResNet18_32x32, ResNet18_224x224, ResNet50
from openood.networks.conf_branch_net import ConfBranchNet
from openood.networks.godin_net import GodinNet
from openood.networks.rot_net import RotNet
from openood.networks.csi_net import CSINet
from openood.networks.udg_net import UDGNet
from openood.networks.cider_net import CIDERNet
from openood.networks.npos_net import NPOSNet
from openood.networks.palm_net import PALMNet
from openood.networks.t2fnorm_net import T2FNormNet
from openood.networks.ascood_net import ASCOODNet


def update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


parser = argparse.ArgumentParser()
parser.add_argument('--root', required=True)
parser.add_argument('--postprocessor', default='msp')
parser.add_argument(
    '--id-data',
    type=str,
    default='cifar10',
    choices=['cifar10', 'cifar100', 'aircraft', 'cub', 'imagenet200'])
parser.add_argument('--batch-size', type=int, default=200)
parser.add_argument('--save-csv', action='store_true')
parser.add_argument('--save-score', action='store_true')
parser.add_argument('--fsood', action='store_true')
parser.add_argument('--wrapper-net',
                    type=str,
                    default=None,
                    choices=['ASCOODNet'])
args = parser.parse_args()

root = args.root

# specify an implemented postprocessor
# 'openmax', 'msp', 'temp_scaling', 'odin'...
postprocessor_name = args.postprocessor

NUM_CLASSES = {'cifar10': 10, 'cifar100': 100, 'imagenet200': 200}
MODEL = {
    'cifar10': ResNet18_32x32,
    'cifar100': ResNet18_32x32,
    'imagenet200': ResNet18_224x224,
}

try:
    num_classes = NUM_CLASSES[args.id_data]
    model_arch = MODEL[args.id_data]
except KeyError:
    raise NotImplementedError(f'ID dataset {args.id_data} is not supported.')

# assume that the root folder contains subfolders each corresponding to
# a training run, e.g., s0, s1, s2
# this structure is automatically created if you use OpenOOD for train
if len(glob(os.path.join(root, 's*'))) == 0:
    raise ValueError(f'No subfolders found in {root}')

# iterate through training runs
all_metrics = []
for subfolder in sorted(glob(os.path.join(root, 's*'))):
    print(subfolder)
    # load pre-setup postprocessor if exists
    if os.path.isfile(
            os.path.join(subfolder, 'postprocessors',
                         f'{postprocessor_name}.pkl')):
        with open(
                os.path.join(subfolder, 'postprocessors',
                             f'{postprocessor_name}.pkl'), 'rb') as f:
            postprocessor = pickle.load(f)
    else:
        postprocessor = None

    # load the pretrained model provided by the user
    if postprocessor_name == 'conf_branch':
        net = ConfBranchNet(backbone=model_arch(num_classes=num_classes),
                            num_classes=num_classes)
    elif postprocessor_name == 'godin':
        backbone = model_arch(num_classes=num_classes)
        net = GodinNet(backbone=backbone,
                       feature_size=backbone.feature_size,
                       num_classes=num_classes)
    elif postprocessor_name == 'rotpred':
        net = RotNet(backbone=model_arch(num_classes=num_classes),
                     num_classes=num_classes)
    elif 'csi' in root:
        backbone = model_arch(num_classes=num_classes)
        net = CSINet(backbone=backbone,
                     feature_size=backbone.feature_size,
                     num_classes=num_classes)
    elif 'udg' in root:
        backbone = model_arch(num_classes=num_classes)
        net = UDGNet(backbone=backbone,
                     num_classes=num_classes,
                     num_clusters=1000)
    elif postprocessor_name in ['cider', 'reweightood']:
        backbone = model_arch(num_classes=num_classes)
        net = CIDERNet(backbone,
                       head='mlp',
                       feat_dim=128,
                       num_classes=num_classes)
    elif postprocessor_name == 'npos':
        backbone = model_arch(num_classes=num_classes)
        net = NPOSNet(backbone,
                      head='mlp',
                      feat_dim=128,
                      num_classes=num_classes)
    elif postprocessor_name == 'palm':
        backbone = model_arch(num_classes=num_classes)
        net = PALMNet(backbone,
                      head='mlp',
                      feat_dim=128,
                      num_classes=num_classes)
        postprocessor_name = 'mds'
    elif postprocessor_name == 't2fnorm':
        backbone = model_arch(num_classes=num_classes)
        net = T2FNormNet(backbone=net, num_classes=num_classes)
    else:
        net = model_arch(num_classes=num_classes)

    if args.wrapper_net is not None:
        net = eval(args.wrapper_net)(backbone=net)

    net.load_state_dict(
        torch.load(os.path.join(subfolder, 'best.ckpt'), map_location='cpu'))
    net.cuda()
    net.eval()

    evaluator = Evaluator(
        net,
        id_name=args.id_data,  # the target ID dataset
        data_root="/data/xhn/current/OpenOOD/data",
        config_root=os.path.join(ROOT_DIR, 'configs'),
        preprocessor=None,  # default preprocessing
        postprocessor_name=postprocessor_name,
        postprocessor=
        postprocessor,  # the user can pass his own postprocessor as well
        batch_size=args.
        batch_size,  # for certain methods the results can be slightly affected by batch size
        shuffle=False,
        num_workers=8)

    # load pre-computed scores if exist
    if os.path.isfile(
            os.path.join(subfolder, 'scores', f'{postprocessor_name}.pkl')):
        with open(
                os.path.join(subfolder, 'scores', f'{postprocessor_name}.pkl'),
                'rb') as f:
            scores = pickle.load(f)
        update(evaluator.scores, scores)
        print('Loaded pre-computed scores from file.')

    # save the postprocessor for future reuse
    if hasattr(evaluator.postprocessor, 'setup_flag'
               ) or evaluator.postprocessor.hyperparam_search_done is True:
        pp_save_root = os.path.join(subfolder, 'postprocessors')
        if not os.path.exists(pp_save_root):
            os.makedirs(pp_save_root)

        if not os.path.isfile(
                os.path.join(pp_save_root, f'{postprocessor_name}.pkl')):
            with open(os.path.join(pp_save_root, f'{postprocessor_name}.pkl'),
                      'wb') as f:
                pickle.dump(evaluator.postprocessor, f,
                            pickle.HIGHEST_PROTOCOL)

    metrics = evaluator.eval_ood(fsood=args.fsood)
    all_metrics.append(metrics.to_numpy())

    # save computed scores
    if args.save_score:
        score_save_root = os.path.join(subfolder, 'scores')
        if not os.path.exists(score_save_root):
            os.makedirs(score_save_root)
        with open(os.path.join(score_save_root, f'{postprocessor_name}.pkl'),
                  'wb') as f:
            pickle.dump(evaluator.scores, f, pickle.HIGHEST_PROTOCOL)
    ood_name = 'mnist'
    kk = {'texture': 'Textures', 'cifar10': 'CIFAR-10', 'svhn': 'SVHN', 'tin': 'TinyImageNet', 'mnist': 'MNIST'}
    in_scores = evaluator.scores['id']['test'][1]
    out_scores = evaluator.scores['ood']['far'][ood_name][1]
    
    [id_pred, id_conf, id_gt] = evaluator.scores['id']['test']
    [ood_pred, ood_conf, ood_gt] = evaluator.scores['ood']['far'][ood_name]
    
    
    ood_gt = -1 * np.ones_like(ood_gt)  # hard set to -1 as ood
    pred = np.concatenate([id_pred, ood_pred])
    conf = np.concatenate([id_conf, ood_conf])
    label = np.concatenate([id_gt, ood_gt])
    from openood.evaluators.metrics import compute_all_metrics
    fpr = compute_all_metrics(conf, label, pred)[0]
    
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    fpr95_threshold = np.percentile(in_scores, 5) 
    plt.rcParams.update({
        'font.size': 35,       # 设置全局字体大小
        'axes.titlesize': 44,  # 标题字体大小
        'axes.labelsize': 44,  # 轴标签字体大小
        'xtick.labelsize': 30, # x轴刻度字体大小
        'ytick.labelsize': 30, # y轴刻度字体大小
        'legend.fontsize': 35  # 图例字体大小
    })
    scale = 1.3
    plt.figure(figsize=(8.0*scale, 5.5*scale))

    ood_id = 0

    # 计算直方图及其 bin 边界
    id_histogram, id_bins = np.histogram(in_scores, bins=100, density=True)
    ood_histogram, ood_bins = np.histogram(out_scores, bins=100, density=True)

    # 计算 bin 中心作为 x 坐标
    x_values_id = (id_bins[1:])
    print(id_histogram, id_bins)
    x_values_ood = (ood_bins[1:])
    print(ood_histogram, ood_bins)

    # 正则化
    mmx = max(id_histogram.max(), ood_histogram.max())
    mx = 1

    x_all = np.concatenate([x_values_id, x_values_ood])
    x_min, x_max = x_all.min(), x_all.max()

    # 归一化横坐标
    # x_values_id = (x_values_id - x_min) / (x_max - x_min)
    # x_values_ood = (x_values_ood - x_min) / (x_max - x_min)

    # 绘制曲线
    plt.plot(x_values_id, id_histogram / mx, alpha=1.0, color='#4F94CD', lw=2, label='ID')
    plt.plot(x_values_ood, ood_histogram / mx, alpha=1.0, color='#FF8C00', lw=2, label='OOD')

    # 填充区域
    plt.fill_between(x_values_id, id_histogram / mx, alpha=0.6, color='#4F94CD')
    plt.fill_between(x_values_ood, ood_histogram / mx, alpha=0.6, color='#FF8C00')
    fpr95_threshold = (fpr95_threshold - x_min) / (x_max - x_min)

    # 填充FPR95阈值右侧的OOD部分
    plt.fill_between(x_values_ood, ood_histogram / mx, where=(x_values_ood >= fpr95_threshold), 
                    color='black', alpha=0.3, hatch='//', label='OOD > 95% ID Threshold')

    # 图例
    legend_rectangles = [
        Rectangle((0, 0), 1, 1, color='#4F94CD', alpha=0.6),
        Rectangle((0, 0), 1, 1, color='#FF8C00', alpha=0.6),
        # Rectangle((0, 0), 1, 1, color='black', alpha=0.3, hatch='//')
    ]
    plt.legend(handles=legend_rectangles, labels=['ID: CIFAR-10', f'OOD: {kk[ood_name]}'],loc='upper right',)

    # 自定义其他图形元素
    plt.xlabel('OOD Scores')
    plt.ylabel('Frequency')
    # plt.title('Energy score distribution')
    # plt.ylim(0, 1)
    # plt.xlim(0, 1)
    plt.text(0.2, 5, f"FPR95: {fpr*100:.2f}%", fontweight='bold', fontsize=47, color="black")
    # ====== 插入放大图 START ======
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

    ax_main = plt.gca()
    axins = inset_axes(ax_main, width="30%", height="40%", loc='upper left', borderpad=2)

    # 画局部放大图
    axins.plot(x_values_id, id_histogram / mx, alpha=1.0, color='#4F94CD', lw=2)
    axins.plot(x_values_ood, ood_histogram / mx, alpha=1.0, color='#FF8C00', lw=2)
    axins.fill_between(x_values_id, id_histogram / mx, alpha=0.6, color='#4F94CD')
    axins.fill_between(x_values_ood, ood_histogram / mx, alpha=0.6, color='#FF8C00')
    axins.fill_between(x_values_ood, ood_histogram / mx, where=(x_values_ood >= fpr95_threshold),
                    color='black', alpha=0.3, hatch='//')

    # 设置缩放区域
    axins.set_xlim(0.9, 1.0)
    axins.set_ylim(0, 5)
    axins.tick_params(axis='both', which='major', labelsize=15)

    # 加放大区域标记线
    mark_inset(ax_main, axins, loc1=2, loc2=4, fc="none", ec="0.5")

    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

    # 保存并显示图形，使用bbox_inches='tight'来去掉多余的空白
    plt.savefig(f'./baseline_{ood_name}_{subfolder[-1]}.pdf', bbox_inches='tight')
    plt.show()

    # 清除绘图，为下次迭代准备
    plt.clf()

# compute mean metrics over training runs
all_metrics = np.stack(all_metrics, axis=0)
metrics_mean = np.mean(all_metrics, axis=0)
metrics_std = np.std(all_metrics, axis=0)

final_metrics = []
for i in range(len(metrics_mean)):
    temp = []
    for j in range(metrics_mean.shape[1]):
        temp.append(u'{:.2f} \u00B1 {:.2f}'.format(metrics_mean[i, j],
                                                   metrics_std[i, j]))
    final_metrics.append(temp)
df = pd.DataFrame(final_metrics, index=metrics.index, columns=metrics.columns)

if args.save_csv:
    saving_root = os.path.join(root, 'ood' if not args.fsood else 'fsood')
    if not os.path.exists(saving_root):
        os.makedirs(saving_root)
    df.to_csv(os.path.join(saving_root, f'{postprocessor_name}.csv'))
else:
    print(df)
