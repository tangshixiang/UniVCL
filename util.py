from __future__ import print_function
import math

import torch
import torch.nn as nn
import numpy as np
import os, subprocess

import torch.distributed as dist


def init_distributed_mode(args):
    """
    Initialize the following variables:
        - world_size
        - rank
    """

    args.is_slurm_job = "SLURM_JOB_ID" in os.environ

    if args.is_slurm_job:
        args.rank = int(os.environ["SLURM_PROCID"])
        # args.world_size = int(os.environ["SLURM_NNODES"]) * int(
        #     os.environ["SLURM_TASKS_PER_NODE"][0]
        # )
        args.world_size = int(os.environ["SLURM_NTASKS"])
        node_list = os.environ["SLURM_NODELIST"]
        addr = subprocess.getoutput(
            "scontrol show hostname {} | head -n1".format(node_list)
        )
        os.environ["MASTER_PORT"] = str(args.tcp_port)
        os.environ["MASTER_ADDR"] = addr
        os.environ["WORLD_SIZE"] = str(args.world_size)
        os.environ["RANK"] = str(args.rank)
    else:
        # multi-GPU job (local or multi-node) - jobs started with torch.distributed.launch
        # read environment variables
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        # args.rank = int(os.environ["LOCAL_RANK"])
        # args.world_size = dist.get_world_size()

    # prepare distributed
    dist.init_process_group(
        backend="nccl",
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )

    # set cuda device
    args.gpu_to_work_on = args.rank % torch.cuda.device_count()
    # device = torch.device("cuda:"+str(args.gpu_to_work_on))
    torch.cuda.set_device(args.gpu_to_work_on)
    return

# NOTE: assumes that the epoch starts with 1
def adjust_learning_rate(epoch, opt, optimizer):
    if hasattr(opt, 'cos') and opt.cos:
        # NOTE: since epoch starts with 1, we have to subtract 1
        new_lr = opt.learning_rate * 0.5 * (1. + math.cos(math.pi * (epoch-1) / opt.epochs))
        print('LR: {}'.format(new_lr))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
    else:
        steps = np.sum(epoch > np.asarray(opt.lr_decay_epochs))
        if steps > 0:
            new_lr = opt.learning_rate * (opt.lr_decay_rate ** steps)
            print('LR: {}'.format(new_lr))
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def subset_classes(dataset, num_classes=10):
    np.random.seed(1234)
    all_classes = sorted(dataset.class_to_idx.items(), key=lambda x: x[1])
    subset_classes = [all_classes[i] for i in np.random.permutation(len(all_classes))[:num_classes]]
    subset_classes = sorted(subset_classes, key=lambda x: x[1])
    dataset.classes_to_idx = {c: i for i, (c, _) in enumerate(subset_classes)}
    dataset.classes = [c for c, _ in subset_classes]
    orig_to_new_inds = {orig_ind: new_ind for new_ind, (_, orig_ind) in enumerate(subset_classes)}
    dataset.samples = [(p, orig_to_new_inds[i]) for p, i in dataset.samples if i in orig_to_new_inds]

def get_activation(activation_type):
    activation_type_list = {'relu': nn.ReLU(), 'identity': nn.Identity(), 'softmax': nn.Softmax()}
    return activation_type_list[activation_type]



if __name__ == '__main__':
    meter = AverageMeter()
