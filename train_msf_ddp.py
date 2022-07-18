import builtins
import os
import sys
import time
import argparse
import random

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter

from PIL import ImageFilter
from util import adjust_learning_rate, AverageMeter, init_distributed_mode
import models.resnet as resnet
from tools import get_logger

import cv2
from PIL import Image

cv2.setNumThreads(0)
memorycache = False
try:
    import mc, io
    memorycache = True
    print("using memory cache")
except:
    print("missing memory cache")
    pass

def pil_loader(img_str):
    buff = io.BytesIO(img_str)
    with Image.open(buff) as img:
        img = img.convert('RGB')
    return img

def parse_option():

    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('data', type=str, help='path to dataset')
    parser.add_argument('--dataset', type=str, default='imagenet',
                        choices=['imagenet', 'imagenet100'],
                        help='use full or subset of the dataset')
    parser.add_argument('--debug', action='store_true', help='whether in debug mode or not')

    parser.add_argument('--print_freq', type=int, default=100, help='print frequency')
    parser.add_argument('--save_freq', type=int, default=10, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=256, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=2, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=200, help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='90,120', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.2, help='decay rate for learning rate')
    parser.add_argument('--cos', action='store_true',
                        help='whether to cosine learning rate or not')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--sgd_momentum', type=float, default=0.9, help='SGD momentum')

    # model definition
    parser.add_argument('--arch', type=str, default='alexnet',
                        choices=['alexnet', 'resnet18', 'resnet50', 'mobilenet'])

    # Mean Shift
    parser.add_argument('--momentum', type=float, default=0.99)
    parser.add_argument('--mem_bank_size', type=int, default=128000)
    parser.add_argument('--topk', type=int, default=5)
    parser.add_argument('--weak_strong', action='store_true',
                        help='whether to strong/strong or weak/strong augmentation')

    parser.add_argument('--weights', type=str, help='weights to initialize the model from')
    parser.add_argument('--resume', default='', type=str,
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--restart', action='store_true')

    # GPU setting
    parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')

    parser.add_argument('--checkpoint_path', default='output/mean_shift_default', type=str,
                        help='where to save checkpoints. ')

    # tcp port setting
    parser.add_argument("--tcp_port", type=str, default="5017")
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up distributed
                    training; see https://pytorch.org/docs/stable/distributed.html""")

    opt = parser.parse_args()

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    return opt


# Extended version of ImageFolder to return index of image too.
class ImageFolderEx(datasets.ImageFolder):
    def __init__(self, root, transform=None):
        super(ImageFolderEx, self).__init__(root)
        self.transform = transform
        self.initialized = False

    def _init_memcached(self):
        if not self.initialized:
            server_list_config_file = "/mnt/lustre/share/memcached_client/server_list.conf"
            client_config_file = "/mnt/lustre/share/memcached_client/client.conf"
            self.mclient = mc.MemcachedClient.GetInstance(server_list_config_file, client_config_file)
            self.initialized = True

    def __getitem__(self, index):
        path, target = self.samples[index]

        global memorycache
        if (not memorycache):
            sample = self.loader(path)
        else:
            ## memcached
            self._init_memcached()
            value = mc.pyvector()
            self.mclient.Get(path, value)
            value_str = mc.ConvertBuffer(value)
            sample = pil_loader(value_str)

        sample = self.transform(sample)

        return index, sample, target

    def __len__(self):
        return len(self.samples)


# Extended version of ImageFolder to return index of image too.
class Image100FolderEx(datasets.ImageFolder):
    def __init__(self, root, transform=None):
        self.class_list = '/mnt/lustre/share/tangshixiang/data/ImageNet/ImageNet100_class_map.txt'
        super(Image100FolderEx, self).__init__(root)
        self.transform = transform
        self.initialized = False

    def _init_memcached(self):
        if not self.initialized:
            server_list_config_file = "/mnt/lustre/share/memcached_client/server_list.conf"
            client_config_file = "/mnt/lustre/share/memcached_client/client.conf"
            self.mclient = mc.MemcachedClient.GetInstance(server_list_config_file, client_config_file)
            self.initialized = True

    def _find_classes(self, dir):
        with open(self.class_list, 'r') as fp:
            classes = fp.read().strip().split("\n")
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def __getitem__(self, index):
        path, target = self.samples[index]

        global memorycache
        if (not memorycache):
            sample = self.loader(path)
        else:
            ## memcached
            self._init_memcached()
            value = mc.pyvector()
            self.mclient.Get(path, value)
            value_str = mc.ConvertBuffer(value)
            sample = pil_loader(value_str)

        sample = self.transform(sample)

        return index, sample, target

    def __len__(self):
        return len(self.samples)


def get_mlp(inp_dim, hidden_dim, out_dim):
    mlp = nn.Sequential(
        nn.Linear(inp_dim, hidden_dim),
        nn.BatchNorm1d(hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_dim, out_dim),
    )
    return mlp


class MeanShift(nn.Module):
    def __init__(self, arch, m=0.99, mem_bank_size=128000, topk=5):
        super(MeanShift, self).__init__()

        # save parameters
        self.m = m
        self.mem_bank_size = mem_bank_size
        self.topk = topk

        # create encoders and projection layers
        # both encoders should have same arch
        if 'resnet' in arch:
            self.encoder_q = resnet.__dict__[arch]()
            self.encoder_t = resnet.__dict__[arch]()

        # save output embedding dimensions
        # assuming that both encoders have same dim
        feat_dim = self.encoder_q.fc.in_features
        hidden_dim = feat_dim * 2
        proj_dim = feat_dim // 4

        # projection layers
        self.encoder_t.fc = get_mlp(feat_dim, hidden_dim, proj_dim)
        self.encoder_q.fc = get_mlp(feat_dim, hidden_dim, proj_dim)

        # prediction layer
        self.predict_q = get_mlp(proj_dim, hidden_dim, proj_dim)

        self.encoder_t = self.encoder_t.cuda()
        self.encoder_q = self.encoder_q.cuda()
        self.predict_q = self.predict_q.cuda()

        # copy query encoder weights to target encoder
        for param_q, param_t in zip(self.encoder_q.parameters(), self.encoder_t.parameters()):
            param_t.data.copy_(param_q.data)
            param_t.requires_grad = False

        print("using mem-bank size {}".format(self.mem_bank_size))
        # setup queue (For Storing Random Targets)
        self.register_buffer('queue', torch.randn(self.mem_bank_size, proj_dim))
        # normalize the queue embeddings
        self.queue = nn.functional.normalize(self.queue, dim=1)
        # initialize the labels queue (For Purity measurement)
        self.register_buffer('labels', -1*torch.ones(self.mem_bank_size).long())
        # setup the queue pointer
        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_target_encoder(self):
        for param_q, param_t in zip(self.encoder_q.parameters(), self.encoder_t.parameters()):
            param_t.data = param_t.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def model_bn_sync(self):
        self.encoder_q = nn.SyncBatchNorm.convert_sync_batchnorm(self.encoder_q)
        self.encoder_t = nn.SyncBatchNorm.convert_sync_batchnorm(self.encoder_t)
        self.predict_q = nn.SyncBatchNorm.convert_sync_batchnorm(self.predict_q)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, targets, labels):
        ptr = int(self.queue_ptr)

        targets = concat_all_gather(targets)
        labels = concat_all_gather(labels)
        batch_size = targets.shape[0]
        assert self.mem_bank_size % batch_size == 0

        # replace the targets at ptr (dequeue and enqueue)
        self.queue[ptr:ptr + batch_size] = targets
        self.labels[ptr:ptr + batch_size] = labels
        ptr = (ptr + batch_size) % self.mem_bank_size  # move pointer

        self.queue_ptr[0] = ptr

    def forward(self, im_q, im_t, labels):
        # compute query features
        feat_q = self.encoder_q(im_q)
        # compute predictions for instance level regression loss
        query = self.predict_q(feat_q)
        query = nn.functional.normalize(query, dim=1)

        # compute target features
        with torch.no_grad():
            # update the target encoder
            self._momentum_update_target_encoder()

            # forward through the target encoder
            current_target = self.encoder_t(im_t)
            current_target = nn.functional.normalize(current_target, dim=1)

            self._dequeue_and_enqueue(current_target, labels)

        # calculate mean shift regression loss
        targets = self.queue.clone().detach()
        # calculate distances between vectors
        dist_t = 2 - 2 * torch.einsum('bc,kc->bk', [current_target, targets])
        dist_q = 2 - 2 * torch.einsum('bc,kc->bk', [query, targets])

        # select the k nearest neighbors [with smallest distance (largest=False)] based on current target
        _, nn_index = dist_t.topk(self.topk, dim=1, largest=False)
        nn_dist_q = torch.gather(dist_q, 1, nn_index)

        labels = labels.unsqueeze(1).expand(nn_dist_q.shape[0], nn_dist_q.shape[1])
        labels_queue = self.labels.clone().detach()
        labels_queue = labels_queue.unsqueeze(0).expand((nn_dist_q.shape[0], self.mem_bank_size))
        labels_queue = torch.gather(labels_queue, dim=1, index=nn_index)
        matches = (labels_queue == labels).float()

        loss = (nn_dist_q.sum(dim=1) / self.topk).mean()
        purity = (matches.sum(dim=1) / self.topk).mean()

        return loss, purity

class TwoCropsTransform:
    """Take two random crops of one image as the query and target."""
    def __init__(self, weak_transform, strong_transform):
        self.weak_transform = weak_transform
        self.strong_transform = strong_transform
        print(self.weak_transform)
        print(self.strong_transform)

    def __call__(self, x):
        q = self.strong_transform(x)
        t = self.weak_transform(x)
        return [q, t]


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


# Create train loader
def get_train_loader(opt):
    traindir = os.path.join(opt.data, 'train')
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    normalize = transforms.Normalize(mean=mean, std=std)

    augmentation_strong = [
        transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ]

    augmentation_weak = [
        transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]

    if opt.dataset == 'imagenet100':
        if opt.weak_strong:
            train_dataset = Image100FolderEx(
                traindir,
                TwoCropsTransform(transforms.Compose(augmentation_weak), transforms.Compose(augmentation_strong))
            )
        else:
            train_dataset = Image100FolderEx(
                traindir,
                TwoCropsTransform(transforms.Compose(augmentation_strong), transforms.Compose(augmentation_strong))
            )
    else:
        if opt.weak_strong:
            train_dataset = ImageFolderEx(
                traindir,
                TwoCropsTransform(transforms.Compose(augmentation_weak), transforms.Compose(augmentation_strong))
            )
        else:
            train_dataset = ImageFolderEx(
                traindir,
                TwoCropsTransform(transforms.Compose(augmentation_strong), transforms.Compose(augmentation_strong))
            )

    print('==> train dataset')
    print(train_dataset)

    # build sampler
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

    # NOTE: remove drop_last
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=False, sampler=train_sampler,
        num_workers=opt.num_workers, pin_memory=True, drop_last=True, persistent_workers=True)

    return train_loader


def main():
    global writer
    args = parse_option()

    init_distributed_mode(args)

    if args.rank == 0:
        os.makedirs(args.checkpoint_path, exist_ok=True)
        os.makedirs(os.path.join(args.checkpoint_path, 'events'), exist_ok=True)
        writer = SummaryWriter(os.path.join(args.checkpoint_path, 'events'))

        if not args.debug:
            os.environ['PYTHONBREAKPOINT'] = '0'
            logger = get_logger(
                logpath=os.path.join(args.checkpoint_path, 'logs'),
                filepath=os.path.abspath(__file__)
            )

            def print_pass(*arg):
                logger.info(*arg)
            builtins.print = print_pass

    print(args)

    train_loader = get_train_loader(args)

    mean_shift = MeanShift(
        args.arch,
        m=args.momentum,
        mem_bank_size=args.mem_bank_size,
        topk=args.topk
    )
    mean_shift.model_bn_sync()
    mean_shift.cuda()
    print(mean_shift)

    mean_shift = nn.parallel.DistributedDataParallel(
        mean_shift,
        device_ids=[args.gpu_to_work_on],
        find_unused_parameters=False,
    )

    params = [p for p in mean_shift.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params,
                                lr=args.learning_rate,
                                momentum=args.sgd_momentum,
                                weight_decay=args.weight_decay)

    cudnn.benchmark = True
    args.start_epoch = 1

    if args.weights:
        print('==> load weights from checkpoint: {}'.format(args.weights))
        ckpt = torch.load(args.weights, map_location=lambda storage, loc: storage.cuda(args.gpu_to_work_on))
        print('==> resume from epoch: {}'.format(ckpt['epoch']))
        if 'model' in ckpt:
            sd = ckpt['model']
        else:
            sd = ckpt['state_dict']
        msg = mean_shift.load_state_dict(sd, strict=False)
        optimizer.load_state_dict(ckpt['optimizer'])
        args.start_epoch = ckpt['epoch'] + 1
        print(msg)

    if os.path.exists(os.path.join(args.checkpoint_path, 'ckpt_last.pth')) and not args.resume:
        try:
            args.resume = os.path.join(args.checkpoint_path, 'ckpt_last.pth')
            print('==> resume from checkpoint: {}'.format(args.resume))
            ckpt = torch.load(args.resume, map_location=lambda storage, loc: storage.cuda(args.gpu_to_work_on))
            # ckpt = torch.load(args.resume, map_location="cuda:{}".format(args.gpu_to_work_on))
            print('==> resume from epoch: {}'.format(ckpt['epoch']))
            mean_shift.load_state_dict(ckpt['state_dict'], strict=True)
            if not args.restart:
                optimizer.load_state_dict(ckpt['optimizer'])
                args.start_epoch = ckpt['epoch'] + 1
        except:
            import glob
            candidate_resumes = glob.glob(os.path.join(args.checkpoint_path, 'ckpt_epoch_*.pth'))
            epoch_num = [int(x.split('/')[-1].split('ckpt_epoch_')[-1].split('.pth')[0]) for x in candidate_resumes]

            if len(epoch_num) == 0:
                if args.resume:
                    print('==> resume from checkpoint: {}'.format(args.resume))
                    ckpt = torch.load(args.resume, map_location=lambda storage, loc: storage.cuda(args.gpu_to_work_on))
                    print('==> resume from epoch: {}'.format(ckpt['epoch']))
                    mean_shift.load_state_dict(ckpt['state_dict'], strict=True)
                    if not args.restart:
                        optimizer.load_state_dict(ckpt['optimizer'])
                        args.start_epoch = ckpt['epoch'] + 1
                else:
                    print('==> resume from scratch!')
            else:
                max_epoch_num = max(epoch_num)
                args.resume = os.path.join(args.checkpoint_path, 'ckpt_epoch_{epoch}.pth'.format(epoch=max_epoch_num))
                print('==> resume from checkpoint: {}'.format(args.resume))
                ckpt = torch.load(args.resume, map_location=lambda storage, loc: storage.cuda(args.gpu_to_work_on))
                print('==> resume from epoch: {}'.format(ckpt['epoch']))
                mean_shift.load_state_dict(ckpt['state_dict'], strict=True)
                if not args.restart:
                    optimizer.load_state_dict(ckpt['optimizer'])
                    args.start_epoch = ckpt['epoch'] + 1
    elif args.resume:
        print('==> resume from checkpoint: {}'.format(args.resume))
        ckpt = torch.load(args.resume, map_location=lambda storage, loc: storage.cuda(args.gpu_to_work_on))
        print('==> resume from epoch: {}'.format(ckpt['epoch']))
        mean_shift.load_state_dict(ckpt['state_dict'], strict=True)
        if not args.restart:
            optimizer.load_state_dict(ckpt['optimizer'])
            args.start_epoch = ckpt['epoch'] + 1
    else:
        print('==> resume from scratch!')

    # if args.resume:
    #     print('==> resume from checkpoint: {}'.format(args.resume))
    #     ckpt = torch.load(args.resume)
    #     print('==> resume from epoch: {}'.format(ckpt['epoch']))
    #     mean_shift.load_state_dict(ckpt['state_dict'], strict=True)
    #     if not args.restart:
    #         optimizer.load_state_dict(ckpt['optimizer'])
    #         args.start_epoch = ckpt['epoch'] + 1

    # routine
    for epoch in range(args.start_epoch, args.epochs + 1):

        adjust_learning_rate(epoch, args, optimizer)
        print("==> training...")

        train_loader.sampler.set_epoch(epoch)

        time1 = time.time()
        train(epoch, train_loader, mean_shift, optimizer, args)

        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        # saving the model
        if args.rank == 0:
            print('==> Saving last...')
            state = {
                'opt': args,
                'state_dict': mean_shift.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
            }

            save_file = os.path.join(args.checkpoint_path, 'ckpt_last.pth')
            torch.save(state, save_file)

            # help release GPU memory
            del state
        torch.cuda.empty_cache()


        if (epoch % args.save_freq == 0) and (args.rank == 0):
            print('==> Saving...')
            state = {
                'opt': args,
                'state_dict': mean_shift.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
            }

            save_file = os.path.join(args.checkpoint_path, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            torch.save(state, save_file)

            # help release GPU memory
            del state
            torch.cuda.empty_cache()

    dist.barrier()
    if args.rank == 0:
        with open(os.path.join(args.checkpoint_path, 'complete.txt'), 'w') as f:
            f.writelines('complete!')

def train(epoch, train_loader, mean_shift, optimizer, opt):
    """
    one epoch training for CompReSS
    """
    mean_shift.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    purity_meter = AverageMeter()

    end = time.time()
    for idx, (indices, (im_q, im_t), labels) in enumerate(train_loader):
        data_time.update(time.time() - end)
        im_q = im_q.cuda(non_blocking=True)
        im_t = im_t.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)

        # ===================forward=====================
        loss, purity = mean_shift(im_q=im_q, im_t=im_t, labels=labels)

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===================meters=====================
        loss_meter.update(loss.item(), im_q.size(0))
        purity_meter.update(purity.item(), im_q.size(0))

        if opt.rank == 0:
            writer.add_scalar('Loss/Iter', loss.item(), idx + epoch * len(train_loader))
            writer.add_scalar('Purity/Iter', purity.item(), idx + epoch * len(train_loader))

        torch.cuda.synchronize()
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0 and opt.rank == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'purity {purity.val:.3f} ({purity.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time,
                   purity=purity_meter,
                   loss=loss_meter))
            sys.stdout.flush()
            sys.stdout.flush()

    return loss_meter.avg

# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output

if __name__ == '__main__':
    main()
