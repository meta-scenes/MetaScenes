
import argparse
from collections import OrderedDict
import math
import wandb

import torch.cuda.amp as amp
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from data.dataset_3d import *

from utils.utils import get_dataset
import models.Scan2Sim_models as models
from utils.tokenizer import SimpleTokenizer
from utils import utils
from data.dataset_3d import customized_collate_fn, pil_loader

import torch.nn.functional as F
import time
from tqdm import tqdm

def get_args_parser():
    parser = argparse.ArgumentParser(description='Scan2Sim training and evaluation', add_help=False)
    # Data
    parser.add_argument('--output-dir', default='./outputs/test_pointbert_8kpts', type=str, help='output dir')
    parser.add_argument('--pretrain_dataset_name', default='shapenet', type=str)
    parser.add_argument('--validate_dataset_name', default='modelnet40', type=str)
    parser.add_argument('--use_height', action='store_true', help='whether to use height informatio, by default enabled with PointNeXt.')
    parser.add_argument('--npoints', default=8192, type=int, help='number of points used for pre-train and test.')
    # Model
    parser.add_argument('--model', default='Scan2Sim_PointBERT_Colored', type=str)


    # Training
    parser.add_argument('--epochs', default=250, type=int)
    parser.add_argument('--warmup-epochs', default=1, type=int)
    parser.add_argument('--start-epoch', default=0, type=int)
    parser.add_argument('--batch-size', default=6, type=int,
                        help='number of samples per-device/per-gpu')
    parser.add_argument('--lr', default=3e-3, type=float)
    parser.add_argument('--lr-start', default=1e-6, type=float,
                        help='initial warmup lr')
    parser.add_argument('--lr-end', default=1e-5, type=float,
                        help='minimum final lr')
    parser.add_argument('--update-freq', default=1, type=int,
                        help='optimizer update frequency (i.e. gradient accumulation steps)')
    parser.add_argument('--wd', default=0.1, type=float)
    parser.add_argument('--betas', default=(0.9, 0.98), nargs=2, type=float)
    parser.add_argument('--eps', default=1e-8, type=float)
    parser.add_argument('--eval-freq', default=1, type=int)
    parser.add_argument('--disable-amp', action='store_true',
                        help='disable mixed-precision training (requires more memory and compute)')
    parser.add_argument('--resume', default='/path/to/resume/checkpoint', type=str, help='path to resume from')
    parser.add_argument('--evaluate_3d', action='store_true', help='eval Scan2Sim only')

    # System
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
    parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                        help='number of data loading workers per process')
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='node rank for distributed training')
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument('--dist-url', default='env://', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--gpu', default=0, type=int, help='GPU id to use.')
    parser.add_argument('--wandb', action='store_true', help='Enable WandB logging')

    return parser

best_acc1 = 0

def main(args):
    utils.init_distributed_mode(args)

    global best_acc1

    if utils.is_main_process() and args.wandb:
        wandb_id = os.path.split(args.output_dir)[-1]
        wandb.init(project='MetaScenes', id=wandb_id, config=args, reinit=True, entity='hhuangyue')

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    # create model
    print("=> creating model: {}".format(args.model))
    model = getattr(models, args.model)(args=args)
    model.cuda(args.gpu)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], bucket_cap_mb=200, find_unused_parameters=False)

    # define loss function (criterion) and optimizer
    criterion = models.get_loss(args).cuda(args.gpu)

    for name, param in model.named_parameters():
        if 'open_clip_model' in name:
            param.requires_grad = False
        if 'point_encoder.cls_head_finetune' not in name and 'point_encoder.norm' not in name and 'blocks.blocks.17' not in name:
            param.requires_grad = False

    p_wd, p_non_wd = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            # print('in optimizer freeze {}'.format(n))
            continue  # frozen weights
        if p.ndim < 2 or 'bias' in n or 'ln' in n or 'bn' in n:
            p_non_wd.append(p)
        else:
            p_wd.append(p)

    optim_params = [{"params": p_wd, "weight_decay": args.wd},
                    {"params": p_non_wd, "weight_decay": 0}]

    optimizer = torch.optim.AdamW(optim_params, lr=args.lr, betas=args.betas,
                                    eps=args.eps, weight_decay=args.wd)
    scaler = amp.GradScaler(enabled=not args.disable_amp)

    # optionally resume from a checkpoint (takes precedence over autoresume)
    if os.path.isfile(args.resume):
        print("=> loading resume checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume, map_location='cpu')

        state_dict = OrderedDict()
        for k, v in checkpoint['state_dict'].items():
            state_dict[k.replace('module.', '')] = v

        epoch = checkpoint['epoch'] if 'epoch' in checkpoint else 0
        args.start_epoch = epoch
        result = model.load_state_dict(state_dict, strict=False)
        scaler.load_state_dict(checkpoint['scaler']) if 'scaler' in checkpoint else ()

        print("=> loaded resume checkpoint '{}' (epoch {})"
              .format(args.resume, epoch))

    else:
        # auto-resume from the latest checkpoint in output directory
        latest = os.path.join(args.output_dir, 'checkpoint_best.pt')
        if os.path.isfile(latest):
            print("=> loading latest checkpoint '{}'".format(latest))
            latest_checkpoint = torch.load(latest, map_location='cpu')
            args.start_epoch = latest_checkpoint['epoch']
            model.load_state_dict(latest_checkpoint['state_dict'])
            optimizer.load_state_dict(latest_checkpoint['optimizer'])
            scaler.load_state_dict(latest_checkpoint['scaler'])
            best_acc1 = latest_checkpoint['best_acc1']
            print("=> loaded latest checkpoint '{}' (epoch {})"
                  .format(latest, latest_checkpoint['epoch']))

    cudnn.benchmark = True


    # Data loading code
    print("=> creating dataset")
    tokenizer = SimpleTokenizer()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
            transforms.ToTensor(),
            normalize
        ])


    test_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            normalize
        ])


    train_dataset = get_dataset(train_transform, tokenizer, args, 'train')
    val_dataset = get_dataset(test_transform, tokenizer, args, 'val')

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True,
        collate_fn=customized_collate_fn)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=(val_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=val_sampler, drop_last=False)

    lr_schedule = utils.cosine_scheduler(args.lr, args.lr_end, args.epochs,
        len(train_loader) // args.update_freq, warmup_epochs=args.warmup_epochs, start_warmup_value=args.lr_start)

    print(args)

    if args.evaluate_3d:
        print('Evaluating...')
        _ = evaluate(val_loader, model, args=args)

    else:
        print("Training...")
        best_epoch = -1
        for epoch in range(args.start_epoch, args.epochs):
            if args.distributed:
                train_sampler.set_epoch(epoch)

            train_stats = train(train_loader, model, criterion, optimizer, scaler, epoch, lr_schedule, args)
            val_stats = {"acc1": -1}

            if epoch % 1 == 0:

                val_stats = evaluate(val_loader, model, args=args)
                acc1 = val_stats["acc1"]
                print(val_stats)

                is_best = acc1 > best_acc1
                if is_best:
                    best_epoch = epoch

                best_acc1 = max(acc1, best_acc1)

                if is_best or epoch % 2 == 0:
                    print("=> saving checkpoint")
                    utils.save_on_master({
                            'epoch': epoch + 1,
                            'state_dict': model.state_dict(),
                            'optimizer' : optimizer.state_dict(),
                            'scaler': scaler.state_dict(),
                            'best_acc1': best_acc1,
                            'args': args,
                        }, is_best, args.output_dir)

                if epoch + 1 == args.epochs:
                    print("=> saving last checkpoint")
                    utils.save_on_master({
                        'epoch': 'last',
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scaler': scaler.state_dict(),
                        'best_acc1': best_acc1,
                        'args': args,
                    }, is_best, args.output_dir)

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         **{f'test_{k}': v for k, v in val_stats.items()},
                         'epoch': epoch,
                         'best_acc1': best_acc1,
                         'best_epoch': best_epoch}

            if utils.is_main_process():
                if args.wandb:
                    wandb.log(log_stats)
                    # wandb.watch(model)
                with open(os.path.join(args.output_dir, 'log.txt'), 'a') as f:
                    f.write(json.dumps(log_stats) + '\n')


def train(train_loader, model, criterion, optimizer, scaler, epoch, lr_schedule, args):
    batch_time = AverageMeter('Time', ':6.2f')
    data_time = AverageMeter('Data', ':6.2f')
    mem = AverageMeter('Mem (GB)', ':6.1f')
    metric_names = models.get_metric_names(args.model)
    iters_per_epoch = len(train_loader) // args.update_freq
    metrics = OrderedDict([(name, AverageMeter(name, ':.2e')) for name in metric_names])
    progress = ProgressMeter(
        iters_per_epoch,
        [batch_time, data_time, mem, *metrics.values()],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for data_iter, inputs in enumerate(tqdm(train_loader, desc="Training", total=len(train_loader))):
        optim_iter = data_iter // args.update_freq

        # measure data loading time
        data_time.update(time.time() - end)

        # update weight decay and learning rate according to their schedule
        it = iters_per_epoch * epoch + optim_iter  # global training iteration
        for k, param_group in enumerate(optimizer.param_groups):
            param_group['lr'] = lr_schedule[it]

        pc_image = inputs[3]['image_pc_data']
        pc_text = inputs[3]['text_pc_data']
        texts = inputs[2]
        image = inputs[4]
        cls_gt = inputs[5]

        group_cnt = pc_image.shape[1]
        assert pc_image.shape[1] == pc_text.shape[1]

        reshape_pc_image = pc_image.reshape(args.batch_size * group_cnt, args.npoints, 6)
        reshape_pc_text = pc_text.reshape(args.batch_size * group_cnt, args.npoints, 6)

        inputs = [reshape_pc_image, reshape_pc_text, texts, image]
        inputs = [tensor.cuda(args.gpu, non_blocking=True) for tensor in inputs]

        zero_images_mask = (image == 0).all(dim=1).all(dim=1).all(dim=1).unsqueeze(1).float() # text-embed only

        # compute output
        with amp.autocast(enabled=not args.disable_amp):
            outputs = model(*inputs)
            outputs['cls_gt'] = cls_gt
            outputs['zero_images_mask'] = zero_images_mask
            loss_dict = criterion(outputs)
            loss = loss_dict['loss']
            loss /= args.update_freq

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()))
            sys.exit(1)

        scaler.scale(loss).backward()

        if (data_iter + 1) % args.update_freq != 0:
            continue

        # compute gradient and do SGD step
        scaler.step(optimizer)
        scaler.update()
        model.zero_grad(set_to_none=True)

        # clamp logit scale to [0, 100]

        utils.get_model(model).logit_scale.data.clamp_(0, 4.6052)
        logit_scale = utils.get_model(model).logit_scale.exp().item()

        for k in loss_dict:
            metrics[k].update(loss_dict[k].item(), args.batch_size)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        mem.update(torch.cuda.max_memory_allocated() // 1e9)

        if optim_iter % args.print_freq == 0:
            if utils.is_main_process() and args.wandb:
                wandb.log({**{k: v.item() for k, v in loss_dict.items()},
                        'scaler': scaler.get_scale()})
            progress.display(optim_iter)

    progress.synchronize()

    return {**{k: v.avg for k, v in metrics.items()},
            'lr': optimizer.param_groups[0]['lr']}


def evaluate(test_loader, model,  args=None):
    batch_time = AverageMeter('Time', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    top1_image = AverageMeter('Acc@1', ':6.2f')
    top5_image = AverageMeter('Acc@5', ':6.2f')

    top1_q = AverageMeter('Acc@1', ':6.2f')
    top5_q = AverageMeter('Acc@5', ':6.2f')

    top1_text = AverageMeter('Acc@1', ':6.2f')
    top5_text = AverageMeter('Acc@5', ':6.2f')

    progress = ProgressMeter(
        len(test_loader),
        [batch_time, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    pred_info = {}

    with torch.no_grad():

        end = time.time()
        for i, inputs in enumerate(test_loader):

            insts = inputs[0]
            scans = inputs[1]
            pc_image = inputs[3]['image_pc_data']
            # pc_text = inputs[3]['text_pc_data']
            image = inputs[4]
            text = inputs[2]
            cls_gt_one_hot = inputs[5]
            zero_images_mask = (image == 0).all(dim=1).all(dim=1).all(dim=1).unsqueeze(1).float()  # text-embed only

            pc_image = pc_image.cuda(args.gpu, non_blocking=True)
            # pc_text = pc_text.cuda(args.gpu, non_blocking=True)
            image = image.cuda(args.gpu, non_blocking=True)
            text = text.cuda(args.gpu, non_blocking=True)
            cls_gt_one_hot = cls_gt_one_hot.cuda(args.gpu, non_blocking=True)

            test_batch = image.shape[0]

            # encode pc
            group_cnt = pc_image.shape[1]
            reshape_pc_image = pc_image.reshape(test_batch * group_cnt, args.npoints, 6)
            pc_features_image, pc_cls = utils.get_model(model).encode_pc(reshape_pc_image)

            # encode text
            text_embed_all = []
            for ii in range(text.shape[0]):
                text_for_one_sample = text[ii]
                text_embed = utils.get_model(model).encode_text(text_for_one_sample)
                text_embed = text_embed / text_embed.norm(dim=-1, keepdim=True)
                text_embed = text_embed.mean(dim=0)
                text_embed = text_embed / text_embed.norm(dim=-1, keepdim=True)
                text_embed_all.append(text_embed)

            text_feature = torch.stack(text_embed_all)
            text_feature = text_feature / text_feature.norm(dim=-1, keepdim=True)


            # encode image
            image_feature = utils.get_model(model).encode_image(image)
            image_feature = image_feature / image_feature.norm(dim=-1, keepdim=True)

            # unpack features
            pc_image_embed_group = pc_features_image.reshape(test_batch, group_cnt, pc_features_image.size(-1))
            # pc_text_embed_group = pc_features_text.reshape(test_batch, group_cnt, pc_features_text.size(-1))

            pc_cls_group = pc_cls.reshape(test_batch, group_cnt, pc_cls.size(-1))
            pc_cls_group = pc_cls_group.squeeze(dim=2)
            # pc_cls_group = F.softmax(pc_cls_group, dim=1)

            pc_image_embed_group = F.normalize(pc_image_embed_group, dim=-1, p=2)
            # pc_text_embed_group = F.normalize(pc_text_embed_group, dim=-1, p=2)

            logits_per_image_pc = torch.einsum('ijk, ik->ij', [pc_image_embed_group, image_feature])
            mask = (1-zero_images_mask).expand_as(logits_per_image_pc).to('cuda')
            # logits_per_image_pc = F.softmax(logits_per_image_pc, dim=1)

            logits_per_text_pc = torch.einsum('ijk, ik->ij', [pc_image_embed_group, text_feature])
            # logits_per_text_pc = F.softmax(logits_per_text_pc, dim=1)
            cls_gt = torch.argmax(cls_gt_one_hot, dim=1).to('cuda')


            # measure accuracy and record loss
            (acc1, acc5), correct = accuracy(logits_per_image_pc * mask + logits_per_text_pc, cls_gt, topk=(1, 5))
            (acc1_image, acc5_image), _ = accuracy(logits_per_image_pc * mask, cls_gt, topk=(1, 5))
            (acc1_q, acc5_q), _ = accuracy(pc_cls_group, cls_gt, topk=(1, 5))
            (acc1_text, acc5_text), _ = accuracy(logits_per_text_pc, cls_gt, topk=(1, 5))

            # TODO: fix the all reduce for the correct variable, assuming only one process for evaluation!
            acc1, acc5 = utils.scaled_all_reduce([acc1, acc5])
            top1.update(acc1.item(), image.size(0))
            top5.update(acc5.item(), image.size(0))

            acc1_image, acc5_image = utils.scaled_all_reduce([acc1_image, acc5_image])
            top1_image.update(acc1_image.item(), image.size(0))
            top5_image.update(acc5_image.item(), image.size(0))

            acc1_q, acc5_q = utils.scaled_all_reduce([acc1_q, acc5_q])
            top1_q.update(acc1_q.item(), image.size(0))
            top5_q.update(acc5_q.item(), image.size(0))

            acc1_text, acc5_text = utils.scaled_all_reduce([acc1_text, acc5_text])
            top1_text.update(acc1_text.item(), image.size(0))
            top5_text.update(acc5_text.item(), image.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    progress.synchronize()

    print('0-shot * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}')

    print ({'acc1': top1.avg, 'acc5': top5.avg, 'acc1_image': top1_image.avg, 'acc5_image': top5_image.avg, 'acc1_q': top1_q.avg, 'acc5_q': top5_q.avg,  'acc1_text': top1_text.avg, 'acc5_text': top5_text.avg})

    return {'acc1': top1.avg, 'acc5': top5.avg, 'acc1_image': top1_image.avg, 'acc5_image': top5_image.avg, 'acc1_q': top1_q.avg, 'acc5_q': top5_q.avg,  'acc1_text': top1_text.avg, 'acc5_text': top5_text.avg}


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def synchronize(self):
        if not utils.is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.sum, self.count], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.sum = int(t[0])
        self.count = t[1]
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def synchronize(self):
        for meter in self.meters:
            meter.synchronize()

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res, correct

if __name__ == '__main__':

    import multiprocessing as mp
    mp.set_start_method('spawn')


    print('running')
    parser = argparse.ArgumentParser('Scan2Sim training and evaluation', parents=[get_args_parser()])
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    print (args)
    main(args)
