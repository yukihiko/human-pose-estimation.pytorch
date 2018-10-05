# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint
import shutil

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
import numpy as np
from torch.autograd import Variable

import _init_paths
from core.config import config
from core.config import update_config
from core.config import update_dir
from core.config import get_model_name
from core.loss import JointsMSELoss
from core.function import train
from core.function import validate
from utils.utils import get_optimizer
from utils.utils import save_checkpoint
from utils.utils import create_logger

import dataset
import models
from models import MnasNet_, MobileNet16_


class OneDriveLogger(object):

    def __init__(self):
        self.logs = []

    def write_oneDrive(self, log):
        """ Write log. """
        self.file = open('C:/Users/aoyagi/OneDrive/pytorch/log.txt', 'a')
        self.file.write(log + '\n')
        self.file.flush()
        self.file.close()
        self.logs.append(log + '\n')

def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    args, rest = parser.parse_known_args()
    # update config
    update_config(args.cfg)

    # training
    parser.add_argument('--frequent',
                        help='frequency of logging',
                        default=config.PRINT_FREQ,
                        type=int)
    parser.add_argument('--gpus',
                        help='gpus',
                        type=str)
    parser.add_argument('--workers',
                        help='num of dataloader workers',
                        type=int)
    parser.add_argument('--resume',
                        type=str)
    parser.add_argument('--useOneDrive',
                        action='store_true')
    parser.add_argument('--useOffset',
                        action='store_true')
    parser.add_argument('--NN',
                        help='NN',
                        type=str)

    args = parser.parse_args()

    return args


def reset_config(config, args):
    if args.gpus:
        config.GPUS = args.gpus
    if args.workers:
        config.WORKERS = args.workers

def fgraph(module, threshold):
    print(module)
    if module != None:
        if isinstance(module, torch.nn.Sequential):
            for child in module.children():
                fgraph(child, threshold)

    if isinstance(module, torch.nn.Conv2d):
        old_weights = module.weight.data.cpu().numpy()
        new_weights = (old_weights > threshold) * old_weights
        module.weight.data = torch.from_numpy(old_weights).cuda()
        #module.weight.data = torch.from_numpy(new_weights).cuda()
        #print(module.weight.data)


def main():
    args = parse_args()
    reset_config(config, args)

    logger, final_output_dir, tb_log_dir = create_logger(
        config, args.cfg, 'train')

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    if args.useOneDrive == True:
        oneDriveLogger = OneDriveLogger()
    else:
        oneDriveLogger = None

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED

    if args.NN == "MobileNet16_":
        model = MobileNet16_()
    elif args.NN == "MnasNet16_":
        model = MnasNet_()
    else:
        model = MnasNet_()

    optimizer_state_dict = None
    if args.resume:
        '''
        checkpoint = torch.load(args.resume)
        state_dict = checkpoint['state_dict']
        # create new OrderedDict that does not contain `module.`
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        # load params
        model.load_state_dict(new_state_dict)
        optimizer_state_dict = checkpoint['optimizer']
        '''
        checkpoint = torch.load(args.resume)
        state_dict = checkpoint['state_dict']
        model.load_state_dict(state_dict)
        optimizer_state_dict = checkpoint['optimizer']
        #model.load_state_dict(torch.load(args.resume))

    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    dump_input = torch.rand((config.TRAIN.BATCH_SIZE,
                             config.MODEL.NUM_JOINTS,
                             config.MODEL.IMAGE_SIZE[1],
                             config.MODEL.IMAGE_SIZE[0]))
    #writer_dict['writer'].add_graph(model, (dump_input, ))

    gpus = [int(i) for i in config.GPUS.split(',')]
    #model = torch.nn.DataParallel(model, device_ids=gpus).cuda()
    model.cuda()

    #pruning modile
    model.eval()
    dummy_input = Variable(torch.randn(1, 3, 256, 256)).cuda()
    model.eval()
    _ = model(dummy_input)
    '''
    all_weights = []
    for p in model.parameters():
        if len(p.data.size()) != 1:
            all_weights += list(p.cpu().data.abs().numpy().flatten())
    threshold = np.percentile(np.array(all_weights), 5.)
    
    fgraph(model.model, threshold)
    
    for child in model.children():
        for param in child.parameters():
            param.reguired_grand = False
    '''
    filename = os.path.join(final_output_dir, 'pruning')
    torch.save(model.state_dict(), filename + '.model')
    # end pruning model

    # define loss function (criterion) and optimizer
    criterion = JointsMSELoss(
        use_target_weight=config.LOSS.USE_TARGET_WEIGHT
    ).cuda()

    optimizer = get_optimizer(config, model)
    if optimizer_state_dict != None:
        optimizer.load_state_dict(optimizer_state_dict)

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, config.TRAIN.LR_STEP, config.TRAIN.LR_FACTOR
    )

    # Data loading code
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_dataset = eval('dataset.'+config.DATASET.DATASET)(
        config,
        config.DATASET.ROOT,
        config.DATASET.TRAIN_SET,
        True,
        transforms.Compose([
            transforms.ToTensor(),
            #normalize,
        ])
    )
    valid_dataset = eval('dataset.'+config.DATASET.DATASET)(
        config,
        config.DATASET.ROOT,
        config.DATASET.TEST_SET,
        False,
        transforms.Compose([
            transforms.ToTensor(),
            #normalize,
        ])
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.TRAIN.BATCH_SIZE*len(gpus),
        shuffle=config.TRAIN.SHUFFLE,
        num_workers=config.WORKERS,
        pin_memory=True
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config.TEST.BATCH_SIZE*len(gpus),
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True
    )

    best_perf = 0.0
    best_model = False
    for epoch in range(config.TRAIN.BEGIN_EPOCH, config.TRAIN.END_EPOCH):
        lr_scheduler.step()
        
        '''
        # train for one epoch
        train(config, train_loader, model, criterion, optimizer, epoch,
              final_output_dir, tb_log_dir, writer_dict, oneDriveLogger, args.useOffset)

        filename = os.path.join(final_output_dir, 'epoch-{0}'.format(epoch + 1))
        torch.save(model.state_dict(), filename + '.model')
        lastestname = os.path.join(final_output_dir, 'lastest')
        torch.save(model.state_dict(), lastestname + '.model')
        if args.useOneDrive == True:
            torch.save(model.state_dict(), 'C:/Users/aoyagi/OneDrive/pytorch/lastest.model')
        '''
        # evaluate on validation set
        perf_indicator = validate(config, valid_loader, valid_dataset, model,
                                  criterion, final_output_dir, tb_log_dir,
                                  writer_dict, oneDriveLogger, args.useOffset)

        if perf_indicator > best_perf:
            best_perf = perf_indicator
            best_model = True
        else:
            best_model = False

        logger.info('=> saving checkpoint to {}'.format(final_output_dir))
        save_checkpoint({
            'epoch': epoch + 1,
            'model': get_model_name(config),
            'state_dict': model.state_dict(),
            'perf': perf_indicator,
            'optimizer': optimizer.state_dict(),
        }, best_model, final_output_dir)
        
    final_model_state_file = os.path.join(final_output_dir,
                                          'final_state.pth.tar')
    logger.info('saving final model state to {}'.format(
        final_model_state_file))
    torch.save(model.module.state_dict(), final_model_state_file)
    #writer_dict['writer'].close()


if __name__ == '__main__':
    main()
