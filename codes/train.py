import os
import math
import argparse
import random
import logging
import torch
import torch.distributed as dist
from torch.utils.data.sampler import Sampler
from simclr import *
import options.options as option
from utils import util
from utils.util import (
    configure_optimizers, load_optimizer,
    configure_schedulers, load_scheduler,
    create_model, create_dataloader, create_dataset,
    print_network,
    torch2img,
    compute_metrics,
    AverageMeter,
)
from criterions.criterion import Criterion
from criterions.criterion import Criterion_val

import warnings

warnings.filterwarnings("ignore")


def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()


def main():
    #### options
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, help='Path to option YMAL file.', default='./conf/train/sample.yml')
    parser.add_argument('--local_rank', type=int, default=-1)
    args = parser.parse_args()
    opt = option.parse(args.opt, is_train=True)

    #### distributed training settings
    rank = args.local_rank
    world_size = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1

    if rank == -1:
        opt['dist'] = False
        print('Disabled distributed training.')
    else:
        opt['dist'] = True
    if world_size > 1:
        torch.cuda.set_device(rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()
    #### loading resume state if exists
    if opt['path'].get('checkpoint', None):
        # distributed resuming: all load into default GPU
        device_id = torch.cuda.current_device()
        checkpoint = torch.load(opt['path']['checkpoint'],
                                map_location=lambda storage, loc: storage.cuda(device_id))
    else:
        checkpoint = None
    #### mkdir and loggers
    if rank <= 0:  # normal training (rank -1) OR distributed training (rank 0)
        if checkpoint is None:
            util.mkdir_and_rename(
                opt['path']['experiments_root'])  # rename experiment folder if exists
            util.mkdirs((path for key, path in opt['path'].items() if not key == 'experiments_root'
                         and 'pretrain_model' not in key and 'resume' not in key))

        # config loggers. Before it, the log will not work
        util.setup_logger('base', opt['path']['log'], 'train_' + opt['name'], level=logging.INFO,
                          screen=True, tofile=True)
        util.setup_logger('val', opt['path']['log'], 'val_' + opt['name'], level=logging.INFO,
                          screen=True, tofile=True)
        logger = logging.getLogger('base')
        logger_val = logging.getLogger('val')  # validation logger
        logger.info(option.dict2str(opt))
        # tensorboard logger
        if opt['use_tb_logger']:
            version = float(torch.__version__[0:3])
            if version >= 1.1:  # PyTorch 1.1
                from torch.utils.tensorboard import SummaryWriter
            else:
                logger.info(
                    'You are using PyTorch {}. Tensorboard will use [tensorboardX]'.format(version))
                from tensorboard import SummaryWriter
            tb_logger = SummaryWriter(log_dir='../../tb_logger/' + opt['name'])
    else:
        util.setup_logger('base', opt['path']['log'], 'train', level=logging.INFO, screen=True)
        logger = logging.getLogger('base')

    # convert to NoneDict, which returns None for missing keys
    opt = option.dict_to_nonedict(opt)

    #### random seed
    seed = opt['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    if rank <= 0:
        logger.info('Random seed: {}'.format(seed))
    util.set_random_seed(seed)

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    #### create train and val dataloader
    mode = opt['train']['mode']
    device = 'cuda' if opt['gpu_ids'] is not None else 'cpu'
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = create_dataset(dataset_opt)
            if mode == 'epoch':
                train_size = int(math.floor(len(train_set) / dataset_opt['batch_size']))
                total_epochs = int(opt['train'][mode]['value'])
                total_iters = train_size * total_epochs
                if 'debug' not in opt['name']:
                    opt['train']['epoch']['val_freq'] *= train_size
            elif mode == 'step':
                train_size = int(math.floor(len(train_set) / dataset_opt['batch_size']))
                total_iters = int(opt['train'][mode]['value'])
                total_epochs = int(math.ceil(total_iters / train_size))
            else:
                raise NotImplementedError('mode [{:s}] is not recognized.'.format(mode))

            if opt['dist']:
                train_sampler = Sampler()
            else:
                train_sampler = None
            train_loader = create_dataloader(train_set, dataset_opt, opt, train_sampler)

            if rank <= 0:
                logger.info('Number of train samples: {:,d}, iters: {:,d}'.format(
                    len(train_set), train_size))
                logger.info('Total epochs needed: {:d} for iters {:,d}'.format(
                    total_epochs, total_iters))
        elif phase == 'val':
            val_set = create_dataset(dataset_opt)
            val_loader = create_dataloader(val_set, dataset_opt, opt, None)
            if rank <= 0:
                logger.info('Number of val samples in [{:s}]: {:d}'.format(
                    dataset_opt['name'], len(val_set)))
        else:
            raise NotImplementedError('Phase [{:s}] is not recognized.'.format(phase))
    assert train_loader is not None
    assert val_loader is not None

    #### create model
    model = create_model(opt, checkpoint, None, rank)
    model = model.to(device)
    # print(device)

    #### create optimizer and schedulers
    optimizer_dict = configure_optimizers(opt, model)
    scheduler_dict = configure_schedulers(opt, optimizer_dict)

    optimizer = load_optimizer(optimizer_dict, 'optimizer', checkpoint)
    aux_optimizer = load_optimizer(optimizer_dict, 'aux_optimizer', checkpoint)

    lr_scheduler = load_scheduler(scheduler_dict, 'lr_scheduler', checkpoint)
    aux_lr_scheduler = load_scheduler(scheduler_dict, 'aux_lr_scheduler', checkpoint)

    #### resume training
    if checkpoint:
        if rank <= 0:
            logger.info('Resuming training from epoch: {}, iter: {}.'.format(
                checkpoint['epoch'], checkpoint['iter']))
        # training state
        start_epoch = checkpoint['epoch']
        best_loss = checkpoint['loss']
        current_step = start_epoch * math.ceil(len(train_loader.dataset) / opt['datasets']['train']['batch_size'])
        checkpoint = None


    #### criterion
    criterion = Criterion(opt)
    criterion_val = Criterion_val(opt)

    # torch.cuda.empty_cache()

    #### training
    if rank <= 0:
        logger.info(
            'Model parameter numbers: {:d}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
        logger.info('Start training from epoch: {:d}, iter: {:d}'.format(start_epoch, current_step))

    loss_cap = opt['train']['loss_cap']  ##loss_cap损失上限
    for epoch in range(start_epoch, total_epochs + 1):
        if opt['dist']:
            train_sampler.set_epoch(epoch)
        if rank <= 0 and mode == 'epoch':
            message = 'lr_main: {:e}'.format(optimizer.param_groups[0]['lr'])
            message += ' | lr_aux: {:e}'.format(aux_optimizer.param_groups[0]['lr'])
            logger.info(message)

        for _, train_data in enumerate(train_loader):
            torch.cuda.empty_cache()
            current_step += 1
            if current_step > total_iters:
                break

            #### training
            model.train()
            # device = next(model.parameters()).device
            gt, noise = train_data
            gt = gt.to(device)
            noise = noise.to(device)

            optimizer.zero_grad()
            aux_optimizer.zero_grad()

            # forward
            out_net = model(noise, gt)
            out_train = criterion(out_net, gt)

            # do optimization if and only if the loss is small (rec is somehow bounded with 0-1)
            optimizer_flag = out_train["loss"].item() >= 0 and out_train["loss"].item() < loss_cap
            if not optimizer_flag:
                message = '[Warning]: network parameters are not optimized due to train loss = {:.4f}.'.format(
                    out_train['loss'].item())
                print(message)
                # logger.info(message)

            # optimizer
            out_train["loss"].backward()
            if opt['train']['clip_max_norm'] > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), opt['train']['clip_max_norm'])
            if not optimizer_flag:
                optimizer.zero_grad()
            optimizer.step()

            #### update learning rate for step mode
            if mode == 'step':
                lr_scheduler.step()
                aux_lr_scheduler.step()

            #### log: weighted loss
            if current_step % opt['logger']['print_freq'] == 0:
                wanted_keys = ['loss', 'bpp_loss', 'contrastive_loss', 'aux_loss']
                message = '<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> [weighted]'.format(epoch, current_step,
                                                                                    optimizer.param_groups[0]['lr'])
                for k, v in out_train.items():
                    # tensorboard logger
                    if opt['use_tb_logger']:
                        if rank <= 0:
                            mode_counter = epoch if mode == 'epoch' else current_step
                            tb_logger.add_scalar('[train]: {}'.format(k), v.item(), mode_counter)
                    # message
                    if k in wanted_keys or 'weighted' in k:
                        k = k.replace('weighted_', '')
                        message += ' | {:s}: {:.4f}'.format(k, v.item())

            # validation
            if current_step % opt['train'][mode]['val_freq'] == 0 and rank <= 0:
                model.eval()
                # device = next(model.parameters()).device

                log = {}
                for k in out_train.keys():
                    log[k] = AverageMeter()
                log['psnr'] = AverageMeter()
                log['ms_ssim'] = AverageMeter()

                with torch.no_grad():
                    mode_counter = epoch if mode == 'epoch' else current_step
                    this_val_dir = os.path.join(opt['path']['val_samples'], '{:d}'.format(mode_counter))
                    if not os.path.exists(this_val_dir):
                        os.makedirs(this_val_dir)
                    for i, val_data in enumerate(val_loader):
                        gt, noise = val_data
                        gt = gt.to(device)
                        noise = noise.to(device)

                        out_net = model(noise, gt)
                        out_val = criterion_val(out_net, gt)
                        out_val['aux_loss'] = model.aux_loss()

                        for k, v in out_val.items():
                            log[k].update(v.item())

                        # save

                        rec = torch2img(out_net['x_hat'])
                        gt = torch2img(gt)
                        noise = torch2img(noise)
                        p, m = compute_metrics(rec, gt)
                        log['psnr'].update(p)
                        log['ms_ssim'].update(m)

                        if i < 12:
                            rec.save(os.path.join(this_val_dir, '{:03d}_rec.png'.format(i)))
                            gt.save(os.path.join(this_val_dir, '{:03d}_gt.png'.format(i)))
                            noise.save(os.path.join(this_val_dir, '{:03d}_noise.png'.format(i)))

                # val tensorboard
                for k, v in log.items():
                    if opt['use_tb_logger']:
                        if rank <= 0:
                            mode_counter = epoch if mode == 'epoch' else current_step
                            tb_logger.add_scalar('[val]: {}'.format(k), v.avg, mode_counter)

                # [val] weighted loss
                wanted_keys = ['loss', 'bpp_loss', 'aux_loss']
                message = '<epoch:{:3d}, iter:{:8,d}> [weighted]'.format(epoch, current_step)
                for k, v in log.items():
                    if k in wanted_keys or 'weighted' in k:
                        k = k.replace('weighted_', '')
                        message += ' | {:s}: {:.4f}'.format(k, v.avg)
                if rank <= 0:
                    logger_val.info(message)


                #### save checkpoints
                loss = log['rd_loss'].avg
                is_best = loss < best_loss
                best_loss = min(loss, best_loss)
                if rank <= 0 and is_best:
                    save_dict = {
                        "epoch": epoch,
                        "iter": current_step,
                        "state_dict": model.state_dict(),
                        "loss": loss,
                        "optimizer": optimizer_dict['optimizer'].state_dict(),
                        "aux_optimizer": optimizer_dict['aux_optimizer'].state_dict(),
                        "lr_scheduler": scheduler_dict['lr_scheduler'].state_dict(),
                        "aux_lr_scheduler": scheduler_dict['aux_lr_scheduler'].state_dict(),
                    }
                    mode_counter = epoch if mode == 'epoch' else current_step
                    save_path = os.path.join(opt['path']['checkpoints'], "checkpoint_best_loss.pth.tar")
                    torch.save(save_dict, save_path)
                    logger.info('best checkpoint saved.')
                    logger_val.info('best checkpoint saved.')

                torch.cuda.empty_cache()

        #### save checkpoints
        if rank <= 0 and (epoch + 1) % opt['logger']['save_checkpoint_freq'] == 0:
            save_dict = {
                "epoch": epoch,
                "iter": current_step,
                "state_dict": model.state_dict(),
                "loss": best_loss,
                "optimizer": optimizer_dict['optimizer'].state_dict(),
                "aux_optimizer": optimizer_dict['aux_optimizer'].state_dict(),
                "lr_scheduler": scheduler_dict['lr_scheduler'].state_dict(),
                "aux_lr_scheduler": scheduler_dict['aux_lr_scheduler'].state_dict(),
            }
            mode_counter = epoch if mode == 'epoch' else current_step
            save_path = os.path.join(opt['path']['checkpoints'], "checkpoint_{:d}.pth.tar".format(mode_counter))
            torch.save(save_dict, save_path)

        #### update learning rate for epoch mode
        if mode == 'epoch':
            lr_scheduler.step()
            aux_lr_scheduler.step()
    if rank <= 0:
        logger.info('Saving the final model.')
        save_dict = {
            "epoch": epoch,
            "iter": current_step,
            "state_dict": model.state_dict(),
            "loss": best_loss,
            "optimizer": optimizer_dict['optimizer'].state_dict(),
            "aux_optimizer": optimizer_dict['aux_optimizer'].state_dict(),
            "lr_scheduler": scheduler_dict['lr_scheduler'].state_dict(),
            "aux_lr_scheduler": scheduler_dict['aux_lr_scheduler'].state_dict(),
        }
        mode_counter = epoch if mode == 'epoch' else current_step
        save_path = os.path.join(opt['path']['checkpoints'], "checkpoint_latest.pth.tar")
        torch.save(save_dict, save_path)
        logger.info('End of training.')


if __name__ == '__main__':
    main()