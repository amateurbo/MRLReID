import logging
import math
import os
import time
import torch
import torch.nn as nn
from torchvision import transforms

from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval
from torch.cuda import amp
import torch.distributed as dist
from utils.keep_cutout import Keep_Cutout, Keep_Cutout_Low
from torch.utils.tensorboard import SummaryWriter
import scipy.stats

def do_train(cfg, args,
             model,
             center_criterion,
             train_loader,
             val_loader,
             optimizer,
             optimizer_center,
             scheduler,
             loss_fn,
             num_query, local_rank,
             keep=None):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD
    start_eval = cfg.SOLVER.START_EVAL

    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    denormal = transforms.Compose([
            transforms.Normalize((-mean[0] / std[0], -mean[1] / std[1], -mean[2] / std[2]),
                                 (1 / std[0], 1 / std[1], 1 / std[2])),
            # transforms.ToPILImage()
        ])


    #记录最大Rank1
    maxrank1 = 0
    maxrank1_cmc =[]
    maxrank1_map = 0
    maxmap = 0
    maxmap_cmc = []
    maxrank1_epoch = 0
    maxmAP_epoch = 0
    last_cmc = []
    last_map = 0

    device = "cuda"
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("transreid.train")
    logger.info('start training')
    _LOCAL_PROCESS_GROUP = None
    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    scaler = amp.GradScaler()
    # train
    writer = SummaryWriter(cfg.OUTPUT_DIR)

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        evaluator.reset()
        scheduler.step(epoch)

        state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'epoch': epoch}

        for n_iter, (img, vid, target_cam, target_view, imgsh, imgsw) in enumerate(train_loader):
            # start = time.time()
            optimizer.zero_grad()
            optimizer_center.zero_grad()
            img = img.to(device)
            target = vid.to(device)
            target_cam = target_cam.to(device)
            target_view = target_view.to(device)

            #keep_cutout
            if args.method != "":
                img = keep(img, model, target_cam, target_view)

            if(n_iter == 0):
                writer.add_images('augimg', img.cpu(), epoch)
                for id in range(img.size(0)):
                    img[id] = denormal(img[id])
                writer.add_images('denoraml(augimg)', img, epoch)

            model.train()
            res_target = imgsh.to(device)

            # if n_iter >= 68:
            #     print('target: {}'.format(target))
            #     print('target_cam: {}'.format(target_cam))
            #     print('res_target: {}'.format(res_target))

            with amp.autocast(enabled=True):
                if cfg.DATALOADER.SAMPLER == 'softmax2_triplet':
                    score, feat, res_score = model(img, target, cam_label=target_cam, view_label=target_view)
                    # print('target_cam: {}'.format(target_cam))
                    # print('res_target: {}'.format(res_target))
                    loss = loss_fn(score, feat, target, target_cam, res_score, res_target)
                elif cfg.DATALOADER.SAMPLER == 'rmse_softmax_triplet':
                    res_score = model(img, target, cam_label=target_cam, view_label=target_view)
                    # print('target_cam: {}'.format(target_cam))
                    # print('res_target: {}'.format(res_target))
                    loss = loss_fn(res_score, res_target)
                else:
                    score, feat = model(img, target, cam_label=target_cam, view_label=target_view)
                    # print('target_cam: {}'.format(target_cam))
                    loss = loss_fn(score, feat, target, target_cam)

            # print("loss={}".format(loss))
            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()

            if 'center' in cfg.MODEL.METRIC_LOSS_TYPE:
                for param in center_criterion.parameters():
                    param.grad.data *= (1. / cfg.SOLVER.CENTER_LOSS_WEIGHT)
                scaler.step(optimizer_center)
                scaler.update()

            if cfg.DATALOADER.SAMPLER == 'rmse_softmax_triplet':
                # acc = loss
                PLCC = scipy.stats.pearsonr(res_score.cpu().squeeze(1).detach().numpy(), res_target.cpu().detach().numpy())[0]
                SROCC = scipy.stats.spearmanr(res_score.cpu().squeeze(1).detach().numpy(), res_target.cpu().detach().numpy())[0]
                acc = (PLCC + SROCC) / 2
            elif isinstance(score, list):
                acc = (score[0].max(1)[1] == target).float().mean()
            else:
                acc = (score.max(1)[1] == target).float().mean()
            # if isinstance(score, list):
            #     acc = (score[0].max(1)[1] == target).float().mean()
            # else:
            #     acc = (score.max(1)[1] == target).float().mean()

            loss_meter.update(loss.item(), img.shape[0])
            acc_meter.update(acc, 1)

            torch.cuda.synchronize()
            if (n_iter + 1) % log_period == 0:
                logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                            .format(epoch, (n_iter + 1), len(train_loader),
                                    loss_meter.avg, acc_meter.avg, scheduler._get_lr(epoch)[0]))
            # print("ont iter cost: {}".format(time.time() - start))


        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        if cfg.MODEL.DIST_TRAIN:
            pass
        else:
            logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                    .format(epoch, time_per_batch, train_loader.batch_size / time_per_batch))

        if epoch % checkpoint_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    torch.save(state,
                               # model.state_dict(),
                               os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))
            else:
                torch.save(state,
                           # model.state_dict(),
                           os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))

        if epoch >= start_eval:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    model.eval()
                    for n_iter, (img, vid, camid, camids, target_view, _, _, _) in enumerate(val_loader):
                        with torch.no_grad():
                            img = img.to(device)
                            camids = camids.to(device)
                            target_view = target_view.to(device)
                            feat = model(img, cam_label=camids, view_label=target_view)
                            evaluator.update((feat, vid, camid))
                    cmc, mAP, _, _, _, _, _ = evaluator.compute()
                    last_cmc = cmc
                    last_map = mAP
                    logger.info("Validation Results - Epoch: {}".format(epoch))
                    logger.info("mAP: {:.1%}".format(mAP))
                    for r in [1, 5, 10]:
                        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                    torch.cuda.empty_cache()
            else:
                model.eval()
                for n_iter, (img, vid, camid, camids, target_view, _, _, _) in enumerate(val_loader):
                    with torch.no_grad():
                        img = img.to(device)
                        camids = camids.to(device)
                        target_view = target_view.to(device)
                        feat = model(img, cam_label=camids, view_label=target_view)
                        evaluator.update((feat, vid, camid))
                cmc, mAP, _, _, _, _, _ = evaluator.compute()
                last_cmc = cmc
                last_map = mAP
                logger.info("Validation Results - Epoch: {}".format(epoch))
                logger.info("mAP: {:.1%}".format(mAP))
                for r in [1, 5, 10]:
                    logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                torch.cuda.empty_cache()
                if cmc[0] >= maxrank1:
                    maxrank1 = cmc[0]
                    maxrank1_cmc = cmc
                    maxrank1_map = mAP
                    maxrank1_epoch = epoch
                    torch.save(state,
                               # model.state_dict(),
                               # os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))
                               os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_best_Rank1'.format(epoch)))
                if mAP > maxmap:
                    maxmap = mAP
                    maxmap_cmc = cmc
                    maxmAP_epoch = epoch
                    torch.save(state,
                               # model.state_dict(),
                               # os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))
                               os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_best_mAP'.format(epoch)))
                print("MaxRank-{:<3}:{:.1%},  epoch: {}".format(1, maxrank1, maxrank1_epoch))
                print("MaxmAP-{:<3}:{:.1%},  epoch: {}".format(1, maxmap, maxmAP_epoch))

        elif epoch % eval_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    model.eval()
                    for n_iter, (img, vid, camid, camids, target_view, _, _, _) in enumerate(val_loader):
                        with torch.no_grad():
                            img = img.to(device)
                            camids = camids.to(device)
                            target_view = target_view.to(device)
                            feat = model(img, cam_label=camids, view_label=target_view)
                            evaluator.update((feat, vid, camid))
                    cmc, mAP, _, _, _, _, _ = evaluator.compute()
                    last_cmc = cmc
                    last_map = mAP
                    logger.info("Validation Results - Epoch: {}".format(epoch))
                    logger.info("mAP: {:.1%}".format(mAP))
                    for r in [1, 5, 10]:
                        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                    torch.cuda.empty_cache()
            else:
                model.eval()
                # for a ,b in enumerate(val_loader):
                #     print(a)
                #     print(b)
                for n_iter, (img, vid, camid, camids, target_view, _, _, _) in enumerate(val_loader):
                    with torch.no_grad():
                        img = img.to(device)
                        camids = camids.to(device)
                        target_view = target_view.to(device)
                        feat = model(img, cam_label=camids, view_label=target_view)
                        evaluator.update((feat, vid, camid))
                cmc, mAP, _, _, _, _, _ = evaluator.compute()
                last_cmc = cmc
                last_map = mAP
                logger.info("Validation Results - Epoch: {}".format(epoch))
                logger.info("mAP: {:.1%}".format(mAP))
                for r in [1, 5, 10]:
                    logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                torch.cuda.empty_cache()
                if cmc[0] >= maxrank1:
                    maxrank1 = cmc[0]
                    maxrank1_cmc = cmc
                    maxrank1_map = mAP
                    maxrank1_epoch = epoch
                    torch.save(state,
                               # model.state_dict(),
                               # os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))
                               os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_best_Rank1'.format(epoch)))
                if mAP > maxmap:
                    maxmap = mAP
                    maxmap_cmc = cmc
                    maxmAP_epoch = epoch
                    torch.save(state,
                               # model.state_dict(),
                               # os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))
                               os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_best_mAP'.format(epoch)))
                print("MaxRank-{:<3}:{:.1%},  epoch: {}".format(1, maxrank1, maxrank1_epoch))
                print("MaxmAP-{:<3}:{:.1%},  epoch: {}".format(1, maxmap, maxmAP_epoch))

    return last_map, last_cmc, maxrank1_map, maxrank1_cmc, maxrank1_epoch, maxmap, maxmap_cmc, maxmAP_epoch



def do_inference(cfg,
                 model,
                 val_loader,
                 num_query):
    device = "cuda"
    logger = logging.getLogger("transreid.test")
    logger.info("Enter inferencing")

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)

    evaluator.reset()

    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()
    img_path_list = []

    if cfg.TEST.EVAL_RESOLUTION:
        numTrue = 0
        numFalse = 0
        for n_iter, (img, vid, camid, camids, target_view, _, target, _) in enumerate(val_loader):
            with torch.no_grad():
                img = img.to(device)
                camids = camids.to(device)
                # target = vid.to(device)
                target_view = target_view.to(device)
                resolution_score = model(img, cam_label=camids, view_label=target_view)

                resolution_score = torch.argmax(resolution_score, axis=1)
                # target = torch.argmax(target, axis=1)
                for pre, tar in zip(resolution_score, target):
                    if pre == tar:
                        numTrue += 1
                    else:
                        print('PreScore: {},   TargetScore:{}'.format(pre, tar))
                        numFalse += 1
                    # print('PreScore: {},   TargetScore:{}'.format(pre, tar))
                print("True: {},   False: {}".format(numTrue, numFalse))

                if numTrue + numFalse >= 2228:
                    return
        return




    for n_iter, (img, pid, camid, camids, target_view, imgpath, target_mos, _) in enumerate(val_loader):
        with torch.no_grad():
            img = img.to(device)
            camids = camids.to(device)
            target_view = target_view.to(device)
            feat = model(img, cam_label=camids, view_label=target_view)
            evaluator.update((feat, pid, camid))
            img_path_list.extend(imgpath)

    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    logger.info("Validation Results ")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    return cmc[0], cmc[4]


