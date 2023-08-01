import sys
import time

import cv2
from torch.utils.tensorboard import SummaryWriter
from threading import Timer

from utils.logger import setup_logger
from utils.loggers import Logger
from datasets import make_dataloader
from model import make_model
from solver import make_optimizer
from solver.scheduler_factory import create_scheduler
from loss import make_loss
from processor import do_train
import random
import torch
import numpy as np
import os
import argparse
# from timm.scheduler import create_scheduler
from config import cfg
from utils.keep_cutout import Keep_Cutout, Keep_Cutout_Low, Keep_Cutout_vit_explain, Keep_Cutout_vit_explain_randomsize
from torchvision import transforms
from PIL import Image
from utils.vit_explain import VITAttentionRollout, show_mask_on_image


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def train1(transform=[]):
    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="configs/MTA_reid/vit_transreid_stride_global_local_res.yml", help="path to config file", type=str
    )
    # parser.add_argument(
    #     "--OUTPUT_DIR", default="/mnt/data/code/reidlog/transreid", help="path to config file", type=str
    # )

    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument("--local_rank", default=0, type=int)

    parser.add_argument("--method", default='', type=str)
    parser.add_argument('--length', type=int, default=96, help='length of the holes')
    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # cfg.OUTPUT_DIR = args.OUTPUT_DIR
    # cfg.freeze()
    #
    # cfg.MODEL.NO_MARGIN = True

    # set_seed(42)
    set_seed(cfg.SOLVER.SEED)

    if cfg.MODEL.DIST_TRAIN:
        torch.cuda.set_device(args.local_rank)

    output_dir = cfg.OUTPUT_DIR
    cur_time = time.strftime('%Y-%m-%d-%H-%M-%S')
    output_dir = os.path.join(output_dir, cur_time)
    cfg.OUTPUT_DIR = output_dir

    # cfg.OUTPUT_DIR = '/mnt/data/code/reidlog/transreid/test'
    # output_dir = cfg.OUTPUT_DIR

    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    sys.stdout = Logger(os.path.join(output_dir, 'mytrainlog.log'))



    logger, _ = setup_logger("transreid", output_dir, if_train=True)
    logger.info("Saving model in the path :{}".format(cfg.OUTPUT_DIR))
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    if cfg.MODEL.DIST_TRAIN:
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID
    train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num, train_transform = make_dataloader(cfg, transforms=transform)
    keep = None

    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num = view_num)

    loss_func, center_criterion = make_loss(cfg, num_classes=num_classes)

    optimizer, optimizer_center = make_optimizer(cfg, model, center_criterion)

    scheduler = create_scheduler(cfg, optimizer)


    do_train(
        cfg, args,
        model,
        center_criterion,
        train_loader,
        val_loader,
        optimizer,
        optimizer_center,
        scheduler,
        loss_func,
        num_query, args.local_rank,
        keep=keep,
    )

    # sys.stdout.close()





if __name__ == '__main__':
    # time.sleep(13000)
    train1()
    #
    # t = Timer(18000, train1)
    # t.start()
    # train1(transform=[])
    # train1(transform=['random_flip'])
    # train1(transform=['body_blur'])
    # train1(transform=['random_erase'])
    # train1(transform=['body_blur', 'random_flip'])
    # train1(transform=['body_blur', 'random_erase'])
    # train1(transform=['body_blur', 'random_flip', 'pad', 'random_crop'])
    # train1(transform=['body_blur', 'random_flip', 'pad', 'random_crop', 'random_erase'])
    # train1(transform=['body_blur', 'pad'])
    # train1(transform=['body_blur', 'random_crop'])


    # train1(transform=['body_blur'])
    #
    # train1(transform=['random_flip', 'body_blur'])


# def train2():
#     parser = argparse.ArgumentParser(description="ReID Baseline Training")
#     parser.add_argument(
#         "--config_file", default="configs/DukeMTMC/swinv2_base.yml", help="path to config file", type=str
#     )
#     # parser.add_argument(
#     #     "--OUTPUT_DIR", default="/mnt/data/code/reidlog/transreid", help="path to config file", type=str
#     # )
#
#     parser.add_argument("opts", help="Modify config options using the command-line", default=None,
#                         nargs=argparse.REMAINDER)
#     parser.add_argument("--local_rank", default=0, type=int)
#
#     parser.add_argument("--method", default='', type=str)
#     parser.add_argument('--length', type=int, default=96, help='length of the holes')
#     args = parser.parse_args()
#
#     if args.config_file != "":
#         cfg.merge_from_file(args.config_file)
#     cfg.merge_from_list(args.opts)
#     # cfg.OUTPUT_DIR = args.OUTPUT_DIR
#     # cfg.freeze()
#     # cfg.MODEL.NO_MARGIN = False
#
#     set_seed(1)
#
#     if cfg.MODEL.DIST_TRAIN:
#         torch.cuda.set_device(args.local_rank)
#
#     output_dir = cfg.OUTPUT_DIR
#     cur_time = time.strftime('%Y-%m-%d-%H-%M-%S')
#     output_dir = os.path.join(output_dir, cur_time)
#     cfg.OUTPUT_DIR = output_dir
#
#     # cfg.OUTPUT_DIR = '/mnt/data/code/reidlog/transreid/test'
#     # output_dir = cfg.OUTPUT_DIR
#
#     if output_dir and not os.path.exists(output_dir):
#         os.makedirs(output_dir)
#
#     sys.stdout = Logger(os.path.join(output_dir, 'mytrainlog.log'))
#
#     logger = setup_logger("transreid", output_dir, if_train=True)
#     logger.info("Saving model in the path :{}".format(cfg.OUTPUT_DIR))
#     logger.info(args)
#
#     if args.config_file != "":
#         logger.info("Loaded configuration file {}".format(args.config_file))
#         with open(args.config_file, 'r') as cf:
#             config_str = "\n" + cf.read()
#             logger.info(config_str)
#     logger.info("Running with config:\n{}".format(cfg))
#
#     if cfg.MODEL.DIST_TRAIN:
#         torch.distributed.init_process_group(backend='nccl', init_method='env://')
#
#     os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID
#     train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num, train_transform = make_dataloader(
#         cfg)
#     # Image Preprocessing
#     # mean = [x / 255.0 for x in [125.3, 123.0, 113.9]]
#     # std = [x / 255.0 for x in [63.0, 62.1, 66.7]]
#     mean = [0.5, 0.5, 0.5]
#     std = [0.5, 0.5, 0.5]
#     # mlr_dukemtmc
#     # mean = [0.4363, 0.4283, 0.4411]
#     # std = [0.2324, 0.2342, 0.2228]
#     normalize = transforms.Normalize(mean, std)
#
#     keep = None
#     if args.method == 'keep_cutout':
#         keep = Keep_Cutout(train_transform, mean, std, args.length, prob_happen=0.5)
#     elif args.method == 'keep_cutout_vit_explain':
#         keep = Keep_Cutout_vit_explain(train_transform, mean, std, args.length, prob_happen=0.5)
#     elif args.method == 'Keep_Cutout_vit_explain_randomsize':
#         keep = Keep_Cutout_vit_explain_randomsize(train_transform, mean, std, args.length, prob_happen=0.5,
#                                                   patch_min_area=0.1,
#                                                   patch_max_area=0.5,
#                                                   patch_min_ratio=0.1, )
#     elif args.method == 'keep_cutout_low':
#         keep = Keep_Cutout_Low(train_transform, mean, std, args.length)
#     elif args.method == 'keep_cutout_early':
#         keep = Keep_Cutout(train_transform, mean, std, args.length, True)
#     elif args.method == 'keep_cutout_low_early':
#         keep = Keep_Cutout_Low(train_transform, mean, std, args.length, True)
#
#     # elif args.method == 'keep_autoaugment':
#     #     keep = Keep_Autoaugment(train_transform, mean, std, args.length, args.N, args.M)
#     # elif args.method == 'keep_autoaugment_low':
#     #     keep = Keep_Autoaugment_Low(train_transform, mean, std, args.length, args.N, args.M)
#     # elif args.method == 'keep_autoaugment_early':
#     #     keep = Keep_Autoaugment(train_transform, mean, std, args.length, args.N, args.M, True)
#     # elif args.method == 'keep_autoaugment_low_early':
#     #     keep = Keep_Autoaugment_Low(train_transform, mean, std, args.length, args.N, args.M, True)
#
#     model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num)
#
#     # model.load_param('/mnt/data/code/reidlog/transreid/2022-07-14-23-41-58/transformer_best')
#     # model = model.cuda()
#     # model.eval()
#     # img = Image.open("/mnt/data/datasets/DukeMTMC-reID/MLR_DukeMTMC/bounding_box_train/0015_c1_f0048365.jpg")
#     # imgNormal = transforms.Compose([
#     #     transforms.Resize((256, 128), interpolation=3),
#     #     transforms.ToTensor(),
#     #     transforms.Normalize(mean, std),
#     # ])
#     # img_dir = '/mnt/data/datasets/DukeMTMC-reID/MLR_DukeMTMC/query'
#     # imgnames = os.listdir(img_dir)
#     # savedir = os.path.join(cfg.OUTPUT_DIR, "IMGS")
#     # os.makedirs(savedir, exist_ok=True)
#     # # for imgname in imgnames:
#     # # imgpath = os.path.join(img_dir, imgname)
#     # # img = Image.open(imgpath)
#     #
#     # attention_rollout = VITAttentionRollout(model, head_fusion='max',
#     #                                         discard_ratio=0)
#     # normal_img = imgNormal(img).unsqueeze(0)
#     # normal_img = normal_img.cuda()
#     # mask = attention_rollout(normal_img, target_cam=1, target_view=1)
#     # img = img.resize((128, 256))
#     # np_img = np.array(img)[:, :, ::-1]
#     # mask = cv2.resize(mask, (np_img.shape[1], np_img.shape[0]))
#     # print((mask > 0.6).sum())
#     # mask = show_mask_on_image(np_img, mask)
#     # # cv2.imwrite(os.path.join(savedir, imgname), mask)
#     # cv2.imshow("Input Image", np_img)
#     # cv2.imshow('test', mask)
#     # # cv2.imwrite("input.png", np_img)
#     # cv2.imwrite('vit_explain/test.png', mask)
#     # cv2.waitKey(-1)
#
#     #
#     # writer = SummaryWriter("/mnt/data/code/reidlog/transreid/viewdata/random_copy_patch")
#     # writer = SummaryWriter(cfg.OUTPUT_DIR)
#     # for epoch in range(300):
#     #     model.cuda()
#     #     device = "cuda"
#     #     for n_iter, (img, vid, target_cam, target_view) in enumerate(train_loader):
#     #         img = img.to(device)
#     #         target = vid.to(device)
#     #         target_cam = target_cam.to(device)
#     #         target_view = target_view.to(device)
#     #         if args.method != "":
#     #             img = keep(img, model, target_cam, target_view)
#     #             # img, maskimg= keep(img, model, target_cam, target_view)
#     #         print('epoch: {}'.format(epoch))
#     #         writer.add_images('figure', img, epoch+1)
#     #         # writer.add_images('mask', maskimg, epoch + 1)
#     #         break
#
#     loss_func, center_criterion = make_loss(cfg, num_classes=num_classes)
#
#     optimizer, optimizer_center = make_optimizer(cfg, model, center_criterion)
#
#     scheduler = create_scheduler(cfg, optimizer)
#
#
#
#     do_train(
#         cfg, args,
#         model,
#         center_criterion,
#         train_loader,
#         val_loader,
#         optimizer,
#         optimizer_center,
#         scheduler,
#         loss_func,
#         num_query, args.local_rank,
#         keep=keep,
#     )