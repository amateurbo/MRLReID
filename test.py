import os

import cv2
import numpy
import torch

from config import cfg
import argparse
from datasets import make_dataloader
from model import make_model
from processor import do_inference
from utils.logger import setup_logger



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="./configs/MTA_reid/vit_transreid_stride_global_local_res.yml", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()



    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger, _ = setup_logger("transreid", output_dir, if_train=False)
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID

    # train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)
    train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num, train_transform = make_dataloader(
        cfg)

    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num = view_num)
    # #preweight_path = '/mnt/data/code/reidlog_3/mta_reid/global_local_res_lr008_lambda0_10/transformer_120.pth'transformer_best_Rank1
    # preweight_path = '/mnt/data/code/reidlog_3/mta_reid/global_local_res_lr008_lambda0_10/transformer_120.pth'
    # param_dict = torch.load(preweight_path)
    # logger.info('Loaded pre weight {}'.format(preweight_path))
    # if 'model' in param_dict.keys():
    #     del param_dict['model']['base.sie_embed']
    #     del param_dict['model']['classifier.weight']
    #     del param_dict['model']['classifier_1.weight']
    #     del param_dict['model']['classifier_2.weight']
    #     del param_dict['model']['classifier_3.weight']
    #     del param_dict['model']['classifier_4.weight']
    #     model.load_state_dict(param_dict['model'], strict=False)
    # else:
    #     model.load_state_dict(param_dict, strict=True)
    model.load_param(cfg.TEST.WEIGHT)



    if cfg.DATASETS.NAMES == 'VehicleID':
        for trial in range(10):
            train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)
            rank_1, rank5 = do_inference(cfg,
                 model,
                 val_loader,
                 num_query)
            if trial == 0:
                all_rank_1 = rank_1
                all_rank_5 = rank5
            else:
                all_rank_1 = all_rank_1 + rank_1
                all_rank_5 = all_rank_5 + rank5

            logger.info("rank_1:{}, rank_5 {} : trial : {}".format(rank_1, rank5, trial))
        logger.info("sum_rank_1:{:.1%}, sum_rank_5 {:.1%}".format(all_rank_1.sum()/10.0, all_rank_5.sum()/10.0))
    else:
       do_inference(cfg,
                 model,
                 val_loader,
                 num_query,)

