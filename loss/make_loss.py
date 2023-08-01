# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch.nn.functional as F
from .softmax_loss import CrossEntropyLabelSmooth, LabelSmoothingCrossEntropy
from .triplet_loss import TripletLoss
from .center_loss import CenterLoss
from .RMSE_loss import RMSELoss

rmseloss = RMSELoss()

def make_loss(cfg, num_classes):    # modified by gu
    sampler = cfg.DATALOADER.SAMPLER
    feat_dim = 1024
    center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True)  # center loss
    if 'triplet' in cfg.MODEL.METRIC_LOSS_TYPE:
        if cfg.MODEL.NO_MARGIN:
            triplet = TripletLoss()
            print("using soft triplet loss for training")
        else:
            triplet = TripletLoss(cfg.SOLVER.MARGIN)  # triplet loss
            print("using triplet loss with margin:{}".format(cfg.SOLVER.MARGIN))
    else:
        print('expected METRIC_LOSS_TYPE should be triplet'
              'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    if cfg.MODEL.IF_LABELSMOOTH == 'on':
        xent = CrossEntropyLabelSmooth(num_classes=num_classes)
        print("label smooth on, numclasses:", num_classes)

    if sampler == 'softmax':
        def loss_func(score, feat, target):
            return F.cross_entropy(score, target)
    elif sampler == 'rmse_softmax_triplet':
        def loss_func(res_score, res_target):
            RES_LOSS = rmseloss(res_score.squeeze(1), res_target)
            return RES_LOSS

    elif cfg.DATALOADER.SAMPLER == 'softmax_triplet':
        def loss_func(score, feat, target, target_cam):
            if cfg.MODEL.METRIC_LOSS_TYPE == 'triplet':
                if cfg.MODEL.IF_LABELSMOOTH == 'on':
                    if isinstance(score, list):
                        ID_LOSS = [xent(scor, target) for scor in score[1:]]
                        ID_LOSS = sum(ID_LOSS) / len(ID_LOSS)
                        ID_LOSS = 0.5 * ID_LOSS + 0.5 * xent(score[0], target)
                    else:
                        ID_LOSS = xent(score, target)

                    if isinstance(feat, list):
                            TRI_LOSS = [triplet(feats, target)[0] for feats in feat[1:]]
                            TRI_LOSS = sum(TRI_LOSS) / len(TRI_LOSS)
                            TRI_LOSS = 0.5 * TRI_LOSS + 0.5 * triplet(feat[0], target)[0]
                    else:
                            TRI_LOSS = triplet(feat, target)[0]

                    return cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + \
                               cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS
                else:
                    if isinstance(score, list):
                        ID_LOSS = [F.cross_entropy(scor, target) for scor in score[1:]]
                        ID_LOSS = sum(ID_LOSS) / len(ID_LOSS)
                        ID_LOSS = 0.5 * ID_LOSS + 0.5 * F.cross_entropy(score[0], target)
                    else:
                        ID_LOSS = F.cross_entropy(score, target)

                    if isinstance(feat, list):
                            TRI_LOSS = [triplet(feats, target)[0] for feats in feat[1:]]
                            TRI_LOSS = sum(TRI_LOSS) / len(TRI_LOSS)
                            TRI_LOSS = 0.5 * TRI_LOSS + 0.5 * triplet(feat[0], target)[0]
                    else:
                            TRI_LOSS = triplet(feat, target)[0]

                    return cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + \
                               cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS
            else:
                print('expected METRIC_LOSS_TYPE should be triplet'
                      'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    elif cfg.DATALOADER.SAMPLER == 'softmax2_triplet':
        def loss_func(score, feat, target, target_cam, res_score, res_target):
            if cfg.MODEL.METRIC_LOSS_TYPE == 'triplet':
                if cfg.MODEL.IF_LABELSMOOTH == 'on':
                    if isinstance(score, list):
                        ID_LOSS = [xent(scor, target) for scor in score[1:]]
                        ID_LOSS = sum(ID_LOSS) / len(ID_LOSS)
                        ID_LOSS = 0.5 * ID_LOSS + 0.5 * xent(score[0], target)
                    else:
                        ID_LOSS = xent(score, target)

                    if isinstance(feat, list):
                            TRI_LOSS = [triplet(feats, target)[0] for feats in feat[1:]]
                            TRI_LOSS = sum(TRI_LOSS) / len(TRI_LOSS)
                            TRI_LOSS = 0.5 * TRI_LOSS + 0.5 * triplet(feat[0], target)[0]
                    else:
                            TRI_LOSS = triplet(feat, target)[0]

                    return cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + \
                               cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS
                else:
                    if isinstance(score, list):
                        ID_LOSS = [F.cross_entropy(scor, target) for scor in score[1:]]
                        ID_LOSS = sum(ID_LOSS) / len(ID_LOSS)
                        ID_LOSS = 0.5 * ID_LOSS + 0.5 * F.cross_entropy(score[0], target)
                    else:
                        ID_LOSS = F.cross_entropy(score, target)

                    # if isinstance(res_score, list):
                    #
                    #     RES_LOSS = [F.cross_entropy(scor, res_target) for scor in res_score[1:]]
                    #     RES_LOSS = sum(RES_LOSS) / len(RES_LOSS)
                    #     RES_LOSS = 0.5 * RES_LOSS + 0.5 * F.cross_entropy(res_score[0], res_target)
                    # else:
                    #     RES_LOSS = F.cross_entropy(res_score, res_target)

                    RES_LOSS = rmseloss(res_score.squeeze(1), res_target)

                    if isinstance(feat, list):
                        if cfg.MODEL.RES_TRI_NUM == 1:
                            RES_NUM = cfg.MODEL.RES_TRI_NUM
                            TRI_RES_LOSS = [triplet(feats, target)[0] for feats in feat[-RES_NUM:]]
                            TRI_RES_LOSS = sum(TRI_RES_LOSS) / len(TRI_RES_LOSS)
                            TRI_LOSS = [triplet(feats, target)[0] for feats in feat[1:-RES_NUM]]
                            TRI_LOSS = sum(TRI_LOSS) / len(TRI_LOSS)
                            # local loss + global loss + resolution loss
                            # TRI_LOSS = 0.5 * TRI_LOSS + 0.5 * triplet(feat[0], target)[0] + cfg.MODEL.RESOLUTION_TRILOSS_WEIGHT * TRI_RES_LOSS
                            TRI_LOSS = (TRI_LOSS + triplet(feat[0], target)[0] + TRI_RES_LOSS) / 3
                        elif cfg.MODEL.RES_TRI_NUM == 2:
                            TRI_RES_LOSS_LOCAL = [triplet(feats, target)[0] for feats in feat[-4:]]
                            TRI_RES_LOSS_LOCAL = sum(TRI_RES_LOSS_LOCAL) / len(TRI_RES_LOSS_LOCAL)
                            TRI_RES_LOSS_LOCAL = TRI_RES_LOSS_LOCAL + triplet(feat[-5], target)[0]

                            TRI_LOSS = [triplet(feats, target)[0] for feats in feat[1:-5]]
                            TRI_LOSS = sum(TRI_LOSS) / len(TRI_LOSS)
                            TRI_LOSS = (TRI_LOSS + triplet(feat[0], target)[0] + TRI_RES_LOSS_LOCAL) * 0.5
                        else:
                            TRI_LOSS = [triplet(feats, target)[0] for feats in feat[1:]]
                            TRI_LOSS = sum(TRI_LOSS) / len(TRI_LOSS)
                            TRI_LOSS = 0.5 * TRI_LOSS + 0.5 * triplet(feat[0], target)[0]
                    else:
                        TRI_LOSS = triplet(feat, target)[0]
                        # print('feat: {}, target: {}, TRI_LOSS: {}'.format(feat, target, TRI_LOSS))

                    # print("ID_LOSS: {},  TRI_LOSS: {}  ,RES_LOSS:{} ".format(ID_LOSS, TRI_LOSS, RES_LOSS))
                    return cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + \
                               cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS + cfg.MODEL.RESOLUTION_LOSS_WEIGHT * RES_LOSS
            else:
                print('expected METRIC_LOSS_TYPE should be triplet'
                      'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    elif cfg.DATALOADER.SAMPLER == 'softmax2_triplet2':
        def loss_func(score, feat, target, target_cam, res_score, res_feat, res_target):
            if cfg.MODEL.METRIC_LOSS_TYPE == 'triplet':
                if cfg.MODEL.IF_LABELSMOOTH == 'on':
                    if isinstance(score, list):
                        ID_LOSS = [xent(scor, target) for scor in score[1:]]
                        ID_LOSS = sum(ID_LOSS) / len(ID_LOSS)
                        ID_LOSS = 0.5 * ID_LOSS + 0.5 * xent(score[0], target)
                    else:
                        ID_LOSS = xent(score, target)

                    if isinstance(feat, list):
                            TRI_LOSS = [triplet(feats, target)[0] for feats in feat[1:]]
                            TRI_LOSS = sum(TRI_LOSS) / len(TRI_LOSS)
                            TRI_LOSS = 0.5 * TRI_LOSS + 0.5 * triplet(feat[0], target)[0]
                    else:
                            TRI_LOSS = triplet(feat, target)[0]

                    return cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + \
                               cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS
                else:
                    if isinstance(score, list):
                        ID_LOSS = [F.cross_entropy(scor, target) for scor in score[1:]]
                        ID_LOSS = sum(ID_LOSS) / len(ID_LOSS)
                        ID_LOSS = 0.5 * ID_LOSS + 0.5 * F.cross_entropy(score[0], target)
                    else:
                        ID_LOSS = F.cross_entropy(score, target)

                    if isinstance(res_score, list):
                        RES_LOSS = [F.cross_entropy(scor, res_target) for scor in res_score[1:]]
                        RES_LOSS = sum(RES_LOSS) / len(RES_LOSS)
                        RES_LOSS = 0.5 * RES_LOSS + 0.5 * F.cross_entropy(res_score[0], res_target)
                    else:
                        RES_LOSS = F.cross_entropy(res_score, res_target)

                    if isinstance(feat, list):
                        TRI_LOSS = [triplet(feats, target)[0] for feats in feat[1:]]
                        TRI_LOSS = sum(TRI_LOSS) / len(TRI_LOSS)
                        TRI_LOSS = 0.5 * TRI_LOSS + 0.5 * triplet(feat[0], target)[0]
                    else:
                        TRI_LOSS = triplet(feat, target)[0]

                    if isinstance(res_feat, list):
                        RES_TRI_LOSS = [triplet(feats, target)[0] for feats in res_feat[1:]]
                        RES_TRI_LOSS = sum(RES_TRI_LOSS) / len(RES_TRI_LOSS)
                        RES_TRI_LOSS = 0.5 * RES_TRI_LOSS + 0.5 * triplet(res_feat[0], target)[0]
                    else:
                        RES_TRI_LOSS = triplet(res_feat, target)[0]

                    return cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + \
                               cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS + cfg.MODEL.RESOLUTION_LOSS_WEIGHT * RES_LOSS + \
                           cfg.MODEL.RESOLUTION_LOSS_WEIGHT * RES_TRI_LOSS

            else:
                print('expected METRIC_LOSS_TYPE should be triplet'
                      'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    elif cfg.DATALOADER.SAMPLER == 'softmax3_triplet2':
        def loss_func(score, feat, target, target_cam, res_score, res_target, part_score, part_feat):
            if cfg.MODEL.METRIC_LOSS_TYPE == 'triplet':
                if cfg.MODEL.IF_LABELSMOOTH == 'on':
                    if isinstance(score, list):
                        ID_LOSS = [xent(scor, target) for scor in score[1:]]
                        ID_LOSS = sum(ID_LOSS) / len(ID_LOSS)
                        ID_LOSS = 0.5 * ID_LOSS + 0.5 * xent(score[0], target)
                    else:
                        ID_LOSS = xent(score, target)

                    if isinstance(feat, list):
                            TRI_LOSS = [triplet(feats, target)[0] for feats in feat[1:]]
                            TRI_LOSS = sum(TRI_LOSS) / len(TRI_LOSS)
                            TRI_LOSS = 0.5 * TRI_LOSS + 0.5 * triplet(feat[0], target)[0]
                    else:
                            TRI_LOSS = triplet(feat, target)[0]

                    return cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + \
                               cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS
                else:
                    if isinstance(score, list):
                        ID_LOSS = [F.cross_entropy(scor, target) for scor in score[1:]]
                        ID_LOSS = sum(ID_LOSS) / len(ID_LOSS)
                        ID_LOSS = 0.5 * ID_LOSS + 0.5 * F.cross_entropy(score[0], target)
                    else:
                        ID_LOSS = F.cross_entropy(score, target)

                    if isinstance(res_score, list):
                        RES_LOSS = [F.cross_entropy(scor, res_target) for scor in res_score[1:]]
                        RES_LOSS = sum(RES_LOSS) / len(RES_LOSS)
                        RES_LOSS = 0.5 * RES_LOSS + 0.5 * F.cross_entropy(res_score[0], res_target)
                    else:
                        RES_LOSS = F.cross_entropy(res_score, res_target)

                    if isinstance(part_score, list):
                        PART_LOSS = [F.cross_entropy(scor, target) for scor in part_score[1:]]
                        PART_LOSS = sum(PART_LOSS) / len(PART_LOSS)
                        PART_LOSS = 0.5 * PART_LOSS + 0.5 * F.cross_entropy(part_score[0], target)
                    else:
                        PART_LOSS = F.cross_entropy(part_score, target)


                    if isinstance(feat, list):
                            TRI_LOSS = [triplet(feats, target)[0] for feats in feat[1:]]
                            TRI_LOSS = sum(TRI_LOSS) / len(TRI_LOSS)
                            TRI_LOSS = 0.5 * TRI_LOSS + 0.5 * triplet(feat[0], target)[0]
                    else:
                            TRI_LOSS = triplet(feat, target)[0]
                            # print('feat: {}, target: {}, TRI_LOSS: {}'.format(feat, target, TRI_LOSS))

                    if isinstance(part_feat, list):
                            TRI_PART_LOSS = [triplet(feats, target)[0] for feats in part_feat[1:]]
                            TRI_PART_LOSS = sum(TRI_PART_LOSS) / len(TRI_PART_LOSS)
                            TRI_PART_LOSS = 0.5 * TRI_PART_LOSS + 0.5 * triplet(part_feat[0], target)[0]
                    else:
                            TRI_PART_LOSS = triplet(part_feat, target)[0]

                    return cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + \
                               cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS + cfg.MODEL.RESOLUTION_LOSS_WEIGHT * RES_LOSS + \
                               cfg.MODEL.TRI_PART_LOSS_WEIGHT * TRI_PART_LOSS + cfg.MODEL.PART_LOSS_WEIGHT * PART_LOSS
            else:
                print('expected METRIC_LOSS_TYPE should be triplet'
                      'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    elif cfg.DATALOADER.SAMPLER == 'softmax2_triplet2':
        def loss_func(score, feat, target, target_cam, part_score, part_feat):
            if cfg.MODEL.METRIC_LOSS_TYPE == 'triplet':
                if cfg.MODEL.IF_LABELSMOOTH == 'on':
                    if isinstance(score, list):
                        ID_LOSS = [xent(scor, target) for scor in score[1:]]
                        ID_LOSS = sum(ID_LOSS) / len(ID_LOSS)
                        ID_LOSS = 0.5 * ID_LOSS + 0.5 * xent(score[0], target)
                    else:
                        ID_LOSS = xent(score, target)

                    if isinstance(feat, list):
                            TRI_LOSS = [triplet(feats, target)[0] for feats in feat[1:]]
                            TRI_LOSS = sum(TRI_LOSS) / len(TRI_LOSS)
                            TRI_LOSS = 0.5 * TRI_LOSS + 0.5 * triplet(feat[0], target)[0]
                    else:
                            TRI_LOSS = triplet(feat, target)[0]

                    return cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + \
                               cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS
                else:
                    if isinstance(score, list):
                        ID_LOSS = [F.cross_entropy(scor, target) for scor in score[1:]]
                        ID_LOSS = sum(ID_LOSS) / len(ID_LOSS)
                        ID_LOSS = 0.5 * ID_LOSS + 0.5 * F.cross_entropy(score[0], target)
                    else:
                        ID_LOSS = F.cross_entropy(score, target)

                    # if isinstance(res_score, list):
                    #     RES_LOSS = [F.cross_entropy(scor, res_target) for scor in res_score[1:]]
                    #     RES_LOSS = sum(RES_LOSS) / len(RES_LOSS)
                    #     RES_LOSS = 0.5 * RES_LOSS + 0.5 * F.cross_entropy(res_score[0], res_target)
                    # else:
                    #     RES_LOSS = F.cross_entropy(res_score, res_target)

                    if isinstance(part_score, list):
                        PART_LOSS = [F.cross_entropy(scor, target) for scor in part_score[1:]]
                        PART_LOSS = sum(PART_LOSS) / len(PART_LOSS)
                        PART_LOSS = 0.5 * PART_LOSS + 0.5 * F.cross_entropy(part_score[0], target)
                    else:
                        PART_LOSS = F.cross_entropy(part_score, target)


                    if isinstance(feat, list):
                            TRI_LOSS = [triplet(feats, target)[0] for feats in feat[1:]]
                            TRI_LOSS = sum(TRI_LOSS) / len(TRI_LOSS)
                            TRI_LOSS = 0.5 * TRI_LOSS + 0.5 * triplet(feat[0], target)[0]
                    else:
                            TRI_LOSS = triplet(feat, target)[0]
                            # print('feat: {}, target: {}, TRI_LOSS: {}'.format(feat, target, TRI_LOSS))

                    if isinstance(part_feat, list):
                            TRI_PART_LOSS = [triplet(feats, target)[0] for feats in part_feat[1:]]
                            TRI_PART_LOSS = sum(TRI_PART_LOSS) / len(TRI_PART_LOSS)
                            TRI_PART_LOSS = 0.5 * TRI_PART_LOSS + 0.5 * triplet(part_feat[0], target)[0]
                    else:
                            TRI_PART_LOSS = triplet(part_feat, target)[0]

                    return cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + \
                               cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS + \
                               cfg.MODEL.TRI_PART_LOSS_WEIGHT * TRI_PART_LOSS + cfg.MODEL.PART_LOSS_WEIGHT * PART_LOSS
            else:
                print('expected METRIC_LOSS_TYPE should be triplet'
                      'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    else:
        print('expected sampler should be softmax, triplet, softmax_triplet or softmax_triplet_center'
              'but got {}'.format(cfg.DATALOADER.SAMPLER))
    return loss_func, center_criterion


