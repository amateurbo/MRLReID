import math
import random
import time

import torch
import numpy as np
import torchvision.transforms as transforms
import torchvision.transforms as T
import torch.nn.functional as F
from utils.vit_explain import VITAttentionRollout, show_mask_on_image
from numba import jit



class Keep_Cutout_Low(object):
    def __init__(self, train_transform, mean, std, length, early=False):
        self.trans = train_transform
        self.length = int(length / 2)
        self.early = early
        self.denomal = transforms.Compose([
            transforms.Normalize((-mean[0] / std[0], -mean[1] / std[1], -mean[2] / std[2]),
                                 (1 / std[0], 1 / std[1], 1 / std[2])),
            transforms.ToPILImage()
        ])


    def __call__(self, images, model, target_cam, target_view):

        for param in model.parameters():
            param.requires_grad = False
        model.eval()
        images_ = images.clone().detach()
        images_half = F.interpolate(images_, scale_factor=0.5, mode='bicubic', align_corners=True)
        images_half.requires_grad = True

        if self.early:
            preds = model(images_half, True)
        else:
            preds = model(images_half, cam_label=target_cam, view_label=target_view)

        score, _ = torch.max(preds, 1)
        score.mean().backward()
        slc_, _ = torch.max(torch.abs(images_half.grad), dim=1)

        b, h, w = slc_.shape
        slc_ = slc_.view(slc_.size(0), -1)
        slc_ -= slc_.min(1, keepdim=True)[0]
        slc_ /= slc_.max(1, keepdim=True)[0]
        slc_ = slc_.view(b, h, w)

        for i, (img, slc) in enumerate(zip(images_, slc_)):
            mask = np.ones((h * 2, w * 2), np.float32)
            while (True):
                y = np.random.randint(h)
                x = np.random.randint(w)

                y1 = np.clip(y - self.length // 2, 0, h)
                y2 = np.clip(y + self.length // 2, 0, h)
                x1 = np.clip(x - self.length // 2, 0, w)
                x2 = np.clip(x + self.length // 2, 0, w)

                if slc[y1: y2, x1: x2].mean() < 0.6:
                    mask[y1 * 2: y2 * 2, x1 * 2: x2 * 2] = 0.
                    break

            mask = torch.from_numpy(mask)
            mask = mask.expand_as(img).cuda()
            img = img * mask
            images[i] = self.trans(self.denomal(img))

        model.train()
        for param in model.parameters():
            param.requires_grad = True
        return images.cuda()

from timm.data.random_erasing import RandomErasing
class Keep_Cutout(object):
    def __init__(self, train_transform, mean, std, length, prob_happen=0.5, early=False):
        self.trans = train_transform
        self.length = length
        self.early = early
        self.prob_happen = prob_happen
        self.denomal = transforms.Compose([
            transforms.Normalize((-mean[0] / std[0], -mean[1] / std[1], -mean[2] / std[2]),
                                 (1 / std[0], 1 / std[1], 1 / std[2])),
            # transforms.Normalize(mean, std),
            transforms.ToPILImage()
        ])
        self.erase = transforms.Compose([
            transforms.Normalize((-mean[0] / std[0], -mean[1] / std[1], -mean[2] / std[2]),
                                 (1 / std[0], 1 / std[1], 1 / std[2])),
            transforms.ToPILImage(),
            # T.Resize((256, 128), interpolation=3),
            # T.RandomHorizontalFlip(p=0.5),
            # T.Pad(10),
            # T.RandomCrop((256, 128)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
            RandomErasing(probability=0.5, mode='pixel', max_count=1, device='cpu'),
        ])
        print(self.erase)

    def __call__(self, images, model, target_cam, target_view):
        for param in model.parameters():
            param.requires_grad = False
        model.eval()
        images_ = images.clone().detach()
        images_.requires_grad = True

        if self.early:
            preds = model(images_, True)
        else:
            preds = model(images_, cam_label=target_cam, view_label=target_view)

        score, _ = torch.max(preds, 1)
        score.mean().backward()
        slc_, _ = torch.max(torch.abs(images_.grad), dim=1)

        b, h, w = slc_.shape

        slc_ = slc_.view(slc_.size(0), -1)
        slc_ -= slc_.min(1, keepdim=True)[0]
        slc_ /= slc_.max(1, keepdim=True)[0]
        slc_ = slc_.view(b, h, w)

        for i, (img, slc) in enumerate(zip(images_, slc_)):
            mask = np.ones((h, w), np.float32)
            if random.uniform(0, 1) > self.prob_happen:
                images[i] = self.erase(img.detach())
                continue
            # n = 0
            while (True):
                # n += 1
                y = np.random.randint(h)
                x = np.random.randint(w)
                #
                # l1 = int(random.uniform(self.length // 4, self.length // 4 * 3))
                # l2 = self.length - l1
                # y1 = np.clip(y - l1, 0, h)
                # y2 = np.clip(y + l1, 0, h)
                # x1 = np.clip(x - l2, 0, w)
                # x2 = np.clip(x + l2, 0, w)

                y1 = np.clip(y - self.length // 2, 0, h)
                y2 = np.clip(y + self.length // 2, 0, h)
                x1 = np.clip(x - self.length // 2, 0, w)
                x2 = np.clip(x + self.length // 2, 0, w)

                # print(slc[y1: y2, x1: x2].mean())
                if slc[y1: y2, x1: x2].mean() < 0.15:
                    mask[y1: y2, x1: x2] = 0.
                    # print(num)
                    break
            # print(n)
            mask = torch.from_numpy(mask)
            mask = mask.expand_as(img).cuda()
            img = img * mask
            # images[i] = self.trans(self.denomal(img))
            # images[i] = self.trans(img)

            images[i] = self.erase(img.detach())
            # print(img.detach())
            # print(images[i])
            # images[i] = img.detach()

        model.train()
        for param in model.parameters():
            param.requires_grad = True
        return images.cuda()

class Keep_Cutout_vit_explain(object):
    def __init__(self, train_transform, mean, std, length, prob_happen=0.5, early=False):
        self.trans = train_transform
        self.length = length
        self.early = early
        self.prob_happen = prob_happen
        self.denomal = transforms.Compose([
            transforms.Normalize((-mean[0] / std[0], -mean[1] / std[1], -mean[2] / std[2]),
                                 (1 / std[0], 1 / std[1], 1 / std[2])),
            transforms.ToPILImage()
        ])
        self.erase = transforms.Compose([
            transforms.Normalize((-mean[0] / std[0], -mean[1] / std[1], -mean[2] / std[2]),
                                 (1 / std[0], 1 / std[1], 1 / std[2])),
            transforms.ToPILImage(),
            # T.Resize((256, 128), interpolation=3),
            # T.RandomHorizontalFlip(p=0.5),
            # T.Pad(10),
            # T.RandomCrop((256, 128)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
            RandomErasing(probability=0.5, mode='pixel', max_count=1, device='cpu'),
        ])
        print(self.erase)


    def __call__(self, images, model, target_cam, target_view):

        for param in model.parameters():
            param.requires_grad = False
        model.eval()
        images_ = images.clone().detach()
        images_.requires_grad = True

        # start = time.time()
        attention_rollout = VITAttentionRollout(model, head_fusion='max',
                                                discard_ratio=0.7)
        slc_ = attention_rollout(images, target_cam=target_cam, target_view=target_view)

        # print("ont iter vit_explain cost time: {}".format(time.time() - start))

        h = 256
        w = 128
        # mask_imgs = torch.ones((64, 3, 256, 128))
        slc_ = slc_.resize_(slc_.size(0), h, w)
        # npslc = slc_.numpy()

        for i, (img, slc) in enumerate(zip(images_, slc_)):

            mask = np.ones((h, w), np.float32)
            # mask_img = show_mask_on_image(img.cpu(), slc.detach().numpy())
            if random.uniform(0, 1) > self.prob_happen:
                images[i] = self.erase(img.detach())
                continue

            while (True):
                y = np.random.randint(h)
                x = np.random.randint(w)
                #
                # l1 = int(random.uniform(self.length // 4, self.length // 4 * 3))
                # l2 = self.length - l1
                # y1 = np.clip(y - l1, 0, h)
                # y2 = np.clip(y + l1, 0, h)
                # x1 = np.clip(x - l2, 0, w)
                # x2 = np.clip(x + l2, 0, w)

                y1 = np.clip(y - self.length // 2, 0, h)
                y2 = np.clip(y + self.length // 2, 0, h)
                x1 = np.clip(x - self.length // 2, 0, w)
                x2 = np.clip(x + self.length // 2, 0, w)

                # print(slc.mean())
                # print(slc[y1: y2, x1: x2].mean())
                # print((slc[y1: y2, x1: x2] > 0.5).sum())
                if slc[y1: y2, x1: x2].mean() < slc.mean():
                # if True:
                    mask[y1: y2, x1: x2] = 0.
                    # print(num)
                    break


            mask = torch.from_numpy(mask)
            mask = mask.expand_as(img).cuda()
            img = img * mask
            # images[i] = self.trans(self.denomal(img))
            # images[i] = self.trans(img)
            # mask_imgs[i] = mask_img

            images[i] = self.erase(img.detach())
            # print(img.detach())
            # print(images[i])
            # images[i] = img.detach()

        model.train()
        for param in model.parameters():
            param.requires_grad = True
        return images.cuda()


class Keep_Cutout_vit_explain_randomsize(object):
    def __init__(self,
            train_transform,
            mean,
            std,
            length,
            prob_happen=0.5,
            early=False,
            patch_min_area = 0.1,
            patch_max_area = 0.5,
            patch_min_ratio = 0.1,
        ):
        self.trans = train_transform
        self.length = length
        self.early = early
        self.prob_happen = prob_happen
        self.patch_min_area = patch_min_area
        self.patch_max_area = patch_max_area
        self.patch_min_ratio = patch_min_ratio
        self.denomal = transforms.Compose([
            transforms.Normalize((-mean[0] / std[0], -mean[1] / std[1], -mean[2] / std[2]),
                                 (1 / std[0], 1 / std[1], 1 / std[2])),
            transforms.ToPILImage()
        ])
        self.erase = transforms.Compose([
            transforms.Normalize((-mean[0] / std[0], -mean[1] / std[1], -mean[2] / std[2]),
                                 (1 / std[0], 1 / std[1], 1 / std[2])),
            transforms.ToPILImage(),
            # T.Resize((256, 128), interpolation=3),
            # T.RandomHorizontalFlip(p=0.5),
            # T.Pad(10),
            # T.RandomCrop((256, 128)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
            RandomErasing(probability=0.5, mode='pixel', max_count=1, device='cpu'),
        ])
        print(self.erase)

    def __call__(self, images, model, target_cam, target_view):

        for param in model.parameters():
            param.requires_grad = False
        model.eval()
        images_ = images.clone().detach()
        images_.requires_grad = True

        attention_rollout = VITAttentionRollout(model, head_fusion='max',
                                                discard_ratio=0.7)
        slc_ = attention_rollout(images, target_cam=target_cam, target_view=target_view)

        h = 256
        w = 128

        slc_ = slc_.resize_(slc_.size(0), h, w)

        for i, (img, slc) in enumerate(zip(images_, slc_)):

            mask = np.ones((h, w), np.float32)
            # mask_img = show_mask_on_image(img.cpu(), slc.detach().numpy())
            if random.uniform(0, 1) > self.prob_happen:
                images[i] = self.erase(img.detach())
                continue

            while (True):
                y = np.random.randint(h)
                x = np.random.randint(w)

                l1 = int(random.uniform(self.length // 4, self.length // 4 * 3))
                l2 = self.length - l1
                y1 = np.clip(y - l1, 0, h)
                y2 = np.clip(y + l1, 0, h)
                x1 = np.clip(x - l2, 0, w)
                x2 = np.clip(x + l2, 0, w)

                if slc[y1: y2, x1: x2].mean() < slc.mean():
                    mask[y1: y2, x1: x2] = 0.
                    break

            mask = torch.from_numpy(mask)
            mask = mask.expand_as(img).cuda()
            img = img * mask

            images[i] = self.erase(img.detach())


        model.train()
        for param in model.parameters():
            param.requires_grad = True
        return images.cuda()