import math
from collections import deque

import cv2
import numpy as np
import torch
import torchvision
import torchvision.transforms as T
import random


from PIL import Image
from utils.random_erasing_patch import RandomErasing as myerase
from utils.random_erasing_patch import RandomBlur, RandomCutBlur, BodyBlur, RandomPatches
from torch.utils.data import DataLoader

from .bases import ImageDataset
from timm.data.random_erasing import RandomErasing
from .sampler import RandomIdentitySampler
from .dukemtmcreid import DukeMTMCreID
from .market1501 import Market1501
from .msmt17 import MSMT17
from .sampler_ddp import RandomIdentitySampler_DDP
import torch.distributed as dist
from .occ_duke import OCC_DukeMTMCreID
from .vehicleid import VehicleID
from .veri import VeRi
from .mlr_market1501 import MLR_Market1501
from .mta_reid import MTA_reid
from .mlr_dukemtmc import MLR_DukeMTMC
from .prai1581 import PRAI1581
from .mlr_msmt17 import MLR_MSMT17
from .caviar import CAVIAR
__factory = {
    'market1501': Market1501,
    'DukeMTMC': DukeMTMCreID,
    'msmt17': MSMT17,
    'occ_duke': OCC_DukeMTMCreID,
    'veri': VeRi,
    'VehicleID': VehicleID,
    'MLR_Market1501': MLR_Market1501,
    'MTA_reid': MTA_reid,
    'MLR_DukeMTMC': MLR_DukeMTMC,
    'PRAI1581': PRAI1581,
    'MLR_MSMT17': MLR_MSMT17,
    'CAVIAR': CAVIAR,
}

def train_collate_fn(batch):
    """
    # collate_fn这个函数的输入就是一个list，list的长度是一个batch size，list中的每个元素都是__getitem__得到的结果
    """
    imgs, pids, camids, viewids , _, imgsh, imgsw = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids = torch.tensor(camids, dtype=torch.int64)
    # imgsh = torch.tensor(imgsh, dtype=torch.int64)
    imgsh = torch.tensor(imgsh, dtype=torch.float16)
    imgsw = torch.tensor(imgsw, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, camids, viewids, imgsh, imgsw

def val_collate_fn(batch):
    imgs, pids, camids, viewids, img_paths, imgsh, imgsw = zip(*batch)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids_batch = torch.tensor(camids, dtype=torch.int64)
    # imgsh = torch.tensor(imgsh, dtype=torch.int64)
    imgsh = torch.tensor(imgsh, dtype=torch.float16)
    imgsw = torch.tensor(imgsw, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, camids, camids_batch, viewids, img_paths, imgsh, imgsw

class gaussian_blur(object):
    def __init__(
        self,
        prob_happen=0.5,
        param=5,
        # level=3,
    ):
        # params=[7, 9, 13, 17, 21]
        self.param = param
        self.prob_happen = prob_happen

    def __repr__(self):
        return self.__class__.__name__ + "(compress_param={0}, prob_happen={1})"\
            .format(self.param, self.prob_happen)

    def __call__(self, img):
        if random.uniform(0, 1) > self.prob_happen:
            return img

        img = np.array(img)
        img = cv2.GaussianBlur(img, (self.param, self.param), self.param * 1.0 / 6)
        img = Image.fromarray(img)
        return img

class jpeg_compression(object):
    def __init__(
        self,
        prob_happen=0.5,
        param=2,
    ):
        self.param = param
        self.prob_happen = prob_happen

    def __repr__(self):
        return self.__class__.__name__ + "(compress_param={0}, prob_happen={1})"\
            .format(self.param, self.prob_happen)

    def __call__(self, img):
        if random.uniform(0, 1) > self.prob_happen:
            return img

        img = np.array(img)

        h, w, _ = img.shape
        s_h = h // self.param
        s_w = w // self.param
        img = cv2.resize(img, (s_w, s_h))
        img = cv2.resize(img, (w, h))

        img = Image.fromarray(img)
        return img

class denoise(object):
    def __init__(
        self,
        p=0.5,
        param=1,
    ):
        self.param = param
        self.p = p

    def __call__(self, img):
        if random.uniform(0, 1) > self.p:
            return img
        img = np.array(img)
        img = cv2.fastNlMeansDenoisingColored(img, h=self.param, templateWindowSize=7, searchWindowSize=21)
        img = Image.fromarray(img)
        return img

class random_copypatch(object):
    def __init__(
        self,
        prob_happen=0.5,
        pool_capacity=500,
        patch_size=16,
        patch_num=8,
        prob_rotate=0.5,
        prob_flip_leftright=0.5,
        transpatch = False,
    ):
        self.patch_size = patch_size
        self.prob_happen = prob_happen
        self.patch_num = patch_num

        self.transpatch = transpatch
        self.prob_rotate = prob_rotate
        self.prob_flip_leftright = prob_flip_leftright

        self.patchpool = deque(maxlen=pool_capacity)

    def __repr__(self):
        if self.transpatch:
            return self.__class__.__name__ + "(patch_size={0}, prob_happen={1}, patch_num={2}, " \
                                             "patch_prob_rotate={3}, patch_prob_flip_leftright={4})"\
            .format(self.patch_size, self.prob_happen, self.patch_num, self.prob_rotate, self.prob_flip_leftright)
        else:
            return self.__class__.__name__ + "(patch_size={0}, prob_happen={1}, patch_num={2})"\
                .format(self.patch_size, self.prob_happen, self.patch_num)

    def generate_wh(self, W, H):
        area = W * H
        for attempt in range(100):
            target_area = random.uniform(
                self.patch_min_area, self.patch_max_area
            ) * area
            aspect_ratio = random.uniform(
                self.patch_min_ratio, 1. / self.patch_min_ratio
            )
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            if w < W and h < H:
                return w, h
        return None, None

    def generate_patchs(self, img):
        self.patchpool.clear()
        w, h = img.size
        l = self.patch_size
        # img = np.array(img)
        for i in range(w // self.patch_size):
            for j in range(h // self.patch_size):
                new_patch = img.crop((l*i, l*j, l*i + l, l*j + l))
                self.patchpool.append(new_patch)


    def transform_patch(self, patch):
        if random.uniform(0, 1) > self.prob_flip_leftright:
            patch = patch.transpose(Image.FLIP_LEFT_RIGHT)
        if random.uniform(0, 1) > self.prob_rotate:
            patch = patch.rotate(random.randint(-10, 10))

        return patch

    def __call__(self, img):
        W, H = img.size # original image size

        if random.uniform(0, 1) > self.prob_happen:
            return img
        # generate this img's patch
        self.generate_patchs(img)
        # paste a randomly selected patch on a random position
        for i in range(self.patch_num):
            patch = random.sample(self.patchpool, 1)[0]
            x1 = random.randint(0, W - self.patch_size)
            y1 = random.randint(0, H - self.patch_size)
            # patch = self.transform_patch(patch)
            img.paste(patch, (x1, y1))
        

        return img


class RandomPatch(object):
    """Random patch data augmentation.

    There is a patch pool that stores randomly extracted pathces from person images.

    For each input image, RandomPatch
        1) extracts a random patch and stores the patch in the patch pool;
        2) randomly selects a patch from the patch pool and pastes it on the
           input (at random position) to simulate occlusion.

    Reference:
        - Zhou et al. Omni-Scale Feature Learning for Person Re-Identification. ICCV, 2019.
        - Zhou et al. Learning Generalisable Omni-Scale Representations
          for Person Re-Identification. TPAMI, 2021.
    """

    def __init__(
            self,
            prob_happen=1,
            pool_capacity=50000,
            min_sample_size=100,
            patch_min_area=0.1,
            patch_max_area=0.5,
            patch_min_ratio=0.1,
            prob_rotate=0.5,
            prob_flip_leftright=0.5,
    ):
        self.prob_happen = prob_happen

        self.patch_min_area = patch_min_area
        self.patch_max_area = patch_max_area
        self.patch_min_ratio = patch_min_ratio

        self.prob_rotate = prob_rotate
        self.prob_flip_leftright = prob_flip_leftright

        self.patchpool = deque(maxlen=pool_capacity)
        self.min_sample_size = min_sample_size

    def generate_wh(self, W, H):
        area = W * H
        for attempt in range(100):
            target_area = random.uniform(
                self.patch_min_area, self.patch_max_area
            ) * area
            aspect_ratio = random.uniform(
                self.patch_min_ratio, 1. / self.patch_min_ratio
            )
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            if w < W and h < H:
                return w, h
        return None, None

    def transform_patch(self, patch):
        if random.uniform(0, 1) > self.prob_flip_leftright:
            patch = patch.transpose(Image.FLIP_LEFT_RIGHT)
        if random.uniform(0, 1) > self.prob_rotate:
            patch = patch.rotate(random.randint(-10, 10))
        return patch

    def __call__(self, img):
        W, H = img.size  # original image size

        # collect new patch
        w, h = self.generate_wh(W, H)
        if w is not None and h is not None:
            x1 = random.randint(0, W - w)
            y1 = random.randint(0, H - h)
            new_patch = img.crop((x1, y1, x1 + w, y1 + h))
            self.patchpool.append(new_patch)

        if len(self.patchpool) < self.min_sample_size:
            return img

        if random.uniform(0, 1) > self.prob_happen:
            return img
        # paste a randomly selected patch on a random position
        patch = random.sample(self.patchpool, 1)[0]
        patchW, patchH = patch.size
        x1 = random.randint(0, W - patchW)
        y1 = random.randint(0, H - patchH)
        patch = self.transform_patch(patch)
        img.paste(patch, (x1, y1))

        return img

def make_dataloader(cfg, transforms=[], usemyaug=False):
    transform_tr = []
    transform_tr += [T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3)]

    if 'random_flip' in transforms:
        print('+ random flip')
        transform_tr += [T.RandomHorizontalFlip(p=cfg.INPUT.PROB)]

    if 'body_blur' in transforms:
        print('+ body blur')
        transform_tr += [
            BodyBlur(probability=0.5, bodyparts=['body'], blurparam=5, blendparam=1, randpart=True, partnum=1)]

    if 'pad' in transforms:
        print('+ pad')
        transform_tr += [T.Pad(cfg.INPUT.PADDING)]

    if 'random_crop' in transforms:
        print('+ random crop')
        transform_tr += [T.RandomCrop(cfg.INPUT.SIZE_TRAIN)]

    transform_tr += [T.ToTensor()]

    if 'random_erase' in transforms:
        print('+ random erase')
        transform_tr += [RandomErasing(probability=cfg.INPUT.RE_PROB, mode='pixel', max_count=1, device='cpu')]

    transform_tr += [T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)]

    transform_tr = T.Compose(transform_tr)
    # print(transform_tr.transforms)
    train_transforms = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
            T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
            T.Pad(cfg.INPUT.PADDING),
            T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
            T.ToTensor(),
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
            RandomErasing(probability=cfg.INPUT.RE_PROB, mode='pixel', max_count=1, device='cpu'),



            # BodyBlur(probability=0.5, bodyparts=['body'], blurparam=5, blendparam=1, randpart=False, partnum=1),
            # RandomCutBlur(probability=0.5, max_count=1, blurparam=5, blendparam=0.5),
            # RandomBlur(probability=0.5, max_count=1, blurparam=3),
            # RandomPatches(probability=0.5, blendparam=0.5),
            # myerase(probability=cfg.INPUT.RE_PROB, mode='pixel', max_count=1, device='cpu',),
            # RandomErasing(probability=cfg.INPUT.RE_PROB, mean=cfg.INPUT.PIXEL_MEAN)
        ])
    #
    # train_transforms = transform_tr
    if usemyaug:
        train_transforms = transform_tr

    print(train_transforms.transforms)
    # print("RandomErasing(probability=cfg.INPUT.RE_PROB, mode='pixel', max_count=1, device='cpu', max_area=0.5)")

    val_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TEST),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    ])

    num_workers = cfg.DATALOADER.NUM_WORKERS

    dataset = __factory[cfg.DATASETS.NAMES](root=cfg.DATASETS.ROOT_DIR)

    train_set = ImageDataset(dataset.train, train_transforms, train=True)
    train_set_normal = ImageDataset(dataset.train, val_transforms, train=True)
    num_classes = dataset.num_train_pids
    cam_num = dataset.num_train_cams
    view_num = dataset.num_train_vids

    if 'triplet' in cfg.DATALOADER.SAMPLER:
        if cfg.MODEL.DIST_TRAIN:
            print('DIST_TRAIN START')
            mini_batch_size = cfg.SOLVER.IMS_PER_BATCH // dist.get_world_size()
            data_sampler = RandomIdentitySampler_DDP(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE)
            batch_sampler = torch.utils.data.sampler.BatchSampler(data_sampler, mini_batch_size, True)
            train_loader = torch.utils.data.DataLoader(
                train_set,
                num_workers=num_workers,
                batch_sampler=batch_sampler,
                collate_fn=train_collate_fn,
                pin_memory=True,
            )
        else:
            train_loader = DataLoader(
                train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH,
                sampler=RandomIdentitySampler(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE),
                num_workers=num_workers, collate_fn=train_collate_fn
            )
    elif cfg.DATALOADER.SAMPLER == 'softmax':
        print('using softmax sampler')
        train_loader = DataLoader(
            train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH, shuffle=True, num_workers=num_workers,
            collate_fn=train_collate_fn
        )
    else:
        print('unsupported sampler! expected softmax or triplet but got {}'.format(cfg.SAMPLER))

    val_set = ImageDataset(dataset.query + dataset.gallery, val_transforms, train=False)

    val_loader = DataLoader(
        val_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn
    )
    train_loader_normal = DataLoader(
        train_set_normal, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn
    )
    return train_loader, train_loader_normal, val_loader, len(dataset.query), num_classes, cam_num, view_num, train_transforms
