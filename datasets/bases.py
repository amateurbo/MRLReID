import argparse
import math
from collections import deque

import cv2
import torchvision.transforms as T
from PIL import Image, ImageFile


from torch.utils.data import Dataset
import os.path as osp
import random
import torch
from torchvision import transforms

import utils.augmentations as augmentations
import numpy as np
from timm.data.random_erasing import RandomErasing

ImageFile.LOAD_TRUNCATED_IMAGES = True


def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


class BaseDataset(object):
    """
    Base class of reid dataset
    """

    def get_imagedata_info(self, data):
        pids, cams, tracks = [], [], []
        imghs, imgws = [], []


        for _, pid, camid, trackid, imgh, imgw in data:
            pids += [pid]
            cams += [camid]
            tracks += [trackid]
            imghs += [imgh]
            imgws += [imgw]

        pids = set(pids)
        cams = set(cams)
        tracks = set(tracks)
        imghs = set(imghs)
        imgws = set(imgws)

        num_pids = len(pids)
        num_cams = len(cams)
        num_imgs = len(data)
        num_views = len(tracks)
        num_imghs = len(imghs)
        num_imgws = len(imgws)

        return num_pids, num_imgs, num_cams, num_views, num_imghs, num_imgws

    def print_dataset_statistics(self):
        raise NotImplementedError


class BaseImageDataset(BaseDataset):
    """
    Base class of image reid dataset
    """

    def print_dataset_statistics(self, train, query, gallery):
        num_train_pids, num_train_imgs, num_train_cams, num_train_views, num_train_imghs, num_train_imgws = self.get_imagedata_info(train)
        num_query_pids, num_query_imgs, num_query_cams, num_train_views, num_query_imghs, num_query_imgws = self.get_imagedata_info(query)
        num_gallery_pids, num_gallery_imgs, num_gallery_cams, num_train_views, num_gallery_imghs, num_gallery_imgws = self.get_imagedata_info(gallery)

        print("Dataset statistics:")
        print("  ----------------------------------------")
        print("  subset   | # ids | # images | # cameras")
        print("  ----------------------------------------")
        print("  train    | {:5d} | {:8d} | {:9d}".format(num_train_pids, num_train_imgs, num_train_cams))
        print("  query    | {:5d} | {:8d} | {:9d}".format(num_query_pids, num_query_imgs, num_query_cams))
        print("  gallery  | {:5d} | {:8d} | {:9d}".format(num_gallery_pids, num_gallery_imgs, num_gallery_cams))
        print("  ----------------------------------------")

class IntraClassCutmix(object):
    """IntraClassCutmix.

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
            prob_happen=0.5,
            pool_capacity=5000,
            min_sample_size=4,
            patch_min_area=0.1,
            patch_max_area=0.5,
            patch_min_ratio=0.2,
            prob_rotate=0.5,
            prob_flip_leftright=0.5,
    ):
        self.prob_happen = prob_happen

        self.patch_min_area = patch_min_area
        self.patch_max_area = patch_max_area
        self.patch_min_ratio = patch_min_ratio

        self.prob_rotate = prob_rotate
        self.prob_flip_leftright = prob_flip_leftright

        self.patchpools = {}
        self.pool_capacity = pool_capacity
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

    def __call__(self, img, pid):
        if pid not in self.patchpools.keys():
            self.patchpools[pid] = deque(maxlen=self.pool_capacity)
        self.patchpool = self.patchpools[pid]

        W, H = img.size  # original image size

        # collect new patch
        w, h = self.generate_wh(W, H)
        if w is not None and h is not None:
            x1 = random.randint(0, W - w)
            y1 = random.randint(0, H - h)
            new_patch = img.crop((x1, y1, x1 + w, y1 + h))
            self.patchpool.append(new_patch)

        # print(len(self.patchpool))

        if len(self.patchpool) < self.min_sample_size:
            return img

        if random.uniform(0, 1) > self.prob_happen:
            return img
        # paste a randomly selected patch on a random position
        a = random.sample(self.patchpool, 1)
        patch = random.sample(self.patchpool, 1)[0]
        patchW, patchH = patch.size
        x1 = random.randint(0, W - patchW)
        y1 = random.randint(0, H - patchH)
        patch = self.transform_patch(patch)

        img.paste(patch, (x1, y1))

        return img

#----------------------------------------------------------------
#----------------------------augmix------------------------------
#----------------------------------------------------------------
augmix_parser = argparse.ArgumentParser(
    description='Trains a CIFAR Classifier',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# AugMix options
augmix_parser.add_argument(
    '--mixture-width',
    default=3,
    type=int,
    help='Number of augmentation chains to mix per augmented example')
augmix_parser.add_argument(
    '--mixture-depth',
    default=-1,
    type=int,
    help='Depth of augmentation chains. -1 denotes stochastic depth in [1, 3]')
augmix_parser.add_argument(
    '--aug-severity',
    default=3,
    type=int,
    help='Severity of base augmentation operators')
augmix_parser.add_argument(
    '--no-jsd',
    '-nj',
    default=True,
    type=bool,
    # action='store_true',
    help='Turn off JSD consistency loss.')
augmix_parser.add_argument(
    '--all-ops',
    '-all',
    default=False,
    type=bool,
    # action='store_true',
    help='Turn on all operations (+brightness,contrast,color,sharpness).')
aug_args = augmix_parser.parse_args()

def aug(image, preprocess):
  """Perform AugMix augmentations and compute mixture.

  Args:
    image: PIL.Image input image
    preprocess: Preprocessing function which should return a torch tensor.

  Returns:
    mixed: Augmented and mixed image.
  """
  aug_list = augmentations.augmentations
  if aug_args.all_ops:
    aug_list = augmentations.augmentations_all

  ws = np.float32(np.random.dirichlet([1] * aug_args.mixture_width))
  m = np.float32(np.random.beta(1, 1))

  mix = torch.zeros_like(preprocess(image))
  for i in range(aug_args.mixture_width):
    image_aug = image.copy()
    depth = aug_args.mixture_depth if aug_args.mixture_depth > 0 else np.random.randint(
        1, 4)
    for _ in range(depth):
      op = np.random.choice(aug_list)
      image_aug = op(image_aug, aug_args.aug_severity)
    # Preprocessing commutes since all coefficients are convex
    mix += ws[i] * preprocess(image_aug)

  mixed = (1 - m) * preprocess(image) + m * mix
  return mixed



class ImageDataset(Dataset):
    def __init__(self, dataset, transform=None, train=False):
        self.dataset = dataset
        self.transform = transform
        self.prob_happen = 0.5
        self.train = train

        self.no_jsd = True

        self.preprocess = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize([0.5] * 3, [0.5] * 3)])

        self.erase = transforms.Compose([
            RandomErasing(probability=0.5, mode='pixel', max_count=1, device='cpu'),
        ])
        # self.Cutmix = IntraClassCutmix()
        # print('IntraClassCutmix')
        # self.pre_transforms = T.Compose([
        #     T.Resize((256, 128)),
        # ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        # img_path, pid, camid, trackid = self.dataset[index]
        img_path, pid, camid, trackid, imgh, imgw = self.dataset[index]
        img = read_image(img_path)

        if self.transform is not None:
            img = self.transform(img)
        return img, pid, camid, trackid, img_path.split('/')[-1], imgh, imgw

        # # ----------------Augmix---------------
        # if self.train:
        #
        #     if (random.uniform(0, 1) > self.prob_happen):
        #         img = aug(img, self.preprocess)
        #         img = self.erase(img)
        #     else:
        #         img = self.erase(self.preprocess(img))
        #     return img, pid, camid, trackid, img_path.split('/')[-1]
        #
        #
        # # else:
        # #     im_tuple = (self.preprocess(x), aug(x, self.preprocess),
        # #                 aug(x, self.preprocess))
        # # ----------------Augmix---------------
        #
        # #------------IntraClassCutmix----------
        # # flag = img_path.find('train')
        # #
        # # if self.pre_transforms is not None:
        # #     img = self.pre_transforms(img)
        # #80
        # # if flag:
        # #     self.Cutmix.__call__(img, pid)
        # # ------------IntraClassCutmix----------
        #
        #
        # # ----------------orignal---------------
        #
        #
        # #
        # else:
        #     img = self.preprocess(img)
        #     return img, pid, camid, trackid,img_path.split('/')[-1]