import random
import math

import PIL
import PIL.Image as Image
# import PIL.ImageShow

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt

_pil_interpolation_to_str = {
    Image.NEAREST: 'PIL.Image.NEAREST',
    Image.BILINEAR: 'PIL.Image.BILINEAR',
    Image.BICUBIC: 'PIL.Image.BICUBIC',
    Image.LANCZOS: 'PIL.Image.LANCZOS',
    Image.HAMMING: 'PIL.Image.HAMMING',
    Image.BOX: 'PIL.Image.BOX',
}

def _get_pixels(per_pixel, rand_color, patch_size, dtype=torch.float32, device='cuda'):
    # NOTE I've seen CUDA illegal memory access errors being caused by the normal_()
    # paths, flip the order so normal is run on CPU if this becomes a problem
    # Issue has been fixed in master https://github.com/pytorch/pytorch/issues/19508
    if per_pixel:
        return torch.empty(patch_size, dtype=dtype, device=device).normal_()
    elif rand_color:
        return torch.empty((patch_size[0], 1, 1), dtype=dtype, device=device).normal_()
    else:
        return torch.zeros((patch_size[0], 1, 1), dtype=dtype, device=device)

class RandomErasing:
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf

        This variant of RandomErasing is intended to be applied to either a batch
        or single image tensor after it has been normalized by dataset mean and std.
    Args:
         probability: Probability that the Random Erasing operation will be performed.
         min_area: Minimum percentage of erased area wrt input image area.
         max_area: Maximum percentage of erased area wrt input image area.
         min_aspect: Minimum aspect ratio of erased area.
         mode: pixel color mode, one of 'const', 'rand', or 'pixel'
            'const' - erase block is constant color of 0 for all channels
            'rand'  - erase block is same per-channel random (normal) color
            'pixel' - erase block is per-pixel random (normal) color
        max_count: maximum number of erasing blocks per image, area per box is scaled by count.
            per-image count is randomly chosen between 1 and this value.
    """

    def __init__(
            self,
            probability=0.5, min_area=0.02, max_area=1/3, min_aspect=0.3, max_aspect=None,
            mode='const', min_count=1, max_count=None, num_splits=0, device='cuda', topstart=0, topend=0, area_change=False):
        self.probability = probability
        self.min_area = min_area
        self.max_area = max_area
        max_aspect = max_aspect or 1 / min_aspect
        self.log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))
        self.min_count = min_count
        self.max_count = max_count or min_count
        self.num_splits = num_splits
        mode = mode.lower()
        self.rand_color = False
        self.per_pixel = False
        if mode == 'rand':
            self.rand_color = True  # per block random normal
        elif mode == 'pixel':
            self.per_pixel = True  # per pixel random normal
        else:
            assert not mode or mode == 'const'
        self.device = device

        self.mode = mode
        self.topstart = topstart
        self.topend = topend
        self.area_change = area_change

    def __repr__(self):
        # probability = 0.5, min_area = 0.02, max_area = 1 / 3, min_aspect = 0.3, max_aspect = None,
        # mode = 'const', min_count = 1, max_count = None, num_splits = 0, device = 'cuda', topstart = 0
        return self.__class__.__name__ + "(probability={0}, max_area={1}, mode={2}, topstart={3}, topend={4}, " \
                                         "area_change(base_cutpatch)={5})"\
            .format(self.probability, self.max_area, self.mode, self.topstart, self.topend, self.area_change)

    def _erase(self, img, chan, img_h, img_w, dtype):
        if random.random() > self.probability:
            return
        topstart = (img_h * self.topstart).__int__()
        topend = (img_h * self.topend).__int__()
        if self.area_change:
            area = (img_h - topstart - topend) * img_w
        else:
            area = img_h * img_w

        count = self.min_count if self.min_count == self.max_count else \
            random.randint(self.min_count, self.max_count)

        for _ in range(count):
            for attempt in range(10):
                target_area = random.uniform(self.min_area, self.max_area) * area / count
                aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
                h = int(round(math.sqrt(target_area * aspect_ratio)))
                w = int(round(math.sqrt(target_area / aspect_ratio)))

                if w < img_w and h + topend < img_h - topstart:
                    # top = random.randint(0, img_h - h)
                    top = random.randint(topstart, img_h - h - topend)
                    left = random.randint(0, img_w - w)
                    img[:, top:top + h, left:left + w] = _get_pixels(
                        self.per_pixel, self.rand_color, (chan, h, w),
                        dtype=dtype, device=self.device)
                    break

    def __call__(self, input):
        if len(input.size()) == 3:
            self._erase(input, *input.size(), input.dtype)
        else:
            batch_size, chan, img_h, img_w = input.size()
            # skip first slice of batch if num_splits is set (for clean portion of samples)
            batch_start = batch_size // self.num_splits if self.num_splits > 1 else 0
            for i in range(batch_start, batch_size):
                self._erase(input[i], chan, img_h, img_w, input.dtype)
        return input



class RandomBlur:
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf

        This variant of RandomErasing is intended to be applied to either a batch
        or single image tensor after it has been normalized by dataset mean and std.
    Args:
         probability: Probability that the Random Erasing operation will be performed.
         min_area: Minimum percentage of erased area wrt input image area.
         max_area: Maximum percentage of erased area wrt input image area.
         min_aspect: Minimum aspect ratio of erased area.
         mode: pixel color mode, one of 'const', 'rand', or 'pixel'
            'const' - erase block is constant color of 0 for all channels
            'rand'  - erase block is same per-channel random (normal) color
            'pixel' - erase block is per-pixel random (normal) color
        max_count: maximum number of erasing blocks per image, area per box is scaled by count.
            per-image count is randomly chosen between 1 and this value.
    """

    def __init__(
            self,
            probability=0.5, min_area=0.02, max_area=1/3, min_aspect=0.3, max_aspect=None,
            min_count=1, max_count=None, num_splits=0, blurparam=10, topstart=0, topend=0, area_change=False):
        self.probability = probability
        self.min_area = min_area
        self.max_area = max_area
        max_aspect = max_aspect or 1 / min_aspect
        self.log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))
        self.min_count = min_count
        self.max_count = max_count or min_count
        self.num_splits = num_splits

        self.topstart = topstart
        self.topend = topend
        self.area_change = area_change
        self.blurparam = blurparam

        # resampling filters (also defined in Imaging.h)
        # NEAREST = NONE = 0
        # BOX = 4
        # BILINEAR = LINEAR = 2
        # HAMMING = 5
        # BICUBIC = CUBIC = 3
        # LANCZOS = ANTIALIAS = 1
        self.DOWN = PIL.Image.CUBIC
        self.UP = PIL.Image.NEAREST

    def __repr__(self):
        return self.__class__.__name__ + "(probability={0}, max_area={1}, blurparam={2} topstart={3}, topend={4}, " \
                                         "area_change(base_cutpatch)={5}, DOWN={6}, UP={7})"\
            .format(self.probability, self.max_area, self.blurparam, self.topstart, self.topend, self.area_change,
                    self.DOWN, self.UP)

    def __call__(self, input):

        img_w, img_h = input.size
        if random.random() > self.probability:
            return input

        topstart = (img_h * self.topstart).__int__()
        topend = (img_h * self.topend).__int__()
        if self.area_change:
            area = (img_h - topstart - topend) * img_w
        else:
            area = img_h * img_w

        count = self.min_count if self.min_count == self.max_count else \
            random.randint(self.min_count, self.max_count)

        for _ in range(count):
            for attempt in range(10):
                target_area = random.uniform(self.min_area, self.max_area) * area / count
                aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
                h = int(round(math.sqrt(target_area * aspect_ratio)))
                w = int(round(math.sqrt(target_area / aspect_ratio)))

                if w < img_w and h + topend < img_h - topstart:
                    # top = random.randint(0, img_h - h)
                    top = random.randint(topstart, img_h - h - topend)
                    left = random.randint(0, img_w - w)
                    cut_img = input.crop((left, top, left + w, top + h))

                    down_size = (int(w / self.blurparam), int(h / self.blurparam))
                    cut_img = cut_img.resize(down_size, resample=self.DOWN)

                    up_size = (w, h)
                    patch = cut_img.resize(up_size, resample=self.UP)
                    input.paste(patch, (left, top))

                    break

        return input

class RandomCutBlur:
    def __init__(
            self,
            probability=0.5, min_area=0.02, max_area=1/3, min_aspect=0.3, max_aspect=None,
            min_count=1, max_count=None, num_splits=0, blurparam=10, blendparam=0.5):
        self.probability = probability
        self.min_area = min_area
        self.max_area = max_area
        max_aspect = max_aspect or 1 / min_aspect
        self.log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))
        self.min_count = min_count
        self.max_count = max_count or min_count
        self.num_splits = num_splits

        self.blurparam = blurparam
        self.blendparam = blendparam

        # resampling filters (also defined in Imaging.h)
        # NEAREST = NONE = 0
        # BOX = 4
        # BILINEAR = LINEAR = 2
        # HAMMING = 5
        # BICUBIC = CUBIC = 3
        # LANCZOS = ANTIALIAS = 1
        self.DOWN = PIL.Image.CUBIC
        self.UP = PIL.Image.NEAREST

    def __repr__(self):
        down_interpolate_str = _pil_interpolation_to_str[self.DOWN]
        up_interpolate_str = _pil_interpolation_to_str[self.UP]
        return self.__class__.__name__ + "(probability={0}, max_area={1}, blurparam={2}, blendparam={5}, DOWN={3}, UP={4} )"\
            .format(self.probability, self.max_area, self.blurparam, down_interpolate_str, up_interpolate_str, self.blendparam)

    def __call__(self, input):

        img_w, img_h = input.size
        if random.random() > self.probability:
            return input

        area = img_h * img_w

        count = self.min_count if self.min_count == self.max_count else \
            random.randint(self.min_count, self.max_count)

        for _ in range(count):
            for attempt in range(10):
                target_area = random.uniform(self.min_area, self.max_area) * area / count
                aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
                h = int(round(math.sqrt(target_area * aspect_ratio)))
                w = int(round(math.sqrt(target_area / aspect_ratio)))

                if w < img_w and h < img_h:
                    top = random.randint(0, img_h - h)
                    left = random.randint(0, img_w - w)
                    cut_img = input.crop((left, top, left + w, top + h))

                    #blur image, first downsample, then upsample
                    down_size = (int(w / self.blurparam), int(h / self.blurparam))
                    cut_img_blur = cut_img.resize(down_size, resample=self.DOWN)
                    up_size = (w, h)
                    cut_img_blur = cut_img_blur.resize(up_size, resample=self.UP)

                    #blend(mix) image, cut_img * (1 - blendparam) + cut_img_blur * blendparam
                    cut_img_blend = Image.blend(cut_img, cut_img_blur, self.blendparam)

                    #merge image
                    input.paste(cut_img_blend, (left, top))

                    break

        return input

class BodyBlur:
    def __init__(
            self, probability=0.5, blurparam=10, blendparam=0.5, bodyparts=['head'], randpart=False, partnum=1):
        self.probability = probability

        self.blurparam = blurparam
        self.blendparam = blendparam

        self.bodyparts = bodyparts

        self.randpart = randpart
        self.partnum = partnum

        # resampling filters (also defined in Imaging.h)
        # NEAREST = NONE = 0
        # BOX = 4
        # BILINEAR = LINEAR = 2
        # HAMMING = 5
        # BICUBIC = CUBIC = 3
        # LANCZOS = ANTIALIAS = 1
        self.DOWN = PIL.Image.CUBIC
        self.UP = PIL.Image.NEAREST

    def __repr__(self):
        down_interpolate_str = _pil_interpolation_to_str[self.DOWN]
        up_interpolate_str = _pil_interpolation_to_str[self.UP]
        return self.__class__.__name__ + "(probability={0}, bodyparts={1}, blurparam={2}, blendparam={5}, DOWN={3}, UP={4}, randompart={6}, partnum={7} )"\
            .format(self.probability, self.bodyparts, self.blurparam, down_interpolate_str, up_interpolate_str, self.blendparam, self.randpart, self.partnum)

    def blur(self, input, h0, h1, img_w):

        cut_img = input.crop((0, h0, img_w, h1))
        w = img_w
        h = h1 - h0
        # blur image, first downsample, then upsample
        down_size = (int(w / self.blurparam), int(h / self.blurparam))
        cut_img_blur = cut_img.resize(down_size, resample=self.DOWN)
        up_size = (w, h)
        cut_img_blur = cut_img_blur.resize(up_size, resample=self.UP)

        # blend(mix) image, cut_img * (1 - blendparam) + cut_img_blur * blendparam
        cut_img_blend = Image.blend(cut_img, cut_img_blur, self.blendparam)

        # merge image
        input.paste(cut_img_blend, (0, h0))

    def __call__(self, input):


        img_w, img_h = input.size
        if random.uniform(0, 1) > self.probability:
            return input

        #
        if self.randpart:
            self.bodyparts = random.choices(['head', 'body', 'legs'], k=self.partnum)

        # The point in the upper left corner is the image (0,0) point
        if 'head' in self.bodyparts:
            h0 = int(img_h * 0)
            h1 = int(img_h * 1/5)
            self.blur(input, h0, h1, img_w)

        if 'body' in self.bodyparts:
            h0 = int(img_h * 1/5)
            h1 = int(img_h * 3/5)
            self.blur(input, h0, h1, img_w)

        if 'legs' in self.bodyparts:
            h0 = int(img_h * 3/5)
            h1 = int(img_h * 5/5)
            self.blur(input, h0, h1, img_w)

        # for _ in range(count):
        #     for attempt in range(10):
        #         target_area = random.uniform(self.min_area, self.max_area) * area / count
        #         aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
        #         h = int(round(math.sqrt(target_area * aspect_ratio)))
        #         w = int(round(math.sqrt(target_area / aspect_ratio)))
        #
        #         if w < img_w and h < img_h:
        #             top = random.randint(0, img_h - h)
        #             left = random.randint(0, img_w - w)
        #             cut_img = input.crop((left, top, left + w, top + h))
        #
        #             #blur image, first downsample, then upsample
        #             down_size = (int(w / self.blurparam), int(h / self.blurparam))
        #             cut_img_blur = cut_img.resize(down_size, resample=self.DOWN)
        #             up_size = (w, h)
        #             cut_img_blur = cut_img_blur.resize(up_size, resample=self.UP)
        #
        #             #blend(mix) image, cut_img * (1 - blendparam) + cut_img_blur * blendparam
        #             cut_img_blend = Image.blend(cut_img, cut_img_blur, self.blendparam)
        #
        #             #merge image
        #             input.paste(cut_img_blend, (left, top))
        #
        #             break

        return input

class RandomPatches:
    def __init__(
            self,
            probability=0.5, min_area=0.02, max_area=1/3, min_aspect=0.3, max_aspect=None,
            min_count=1, max_count=None, num_splits=0, blurparam=10, blendparam=0.5):
        self.probability = probability
        self.min_area = min_area
        self.max_area = max_area
        max_aspect = max_aspect or 1 / min_aspect
        self.log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))
        self.min_count = min_count
        self.max_count = max_count or min_count
        self.num_splits = num_splits
        self.blendparam = blendparam

        self.patchsize = 16

    def __repr__(self):
        return self.__class__.__name__ + "(probability={0}, max_area={1}, patch_size={2}, blendparam={3})"\
            .format(self.probability, self.max_area, self.patchsize, self.blendparam)

    def __call__(self, input):

        img_w, img_h = input.size
        if random.random() > self.probability:
            return input
        area = img_h * img_w
        count = self.min_count if self.min_count == self.max_count else \
            random.randint(self.min_count, self.max_count)

        for _ in range(count):
            for attempt in range(10):
                target_area = random.uniform(self.min_area, self.max_area) * area / count
                aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
                h = int(round(math.sqrt(target_area * aspect_ratio)))
                w = int(round(math.sqrt(target_area / aspect_ratio)))

                locations = []
                img_patches = []
                def getpatches(img, left, top):
                    num_h = h // self.patchsize
                    num_w = w // self.patchsize
                    for i in range(num_h):
                        for j in range(num_w):
                            l = j * self.patchsize
                            t = i * self.patchsize
                            locations.append((l, t))
                            patch = img.crop((l, t, l + self.patchsize, t + self.patchsize))
                            # patch.show()
                            img_patches.append(patch)

                def pastepatches(img):
                    for i in range(len(img_patches)):
                        location = locations[i]
                        patch = img_patches[i]
                        img.paste(patch, location)
                        # img.show()
                    return img


                if w < img_w and h < img_h:
                    top = random.randint(0, img_h - h)
                    left = random.randint(0, img_w - w)
                    cut_img = input.crop((left, top, left + w, top + h))

                    getpatches(cut_img, left, top)
                    random.shuffle(img_patches)
                    rand_patchimg = pastepatches(cut_img.copy())
                    cut_img_blend = Image.blend(cut_img, rand_patchimg, self.blendparam)

                    input.paste(cut_img_blend, (left, top))
                    break

        return input