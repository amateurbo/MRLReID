import argparse

import torch
from PIL import Image
import numpy
import sys

from numba.cuda import jit
from torchvision import transforms
import numpy as np
import cv2


def rollout(attentions, discard_ratio, head_fusion):
    result = torch.eye(attentions[0].size(-1))
    attentions1 = attentions[0:12]
    with torch.no_grad():
        for attention in attentions1:
            # attention.resize_(1, 12, 211, 211)
            if head_fusion == "mean":
                attention_heads_fused = attention.mean(axis=1)
            elif head_fusion == "max":
                attention_heads_fused = attention.max(axis=1)[0]
            elif head_fusion == "min":
                attention_heads_fused = attention.min(axis=1)[0]
            else:
                raise "Attention head fusion type Not supported"

            # Drop the lowest attentions, but
            # don't drop the class token
            flat = attention_heads_fused.view(attention_heads_fused.size(0), -1)
            _, indices = flat.topk(int(flat.size(-1) * discard_ratio), -1, False)
            indices = indices[indices != 0]
            flat[0, indices] = 0

            I = torch.eye(attention_heads_fused.size(-1))
            a = (attention_heads_fused + 1.0 * I) / 2
            a = a / a.sum(dim=-1)


            result = torch.matmul(a, result)

    # mask = result[]

    # Look at the total attention between the class token,
    # and the image patches
    mask = result[0, 0, 1:]
    # In case of 224x224 image, this brings us from 196 to 14
    width = int(mask.size(-1) ** 0.5)

    # mask = mask.resize_(196)
    # return mask.numpy()

    mask = mask.reshape(21, 10).numpy()
    mask = mask / np.max(mask)
    return cv2.resize(mask, (256, 128))


def rollouts(attentions, discard_ratio, head_fusion):

    # print(attentions.size())
    results = torch.ones(attentions[0].size()).cuda()
    for i in range(attentions[0].size(0)):
        results[i] = torch.eye(attentions[0].size(-1))

    # result = torch.eye(attentions[0].size(-1))
    attentions1 = attentions[0:12]
    with torch.no_grad():
        # for attention in attentions1:
        for i in range(12):
            # attention =
            # attention.resize_(1, 12, 211, 211)
            if head_fusion == "mean":
                attention_heads_fused = attentions[i].mean(axis=1)
            elif head_fusion == "max":
                attention_heads_fused = attentions[i].max(axis=1)[0]
            elif head_fusion == "min":
                attention_heads_fused = attentions[i].min(axis=1)[0]
            else:
                raise "Attention head fusion type Not supported"

            # Drop the lowest attentions, but
            # don't drop the class token
            flat = attention_heads_fused.view(attention_heads_fused.size(0), -1)
            _, indices = flat.topk(int(flat.size(-1) * discard_ratio), -1, False)

            for i in range(indices.size(0)):
                flat[i, indices[i]] = 0
                I = torch.eye(attention_heads_fused[i].size(-1)).cuda()
                a = (attention_heads_fused[i] + 1.0 * I) / 2
                a = a / a.sum(dim=-1)
                results[i] = torch.matmul(a, results[i])

        mask = results[:, 0, 0, 1:]
        mask = mask.reshape(-1, 21, 10)
        for i in range(attentions[0].size(0)):
            mask[i] = mask[i] / torch.max(mask[i])

    return mask


class VITAttentionRollout:
    def __init__(self, model, attention_layer_name='attn_drop', head_fusion="mean",
                 discard_ratio=0.9):
        self.model = model
        self.head_fusion = head_fusion
        self.discard_ratio = discard_ratio
        self.attention_layer_name = attention_layer_name
        self.handel = []
        for name, module in self.model.named_modules():
            if attention_layer_name in name:
                self.handel.append(module.register_forward_hook(self.get_attention))

        self.attentions = []

    def get_attention(self, module, input, output):
        self.attentions.append(output)

    def __call__(self, input_tensor, target_cam, target_view):
        self.attentions = []
        with torch.no_grad():
            output = self.model(input_tensor, target_cam, target_view)

        # remove forward_hook to avoid memory out
        for handel in enumerate(self.handel):
            handel[1].remove()

        return rollouts(self.attentions, self.discard_ratio, self.head_fusion)

def show_mask_on_image(img, mask):
    img = np.float32(img) / 255
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_cuda', action='store_true', default=False,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument('--image_path', type=str, default='/mnt/data/datasets/DukeMTMC-reID/query/0005_c2_f0046985.jpg',
                        help='Input image path')
    parser.add_argument('--head_fusion', type=str, default='max',
                        help='How to fuse the attention heads for attention rollout. \
                        Can be mean/max/min')
    parser.add_argument('--discard_ratio', type=float, default=0.9,
                        help='How many of the lowest 14x14 attention paths should we discard')
    parser.add_argument('--category_index', type=int, default=None,
                        help='The category index for gradient rollout')
    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print("Using GPU")
    else:
        print("Using CPU")

    return args

if __name__ == '__main__':
    args = get_args()
    model = torch.hub.load('facebookresearch/deit:main',
        'deit_tiny_patch16_224', pretrained=True)
    model.eval()

    if args.use_cuda:
        model = model.cuda()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    img = Image.open(args.image_path)
    img = img.resize((224, 224))
    input_tensor = transform(img).unsqueeze(0)
    if args.use_cuda:
        input_tensor = input_tensor.cuda()

    if args.category_index is None:
        print("Doing Attention Rollout")
        attention_rollout = VITAttentionRollout(model, head_fusion=args.head_fusion,
            discard_ratio=args.discard_ratio)
        mask = attention_rollout(input_tensor)
        name = "attention_rollout_{:.3f}_{}.png".format(args.discard_ratio, args.head_fusion)
    # else:
    #     print("Doing Gradient Attention Rollout")
    #     grad_rollout = VITAttentionGradRollout(model, discard_ratio=args.discard_ratio)
    #     mask = grad_rollout(input_tensor, args.category_index)
    #     name = "grad_rollout_{}_{:.3f}_{}.png".format(args.category_index,
    #         args.discard_ratio, args.head_fusion)


    np_img = np.array(img)[:, :, ::-1]
    mask = cv2.resize(mask, (np_img.shape[1], np_img.shape[0]))
    print((mask > 0.5).sum())
    mask = show_mask_on_image(np_img, mask)
    cv2.imshow("Input Image", np_img)
    cv2.imshow(name, mask)
    cv2.imwrite("input.png", np_img)
    cv2.imwrite(name, mask)
    cv2.waitKey(-1)