import concurrent.futures
import math
import multiprocessing
import os.path as osp
import os
import glob

import cv2
import tqdm
from PIL import Image

from .bases import BaseImageDataset


def process_image(filepath):
    # img = cv2.imread(filepath)
    # h, w, _ = img.shape

    img = Image.open(filepath)
    w, h = img.size[0], img.size[1]
    return h, w

class MTA_reid(BaseImageDataset):
    dataset_dir = 'MTA_reid'

    def __init__(self, root='', verbose=True, pid_begin = 0, **kwargs):
        super(MTA_reid, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'test')

        self._check_before_run()
        self.pid_begin = pid_begin
        train = self._process_dir(self.train_dir, relabel=True)
        query = self._process_dir(self.query_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False)

        if verbose:
            print("=> MTA_reid loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        # self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(self.train)
        # self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(self.query)
        # self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(self.gallery)

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids, self.imgs_h, self.imgs_w = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids, self.imgs_h, self.imgs_w = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids, self.imgs_h, self.imgs_w = self.get_imagedata_info(self.gallery)


    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))



    def _process_dir(self, dir_path, relabel=False):



        imgnames = os.listdir(dir_path)
        pid_con = set()
        for imgname in imgnames:
            pid = int(imgname.split('_')[5].split('.')[0]) - 1
            pid_con.add(pid)
        pidlabel = {pid: label for label, pid in enumerate(pid_con)}

        list = []
        # for imgname in imgnames:
        with concurrent.futures.ProcessPoolExecutor() as executor:

            for imgname in tqdm.tqdm(imgnames):
                imgpath = osp.join(dir_path, imgname)
                pid = int(imgname.split('_')[5].split('.')[0]) - 1
                camid = int(imgname.split('_')[3]) - 1

                # img = cv2.imread(imgpath)
                # h, w = img.shape[0:2]
                # img = Image.open(imgpath)
                # w, h = img.size[0], img.size[1]
                future = executor.submit(process_image, imgpath)
                h, w = future.result()
                mos = math.sqrt(h * w)
                # if mos > 180:
                #     mos = 1
                # elif mos > 90:
                #     mos = 2
                # elif mos > 45:
                #     mos = 3
                # elif mos > 22.5:
                #     mos = 4
                # else:
                #     mos = 5


                # if((dir_path.split('/')[-1] == 'query')):
                #     camid = camid - 1
                # assert 0 <= camid <= 1
                if relabel:
                    pid = pidlabel[pid]
                # list.append((imgpath, self.pid_begin + pid, camid, 1))
                # list.append((imgpath, self.pid_begin + pid, camid, 1, mos - 1, 1))
                list.append((imgpath, self.pid_begin + pid, camid, 1, mos, 1))

        return list