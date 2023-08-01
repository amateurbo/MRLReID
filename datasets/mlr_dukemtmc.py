import math
import os.path as osp
import os
import glob

import cv2
import tqdm

from .bases import BaseImageDataset


class MLR_DukeMTMC(BaseImageDataset):

    dataset_dir = 'MLR_DukeMTMC'
    def __init__(self, root='', verbose=True, pid_begin = 0, **kwargs):
        super(MLR_DukeMTMC, self).__init__()
        # self.dataset_dir = osp.join(root, self.dataset_dir)
        self.dataset_dir = root
        self.train_dir = osp.join(self.dataset_dir, 'bounding_box_train')
        # self.train_dir = osp.join(self.dataset_dir, 'bounding_box_train_mlr_hr')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'bounding_box_test')

        # add bodyblur_randpart_1
        print('--------------------------------------------------------')
        # print('==========> add offline data augment bodyblur_randpart_1')
        # bodyblur_randpart_1 = osp.join(self.dataset_dir, 'bodyblur_randpart_1')
        # add bodyblur_randpart_1
        # print('==========> add offline data augment bodyblur_randpart_2')
        # bodyblur_randpart_2 = osp.join(self.dataset_dir, 'bodyblur_randpart_2')

        self._check_before_run()
        self.pid_begin = pid_begin

        # train = self._process_dir_train([self.train_dir, bodyblur_randpart_2], relabel=True)

        train = self._process_dir(self.train_dir, relabel=True)
        query = self._process_dir(self.query_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False)

        if verbose:
            print("=> MLR_DukeMTMC loaded")
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
                pid = int(imgname.split('_')[0]) - 1
                pid_con.add(pid)
            pidlabel = {pid: label for label, pid in enumerate(pid_con)}

            list = []
            for imgname in tqdm.tqdm(imgnames):
                imgpath = osp.join(dir_path, imgname)
                pid = int(imgname.split('_')[0]) - 1
                camid = int(imgname.split('_')[1].split('c')[1]) - 1

                img = cv2.imread(imgpath)
                h, w = img.shape[0:2]
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
                list.append((imgpath, self.pid_begin + pid, camid, 1, mos, 1))

            return list

    def _process_dir_train(self, dir_paths, relabel=False):
        imgnames = os.listdir(dir_paths[0])
        pid_con = set()
        for imgname in imgnames:
            pid = int(imgname.split('_')[0]) - 1
            pid_con.add(pid)
        pidlabel = {pid: label for label, pid in enumerate(pid_con)}

        list = []
        for dir_path in dir_paths:
            imgnames = os.listdir(dir_path)
            for imgname in imgnames:
                imgpath = osp.join(dir_path, imgname)
                pid = int(imgname.split('_')[0]) - 1
                camid = int(imgname.split('_')[1].split('c')[1]) - 1
                # if((dir_path.split('/')[-1] == 'query')):
                #     camid = camid - 1
                # assert 0 <= camid <= 1
                if relabel:
                    pid = pidlabel[pid]
                list.append((imgpath, self.pid_begin + pid, camid, 1))

        return list