"""
Main functions of real IMDB includes:
_load_gt_roidb
_evaluate_detections

General functions:
property: name, classes, num_classes, roidb, num_images
append_flipped_images
evaluate_detections

roidb is a list of roi_rec
roi_rec is a dict of keys ["index", "image", "height", "width", "boxes", "gt_classes", "flipped"]
"""

from symnet.logger import logger
import os
import pickle


class IMDB(object):
    classes = []

    def __init__(self, name, root_path):
        """
        图像数据库的基本信息
        :param root_path: 存储缓存和提议数据
        """
        self._name = name
        self._root_path = root_path

        # abstract attributes
        self._classes = []
        self._roidb = []

        # 新建cache文件
        cache_folder = os.path.join(self._root_path, 'cache')
        if not os.path.exists(cache_folder):
            os.mkdir(cache_folder)

    @property
    def name(self):
        return self._name

    @property
    def num_classes(self):
        return len(self.classes)

    @property
    def roidb(self):
        return self._roidb

    @property
    def num_images(self):
        return len(self._roidb)

    def filter_roidb(self):
        """Remove images without usable rois"""
        num_roidb = len(self._roidb)
        self._roidb = [roi_rec for roi_rec in self._roidb if len(roi_rec['gt_classes'])]
        num_after = len(self._roidb)
        logger.info('filter roidb: {} -> {}'.format(num_roidb, num_after))

    def append_flipped_images(self):
        """仅仅翻转边界框的坐标x1,x2,图像加载到网络时将翻转"""
        logger.info('%s append flipped images to roidb' % self._name)
        roidb_flipped = []
        for roi_rec in self._roidb:
            boxes = roi_rec['boxes'].copy()
            oldx1 = boxes[:, 0].copy()
            oldx2 = boxes[:, 2].copy()
            boxes[:, 0] = roi_rec['width'] - oldx2 - 1
            boxes[:, 2] = roi_rec['width'] - oldx1 - 1
            assert (boxes[:, 2] >= boxes[:, 0]).all()
            roi_rec_flipped = roi_rec.copy()
            roi_rec_flipped['boxes'] = boxes
            roi_rec_flipped['flipped'] = True
            roidb_flipped.append(roi_rec_flipped)
        self._roidb.extend(roidb_flipped)

    def evaluate_detections(self, detections, **kwargs):
        cache_path = os.path.join(self._root_path, 'cache', '{}_{}.pkl'.format(self._name, 'detections'))
        logger.info('saving cache {}'.format(cache_path))
        with open(cache_path, 'wb') as fid:
            pickle.dump(detections, fid, pickle.HIGHEST_PROTOCOL)
        self._evaluate_detections(detections, **kwargs)

    def _get_cached(self, cache_item, fn):
        cache_path = os.path.join(self._root_path, 'cache', '{}_{}.pkl'.format(self._name, cache_item))
        if os.path.exists(cache_path):
            logger.info('loading cache {}'.format(cache_path))
            with open(cache_path, 'rb') as fid:
                cached = pickle.load(fid)
            return cached
        else:
            logger.info('computing cache {}'.format(cache_path))
            cached = fn()
            logger.info('saving cache {}'.format(cache_path))
            with open(cache_path, 'wb') as fid:
                pickle.dump(cached, fid, pickle.HIGHEST_PROTOCOL)
            return cached

    def _load_gt_roidb(self):
        raise NotImplementedError

    def _evaluate_detections(self, detections, **kwargs):
        raise NotImplementedError
