import numpy as np
from symdata.bbox import bbox_overlaps, bbox_transform


class AnchorGenerator:
    '''
    比如800*600的原图,计算之后的特征图缩小了16倍,特征图大小为50*38
    9个基本锚框的移动,输出结果就是50*38*9个锚框,这样的锚框就和特征图产生了一一对应的关系
    '''
    def __init__(self, feat_stride=16, anchor_scales=(8, 16, 32), anchor_ratios=(0.5, 1, 2)):
        self._num_anchors = len(anchor_scales) * len(anchor_ratios)#9，锚框数
        self._feat_stride = feat_stride
        self._base_anchors = self._generate_base_anchors(feat_stride, np.array(anchor_scales), np.array(anchor_ratios))

    def generate(self, feat_height, feat_width):
        # 将特征图的宽高进行feat_stride=16倍扩大至原图
        shift_x = np.arange(0, feat_width) * self._feat_stride
        shift_y = np.arange(0, feat_height) * self._feat_stride
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)#生成原图的网格点
        shifts = np.vstack((shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel())).transpose()
        # 锚框数的转换(注意看形状的变换)
        # (1, A, 4)+(K, 1, 4) ==> (K, A, 4) ==> (K*A, 4)
        A = self._num_anchors # 锚框数
        K = shifts.shape[0] # 比如800*600的缩小16倍是50*38
        all_anchors = self._base_anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2))
        all_anchors = all_anchors.reshape((K * A, 4))
        return all_anchors

    @staticmethod
    def _generate_base_anchors(base_size, scales, ratios):
        """
        通过为一个引用(0,0,15,15)窗口枚举宽高比X缩放来生成锚框(引用)窗口
        中心为(7.5,7.5)的9个锚框平移到全图上,覆盖所有可能的区域
        """
        base_anchor = np.array([1, 1, base_size, base_size]) - 1
        ratio_anchors = AnchorGenerator._ratio_enum(base_anchor, ratios)
        anchors = np.vstack([AnchorGenerator._scale_enum(ratio_anchors[i, :], scales)
                             for i in range(ratio_anchors.shape[0])])
        return anchors

    @staticmethod
    def _whctrs(anchor):
        """
        返回一个锚框的宽度、高度、x中心和y中心
        """
        w = anchor[2] - anchor[0] + 1
        h = anchor[3] - anchor[1] + 1
        x_ctr = anchor[0] + 0.5 * (w - 1)
        y_ctr = anchor[1] + 0.5 * (h - 1)
        return w, h, x_ctr, y_ctr

    @staticmethod
    def _mkanchors(ws, hs, x_ctr, y_ctr):
        """
        给定一个围绕中心(x_ctr, y_ctr)的宽度(ws)和高度(hs)向量，输出一组锚点(窗口)
        """
        ws = ws[:, np.newaxis]
        hs = hs[:, np.newaxis]
        anchors = np.hstack((x_ctr - 0.5 * (ws - 1),
                             y_ctr - 0.5 * (hs - 1),
                             x_ctr + 0.5 * (ws - 1),
                             y_ctr + 0.5 * (hs - 1)))
        return anchors

    @staticmethod
    def _ratio_enum(anchor, ratios):
        """
        为每个锚框的纵横比，输出一组锚框
        """
        w, h, x_ctr, y_ctr = AnchorGenerator._whctrs(anchor)
        size = w * h
        size_ratios = size / ratios
        ws = np.round(np.sqrt(size_ratios))
        hs = np.round(ws * ratios)
        anchors = AnchorGenerator._mkanchors(ws, hs, x_ctr, y_ctr)
        return anchors

    @staticmethod
    def _scale_enum(anchor, scales):
        """
        为每个锚框的尺度，输出一组锚框
        """
        w, h, x_ctr, y_ctr = AnchorGenerator._whctrs(anchor)
        ws = w * scales
        hs = h * scales
        anchors = AnchorGenerator._mkanchors(ws, hs, x_ctr, y_ctr)
        return anchors


class AnchorSampler:
    def __init__(self, allowed_border=0, batch_rois=256, fg_fraction=0.5, fg_overlap=0.7, bg_overlap=0.3):
        self._allowed_border = allowed_border
        self._num_batch = batch_rois
        self._num_fg = int(batch_rois * fg_fraction)
        self._fg_overlap = fg_overlap
        self._bg_overlap = bg_overlap

    def assign(self, anchors, gt_boxes, im_height, im_width):
        num_anchors = anchors.shape[0]

        # 过滤掉无效的标签
        valid_labels = np.where(gt_boxes[:, -1] > 0)[0]
        gt_boxes = gt_boxes[valid_labels]

        # 过滤掉区域外的锚框
        inds_inside = np.where((anchors[:, 0] >= -self._allowed_border) &
                               (anchors[:, 2] < im_width + self._allowed_border) &
                               (anchors[:, 1] >= -self._allowed_border) &
                               (anchors[:, 3] < im_height + self._allowed_border))[0]
        anchors = anchors[inds_inside, :]
        num_valid = len(inds_inside)

        # 标签值: 1为正类, 0为负类, -1忽略
        labels = np.ones((num_valid,), dtype=np.float32) * -1
        bbox_targets = np.zeros((num_valid, 4), dtype=np.float32)
        bbox_weights = np.zeros((num_valid, 4), dtype=np.float32)

        if gt_boxes.size > 0:
            # 锚框与真实框的重叠
            overlaps = bbox_overlaps(anchors.astype(np.float), gt_boxes.astype(np.float))

            # 前景锚框:每个gt的最高重叠的锚框
            gt_max_overlaps = overlaps.max(axis=0)
            argmax_inds = np.where(overlaps == gt_max_overlaps)[0]
            labels[argmax_inds] = 1

            # 前景锚框：大于交并比阈值的锚框
            max_overlaps = overlaps.max(axis=1)
            labels[max_overlaps >= self._fg_overlap] = 1

            # 背景锚框: 小于交并比阈值的锚框
            labels[max_overlaps < self._bg_overlap] = 0

            # sanity check
            fg_inds = np.where(labels == 1)[0]
            bg_inds = np.where(labels == 0)[0]
            #返回np.intersect1d相同的元素值(一维数组)
            assert len(np.intersect1d(fg_inds, bg_inds)) == 0

            # 子样本的正类锚框
            cur_fg = len(fg_inds)
            if cur_fg > self._num_fg:
                disable_inds = np.random.choice(fg_inds, size=(cur_fg - self._num_fg), replace=False)
                labels[disable_inds] = -1

            # 子样本的负类锚框
            cur_bg = len(bg_inds)
            max_neg = self._num_batch - min(self._num_fg, cur_fg)
            if cur_bg > max_neg:
                disable_inds = np.random.choice(bg_inds, size=(cur_bg - max_neg), replace=False)
                labels[disable_inds] = -1

            # 每行重叠的最大索引值，那就是找出最接近真实框gt_boxes的值了
            fg_inds = np.where(labels == 1)[0]
            argmax_overlaps = overlaps.argmax(axis=1)
            bbox_targets[fg_inds, :] = bbox_transform(anchors[fg_inds, :], gt_boxes[argmax_overlaps[fg_inds], :],
                                                      box_stds=(1.0, 1.0, 1.0, 1.0))

            # 只有前景锚框才有bbox_targets
            bbox_weights[fg_inds, :] = 1
        else:
            # 随机画背景锚框
            bg_inds = np.random.choice(np.arange(num_valid), size=self._num_batch, replace=False)
            labels[bg_inds] = 0

        all_labels = np.ones((num_anchors,), dtype=np.float32) * -1
        all_labels[inds_inside] = labels
        all_bbox_targets = np.zeros((num_anchors, 4), dtype=np.float32)
        all_bbox_targets[inds_inside, :] = bbox_targets
        all_bbox_weights = np.zeros((num_anchors, 4), dtype=np.float32)
        all_bbox_weights[inds_inside, :] = bbox_weights

        return all_labels, all_bbox_targets, all_bbox_weights
