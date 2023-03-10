"""
Proposal Target Operator selects foreground and background roi and assigns label, bbox_transform to them.
"""

import mxnet as mx
import numpy as np

from symdata.bbox import bbox_overlaps, bbox_transform


def sample_rois(rois, gt_boxes, num_classes, rois_per_image, fg_rois_per_image, fg_overlap, box_stds):
    """
    由前景和背景示例组成的ROIs中随机抽取样本
    前景:背景=1:3左右,就是正负样本的比例
    :param rois: [n, 5] (batch_index, x1, y1, x2, y2)
    :param gt_boxes: [n, 5] (x1, y1, x2, y2, cls)
    :param num_classes: 类别数
    :param rois_per_image: 每张图的总roi数量int(batch_rois/batch_images)这里的batch_images为1,所以默认是128
    :param fg_rois_per_image: 每张图的前景roi数量
    :param fg_overlap: 前景roi的重叠阈值
    :param box_stds: 边界框标准差
    :return: (labels, rois, bbox_targets, bbox_weights)
    """
    overlaps = bbox_overlaps(rois[:, 1:], gt_boxes[:, :4]) #roi与真实框的IoU
    gt_assignment = overlaps.argmax(axis=1) # 每个roi与真实框的最大IoU
    labels = gt_boxes[gt_assignment, 4] # 每个roi对应的最大交并比的标签
    max_overlaps = overlaps.max(axis=1) # 最大交并比

    # 大于等于前景重叠阈值就选为前景[FG_THRESH默认0.5]
    fg_indexes = np.where(max_overlaps >= fg_overlap)[0]
    # 避免超过每张图的前景roi数量
    # 其中replace=False确保抽取的不重复
    fg_rois_this_image = min(fg_rois_per_image, len(fg_indexes))
    if len(fg_indexes) > fg_rois_this_image:
        fg_indexes = np.random.choice(fg_indexes, size=fg_rois_this_image, replace=False)

    # 小于前景重叠阈值就选为背景 [0, FG_THRESH)
    bg_indexes = np.where(max_overlaps < fg_overlap)[0]
    # 背景roi数量为总roi数量减去前景roi数量
    bg_rois_this_image = rois_per_image - fg_rois_this_image
    bg_rois_this_image = min(bg_rois_this_image, len(bg_indexes))
    if len(bg_indexes) > bg_rois_this_image:
        bg_indexes = np.random.choice(bg_indexes, size=bg_rois_this_image, replace=False)

    # 前景roi与背景roi两者的index的结合
    keep_indexes = np.append(fg_indexes, bg_indexes)
    # 如果数量小于总的rois,就填充更多背景rois以确保固定的小批量大小
    while len(keep_indexes) < rois_per_image:
        gap = min(len(bg_indexes), rois_per_image - len(keep_indexes))
        gap_indexes = np.random.choice(range(len(bg_indexes)), size=gap, replace=False)
        keep_indexes = np.append(keep_indexes, bg_indexes[gap_indexes])

    rois = rois[keep_indexes]
    labels = labels[keep_indexes]
    # 前景之后的背景rois的标签值设定为0
    labels[fg_rois_this_image:] = 0

    # 加载并计算bbox_target
    targets = bbox_transform(rois[:, 1:], gt_boxes[gt_assignment[keep_indexes], :4], box_stds=box_stds)
    bbox_targets = np.zeros((rois_per_image, 4 * num_classes), dtype=np.float32)
    bbox_weights = np.zeros((rois_per_image, 4 * num_classes), dtype=np.float32)
    for i in range(fg_rois_this_image):
        cls_ind = int(labels[i])
        bbox_targets[i, cls_ind * 4:(cls_ind + 1) * 4] = targets[i]
        bbox_weights[i, cls_ind * 4:(cls_ind + 1) * 4] = 1

    return rois, labels, bbox_targets, bbox_weights


class ProposalTargetOperator(mx.operator.CustomOp):
    def __init__(self, num_classes, batch_images, batch_rois, fg_fraction, fg_overlap, box_stds):
        super(ProposalTargetOperator, self).__init__()
        self._num_classes = num_classes
        self._batch_images = batch_images
        self._batch_rois = batch_rois
        self._rois_per_image = int(batch_rois / batch_images)
        self._fg_rois_per_image = int(round(fg_fraction * self._rois_per_image))
        self._fg_overlap = fg_overlap
        self._box_stds = box_stds

    def forward(self, is_train, req, in_data, out_data, aux):
        assert self._batch_images == in_data[1].shape[0], 'check batch size of gt_boxes'

        all_rois = in_data[0].asnumpy()
        all_gt_boxes = in_data[1].asnumpy()

        rois = np.empty((0, 5), dtype=np.float32)
        labels = np.empty((0, ), dtype=np.float32)
        bbox_targets = np.empty((0, 4 * self._num_classes), dtype=np.float32)
        bbox_weights = np.empty((0, 4 * self._num_classes), dtype=np.float32)
        for batch_idx in range(self._batch_images):
            b_rois = all_rois[np.where(all_rois[:, 0] == batch_idx)[0]]
            b_gt_boxes = all_gt_boxes[batch_idx]
            b_gt_boxes = b_gt_boxes[np.where(b_gt_boxes[:, -1] > 0)[0]]

            # Include ground-truth boxes in the set of candidate rois
            batch_pad = batch_idx * np.ones((b_gt_boxes.shape[0], 1), dtype=b_gt_boxes.dtype)
            b_rois = np.vstack((b_rois, np.hstack((batch_pad, b_gt_boxes[:, :-1]))))

            b_rois, b_labels, b_bbox_targets, b_bbox_weights = \
                sample_rois(b_rois, b_gt_boxes, num_classes=self._num_classes, rois_per_image=self._rois_per_image,
                            fg_rois_per_image=self._fg_rois_per_image, fg_overlap=self._fg_overlap, box_stds=self._box_stds)

            rois = np.vstack((rois, b_rois))
            labels = np.hstack((labels, b_labels))
            bbox_targets = np.vstack((bbox_targets, b_bbox_targets))
            bbox_weights = np.vstack((bbox_weights, b_bbox_weights))

        self.assign(out_data[0], req[0], rois)
        self.assign(out_data[1], req[1], labels)
        self.assign(out_data[2], req[2], bbox_targets)
        self.assign(out_data[3], req[3], bbox_weights)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        self.assign(in_grad[0], req[0], 0)
        self.assign(in_grad[1], req[1], 0)


@mx.operator.register('proposal_target')
class ProposalTargetProp(mx.operator.CustomOpProp):
    def __init__(self, num_classes='21', batch_images='1', batch_rois='128', fg_fraction='0.25',
                 fg_overlap='0.5', box_stds='(0.1, 0.1, 0.2, 0.2)'):
        super(ProposalTargetProp, self).__init__(need_top_grad=False)#False：此层不需要传来的梯度
        self._num_classes = int(num_classes)
        self._batch_images = int(batch_images)
        self._batch_rois = int(batch_rois)
        self._fg_fraction = float(fg_fraction)
        self._fg_overlap = float(fg_overlap)
        self._box_stds = tuple(np.fromstring(box_stds[1:-1], dtype=float, sep=','))

    def list_arguments(self):
        return ['rois', 'gt_boxes']

    def list_outputs(self):
        return ['rois_output', 'label', 'bbox_target', 'bbox_weight']

    def infer_shape(self, in_shape):
        assert self._batch_rois % self._batch_images == 0, \
            'BATCHIMAGES {} must devide BATCH_ROIS {}'.format(self._batch_images, self._batch_rois)

        rpn_rois_shape = in_shape[0]
        gt_boxes_shape = in_shape[1]

        output_rois_shape = (self._batch_rois, 5)
        label_shape = (self._batch_rois, )
        bbox_target_shape = (self._batch_rois, self._num_classes * 4)
        bbox_weight_shape = (self._batch_rois, self._num_classes * 4)

        return [rpn_rois_shape, gt_boxes_shape], \
               [output_rois_shape, label_shape, bbox_target_shape, bbox_weight_shape]

    def create_operator(self, ctx, shapes, dtypes):
        return ProposalTargetOperator(self._num_classes, self._batch_images, self._batch_rois, self._fg_fraction,
                                      self._fg_overlap, self._box_stds)

    def declare_backward_dependency(self, out_grad, in_data, out_data):
        return []
