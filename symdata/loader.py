import mxnet as mx
import numpy as np

from symdata.anchor import AnchorGenerator, AnchorSampler
from symdata.image import imdecode, resize, transform, get_image, tensor_vstack


def load_test(filename, short, max_size, mean, std):
    # 读取和转换图片
    im_orig = imdecode(filename)
    im, im_scale = resize(im_orig, short, max_size)
    height, width = im.shape[:2]
    im_info = mx.nd.array([height, width, im_scale])

    # 转成需要的格式并标准化处理
    im_tensor = transform(im, mean, std)

    # 增加一维N，（N,C,H,W）
    im_tensor = mx.nd.array(im_tensor).expand_dims(0)
    im_info = mx.nd.array(im_info).expand_dims(0)

    #CV2是BGR,需转成RGB
    im_orig = im_orig[:, :, (2, 1, 0)]
    return im_tensor, im_info, im_orig


def generate_batch(im_tensor, im_info):
    data = [im_tensor, im_info]
    #类似[('data', (1, 3, 668, 600)), ('im_info', (1, 3))]
    data_shapes = [('data', im_tensor.shape), ('im_info', im_info.shape)]
    data_batch = mx.io.DataBatch(data=data, label=None, provide_data=data_shapes, provide_label=None)
    return data_batch


class TestLoader(mx.io.DataIter):
    def __init__(self, roidb, batch_size, short, max_size, mean, std):
        super(TestLoader, self).__init__()

        self._roidb = roidb
        self._batch_size = batch_size
        self._short = short
        self._max_size = max_size
        self._mean = mean
        self._std = std

        # 从roidb推断属性
        self._size = len(self._roidb)
        self._index = np.arange(self._size)

        # 决定数据名称和标签名称 (仅在训练时)
        self._data_name = ['data', 'im_info']
        self._label_name = None

        # 状态变量
        self._cur = 0
        self._data = None
        self._label = None

        self.next()
        self.reset()

    @property
    def provide_data(self):
        return [(k, v.shape) for k, v in zip(self._data_name, self._data)]

    @property
    def provide_label(self):
        return None

    def reset(self):
        self._cur = 0

    def iter_next(self):
        return self._cur + self._batch_size <= self._size

    def next(self):
        if self.iter_next():
            data_batch = mx.io.DataBatch(data=self.getdata(), label=self.getlabel(),
                                         pad=self.getpad(), index=self.getindex(),
                                         provide_data=self.provide_data, provide_label=self.provide_label)
            self._cur += self._batch_size
            return data_batch
        else:
            raise StopIteration

    def getdata(self):
        indices = self.getindex()
        im_tensor, im_info = [], []
        for index in indices:
            roi_rec = self._roidb[index]
            b_im_tensor, b_im_info, _ = get_image(roi_rec, self._short, self._max_size, self._mean, self._std)
            im_tensor.append(b_im_tensor)
            im_info.append(b_im_info)
        im_tensor = mx.nd.array(tensor_vstack(im_tensor, pad=0))
        im_info = mx.nd.array(tensor_vstack(im_info, pad=0))
        self._data = im_tensor, im_info
        return self._data

    def getlabel(self):
        return None

    def getindex(self):
        cur_from = self._cur
        cur_to = min(cur_from + self._batch_size, self._size)
        return np.arange(cur_from, cur_to)

    def getpad(self):
        return max(self._cur + self.batch_size - self._size, 0)


class AnchorLoader(mx.io.DataIter):
    def __init__(self, roidb, batch_size, short, max_size, mean, std,
                 feat_sym, anchor_generator: AnchorGenerator, anchor_sampler: AnchorSampler,
                 shuffle=False):
        super(AnchorLoader, self).__init__()

        # 将参数当属性保存
        self._roidb = roidb
        self._batch_size = batch_size
        self._short = short
        self._max_size = max_size
        self._mean = mean
        self._std = std
        self._feat_sym = feat_sym
        self._ag = anchor_generator
        self._as = anchor_sampler
        self._shuffle = shuffle

        # 从roidb推断属性
        self._size = len(roidb)
        self._index = np.arange(self._size)

        self._data_name = ['data', 'im_info', 'gt_boxes']
        self._label_name = ['label', 'bbox_target', 'bbox_weight']

        # 状态变量
        self._cur = 0
        self._data = None
        self._label = None

        # get first batch to fill in provide_data and provide_label
        self.next()
        self.reset()

    @property
    def provide_data(self):
        return [(k, v.shape) for k, v in zip(self._data_name, self._data)]

    @property
    def provide_label(self):
        return [(k, v.shape) for k, v in zip(self._label_name, self._label)]

    def reset(self):
        self._cur = 0
        if self._shuffle:
            np.random.shuffle(self._index)

    def iter_next(self):
        return self._cur + self._batch_size <= self._size

    def next(self):
        if self.iter_next():
            data_batch = mx.io.DataBatch(data=self.getdata(), label=self.getlabel(),
                                         pad=self.getpad(), index=self.getindex(),
                                         provide_data=self.provide_data, provide_label=self.provide_label)
            self._cur += self._batch_size
            return data_batch
        else:
            raise StopIteration

    def getdata(self):
        indices = self.getindex()
        im_tensor, im_info, gt_boxes = [], [], []
        for index in indices:
            roi_rec = self._roidb[index]
            b_im_tensor, b_im_info, b_gt_boxes = get_image(roi_rec, self._short, self._max_size, self._mean, self._std)
            im_tensor.append(b_im_tensor)
            im_info.append(b_im_info)
            gt_boxes.append(b_gt_boxes)
        im_tensor = mx.nd.array(tensor_vstack(im_tensor, pad=0))
        im_info = mx.nd.array(tensor_vstack(im_info, pad=0))
        gt_boxes = mx.nd.array(tensor_vstack(gt_boxes, pad=-1))
        self._data = im_tensor, im_info, gt_boxes
        return self._data

    def getlabel(self):
        im_tensor, im_info, gt_boxes = self._data

        # 所有堆叠的图像共享相同的锚框
        _, out_shape, _ = self._feat_sym.infer_shape(data=im_tensor.shape)
        feat_height, feat_width = out_shape[0][-2:]
        anchors = self._ag.generate(feat_height, feat_width)

        # 根据它们在im_info中编码的实际大小分配锚框
        label, bbox_target, bbox_weight = [], [], []
        for batch_ind in range(im_info.shape[0]):
            b_im_info = im_info[batch_ind].asnumpy()
            b_gt_boxes = gt_boxes[batch_ind].asnumpy()
            b_im_height, b_im_width = b_im_info[:2]

            b_label, b_bbox_target, b_bbox_weight = self._as.assign(anchors, b_gt_boxes, b_im_height, b_im_width)

            b_label = b_label.reshape((feat_height, feat_width, -1)).transpose((2, 0, 1)).flatten()
            b_bbox_target = b_bbox_target.reshape((feat_height, feat_width, -1)).transpose((2, 0, 1))
            b_bbox_weight = b_bbox_weight.reshape((feat_height, feat_width, -1)).transpose((2, 0, 1))

            label.append(b_label)
            bbox_target.append(b_bbox_target)
            bbox_weight.append(b_bbox_weight)

        label = mx.nd.array(tensor_vstack(label, pad=-1))
        bbox_target = mx.nd.array(tensor_vstack(bbox_target, pad=0))
        bbox_weight = mx.nd.array(tensor_vstack(bbox_weight, pad=0))
        self._label = label, bbox_target, bbox_weight
        return self._label

    def getindex(self):
        cur_from = self._cur
        cur_to = min(cur_from + self._batch_size, self._size)
        return np.arange(cur_from, cur_to)

    def getpad(self):
        return max(self._cur + self.batch_size - self._size, 0)
