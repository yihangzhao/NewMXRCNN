import argparse
import ast
import pprint

import mxnet as mx
from mxnet.module import Module
import numpy as np
from tqdm import tqdm

from symdata.bbox import im_detect
from symdata.loader import TestLoader
from symnet.logger import logger
from symnet.model import load_param, check_shape

#python test.py --dataset voc --network vgg16 --params model/vgg16-0010.params
def test_net(sym, imdb, args):
    logger.info('called with args\n{}'.format(pprint.pformat(vars(args))))
    ctx = mx.gpu(args.gpu)
    # 加载测试数据
    test_data = TestLoader(imdb.roidb, batch_size=1, short=args.img_short_side, max_size=args.img_long_side,
                           mean=args.img_pixel_means, std=args.img_pixel_stds)

    # 加载参数
    arg_params, aux_params = load_param(args.params, ctx=ctx)

    data_names = ['data', 'im_info']
    label_names = None
    data_shapes = [('data', (1, 3, args.img_long_side, args.img_long_side)), ('im_info', (1, 3))]
    label_shapes = None

    check_shape(sym, data_shapes, arg_params, aux_params)

    # 创建和绑定模型，初始化参数和辅助参数
    # 符号式编程可以查阅：https://blog.csdn.net/weixin_41896770/article/details/127400770
    # https://blog.csdn.net/weixin_41896770/article/details/125370472
    mod = Module(sym, data_names, label_names, context=ctx)
    mod.bind(data_shapes, label_shapes, for_training=False)
    mod.init_params(arg_params=arg_params, aux_params=aux_params)

    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    all_boxes = [[[] for _ in range(imdb.num_images)] for _ in range(imdb.num_classes)]

    # tqdm是进度条
    with tqdm(total=imdb.num_images) as pbar:
        for i, data_batch in enumerate(test_data):
            # 前向计算
            im_info = data_batch.data[1][0]
            mod.forward(data_batch)
            rois, scores, bbox_deltas = mod.get_outputs()
            rois = rois[:, 1:]
            scores = scores[0]
            bbox_deltas = bbox_deltas[0]

            det = im_detect(rois, scores, bbox_deltas, im_info,
                            bbox_stds=args.rcnn_bbox_stds, nms_thresh=args.rcnn_nms_thresh,
                            conf_thresh=args.rcnn_conf_thresh)
            for j in range(1, imdb.num_classes):
                indexes = np.where(det[:, 0] == j)[0]
                all_boxes[j][i] = np.concatenate((det[:, -4:], det[:, [1]]), axis=-1)[indexes, :]#axis=-1表最后维
            pbar.update(data_batch.data[0].shape[0])

    # 评估模型
    imdb.evaluate_detections(all_boxes)


def parse_args():
    parser = argparse.ArgumentParser(description='Test a Faster R-CNN network',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--network', type=str, default='vgg16', help='base network')
    parser.add_argument('--params', type=str, default='', help='path to trained model')
    parser.add_argument('--dataset', type=str, default='voc', help='training dataset')
    parser.add_argument('--imageset', type=str, default='', help='imageset splits')
    parser.add_argument('--gpu', type=int, default=0, help='gpu device eg. 0')
    # faster rcnn params
    parser.add_argument('--img-short-side', type=int, default=600)
    parser.add_argument('--img-long-side', type=int, default=1000)
    parser.add_argument('--img-pixel-means', type=str, default='(0.0, 0.0, 0.0)')
    parser.add_argument('--img-pixel-stds', type=str, default='(1.0, 1.0, 1.0)')
    parser.add_argument('--rpn-feat-stride', type=int, default=16)
    parser.add_argument('--rpn-anchor-scales', type=str, default='(8, 16, 32)')
    parser.add_argument('--rpn-anchor-ratios', type=str, default='(0.5, 1, 2)')
    parser.add_argument('--rpn-pre-nms-topk', type=int, default=6000)
    parser.add_argument('--rpn-post-nms-topk', type=int, default=300)
    parser.add_argument('--rpn-nms-thresh', type=float, default=0.7)
    parser.add_argument('--rpn-min-size', type=int, default=16)
    parser.add_argument('--rcnn-num-classes', type=int, default=21)
    parser.add_argument('--rcnn-feat-stride', type=int, default=16)
    parser.add_argument('--rcnn-pooled-size', type=str, default='(14, 14)')
    parser.add_argument('--rcnn-batch-size', type=int, default=1)
    parser.add_argument('--rcnn-bbox-stds', type=str, default='(0.1, 0.1, 0.2, 0.2)')
    parser.add_argument('--rcnn-nms-thresh', type=float, default=0.3)
    parser.add_argument('--rcnn-conf-thresh', type=float, default=1e-3)
    args = parser.parse_args()
    args.img_pixel_means = ast.literal_eval(args.img_pixel_means)
    args.img_pixel_stds = ast.literal_eval(args.img_pixel_stds)
    args.rpn_anchor_scales = ast.literal_eval(args.rpn_anchor_scales)
    args.rpn_anchor_ratios = ast.literal_eval(args.rpn_anchor_ratios)
    args.rcnn_pooled_size = ast.literal_eval(args.rcnn_pooled_size)
    args.rcnn_bbox_stds = ast.literal_eval(args.rcnn_bbox_stds)
    return args


def get_voc(args):
    from symimdb.pascal_voc import PascalVOC
    if not args.imageset:
        args.imageset = '2007_test'
    args.rcnn_num_classes = len(PascalVOC.classes)
    #return PascalVOC(args.imageset, 'data', 'data/VOCdevkit')
    return PascalVOC(args.imageset, '../data', '../data/VOCdevkit')


def get_coco(args):
    from symimdb.coco import coco
    if not args.imageset:
        args.imageset = 'val2017'
    args.rcnn_num_classes = len(coco.classes)
    return coco(args.imageset, 'data', 'data/coco')


def get_vgg16_test(args):
    from symnet.symbol_vgg import get_vgg_test
    if not args.params:
        args.params = 'model/vgg16-0010.params'
    args.img_pixel_means = (123.68, 116.779, 103.939)
    args.img_pixel_stds = (1.0, 1.0, 1.0)
    args.net_fixed_params = ['conv1', 'conv2']
    args.rpn_feat_stride = 16
    args.rcnn_feat_stride = 16
    args.rcnn_pooled_size = (7, 7)
    return get_vgg_test(anchor_scales=args.rpn_anchor_scales, anchor_ratios=args.rpn_anchor_ratios,
                        rpn_feature_stride=args.rpn_feat_stride, rpn_pre_topk=args.rpn_pre_nms_topk,
                        rpn_post_topk=args.rpn_post_nms_topk, rpn_nms_thresh=args.rpn_nms_thresh,
                        rpn_min_size=args.rpn_min_size,
                        num_classes=args.rcnn_num_classes, rcnn_feature_stride=args.rcnn_feat_stride,
                        rcnn_pooled_size=args.rcnn_pooled_size, rcnn_batch_size=args.rcnn_batch_size)


def get_resnet50_test(args):
    from symnet.symbol_resnet import get_resnet_test
    if not args.params:
        args.params = 'model/resnet50-0010.params'
    args.img_pixel_means = (0.0, 0.0, 0.0)
    args.img_pixel_stds = (1.0, 1.0, 1.0)
    args.rpn_feat_stride = 16
    args.rcnn_feat_stride = 16
    args.rcnn_pooled_size = (14, 14)
    return get_resnet_test(anchor_scales=args.rpn_anchor_scales, anchor_ratios=args.rpn_anchor_ratios,
                           rpn_feature_stride=args.rpn_feat_stride, rpn_pre_topk=args.rpn_pre_nms_topk,
                           rpn_post_topk=args.rpn_post_nms_topk, rpn_nms_thresh=args.rpn_nms_thresh,
                           rpn_min_size=args.rpn_min_size,
                           num_classes=args.rcnn_num_classes, rcnn_feature_stride=args.rcnn_feat_stride,
                           rcnn_pooled_size=args.rcnn_pooled_size, rcnn_batch_size=args.rcnn_batch_size,
                           units=(3, 4, 6, 3), filter_list=(256, 512, 1024, 2048))


def get_resnet101_test(args):
    from symnet.symbol_resnet import get_resnet_test
    if not args.params:
        args.params = 'model/resnet101-0010.params'
    args.img_pixel_means = (0.0, 0.0, 0.0)
    args.img_pixel_stds = (1.0, 1.0, 1.0)
    args.rpn_feat_stride = 16
    args.rcnn_feat_stride = 16
    args.rcnn_pooled_size = (14, 14)
    return get_resnet_test(anchor_scales=args.rpn_anchor_scales, anchor_ratios=args.rpn_anchor_ratios,
                           rpn_feature_stride=args.rpn_feat_stride, rpn_pre_topk=args.rpn_pre_nms_topk,
                           rpn_post_topk=args.rpn_post_nms_topk, rpn_nms_thresh=args.rpn_nms_thresh,
                           rpn_min_size=args.rpn_min_size,
                           num_classes=args.rcnn_num_classes, rcnn_feature_stride=args.rcnn_feat_stride,
                           rcnn_pooled_size=args.rcnn_pooled_size, rcnn_batch_size=args.rcnn_batch_size,
                           units=(3, 4, 23, 3), filter_list=(256, 512, 1024, 2048))


def get_dataset(dataset, args):
    datasets = {
        'voc': get_voc,
        'coco': get_coco
    }
    if dataset not in datasets:
        raise ValueError("dataset {} not supported".format(dataset))
    return datasets[dataset](args)


def get_network(network, args):
    networks = {
        'vgg16': get_vgg16_test,
        'resnet50': get_resnet50_test,
        'resnet101': get_resnet101_test
    }
    if network not in networks:
        raise ValueError("network {} not supported".format(network))
    return networks[network](args)


def main():
    args = parse_args()
    imdb = get_dataset(args.dataset, args)
    #print(type(imdb),imdb.num_classes,imdb.name)#<class 'symimdb.pascal_voc.PascalVOC'> 21 voc_2007_test
    sym = get_network(args.network, args)
    #test_net(sym, imdb, args)

if __name__ == '__main__':
    main()