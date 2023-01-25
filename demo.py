import argparse
import ast
import pprint

import mxnet as mx
from mxnet.module import Module

from symdata.bbox import im_detect
from symdata.loader import load_test, generate_batch
from symdata.vis import vis_detection
from symnet.model import load_param, check_shape

# 测试结果
# python demo.py --dataset voc --network vgg16 --params model/vgg_voc07-0010.params --image hi.jpg
# 可视化
# python demo.py --dataset voc --network vgg16 --params model/vgg_voc07-0010.params --image hi.jpg --vis

def demo_net(sym, class_names, args):
    # vars()：返回对象的属性与属性值的字典对象，指定参数args就是返回这个属性的内容
    # 返回的必须是字典对象，也就是其拥有 __dict__ 属性
    #print('called with args\n{}'.format(pprint.pformat(vars(args))))
    if args.gpu:
        ctx = mx.gpu(int(args.gpu))
    else:
        ctx = mx.cpu(0)

    # 加载单个测试
    im_tensor, im_info, im_orig = load_test(args.image, short=args.img_short_side, max_size=args.img_long_side,
                                            mean=args.img_pixel_means, std=args.img_pixel_stds)

    # 生成批处理数据
    # 类似DataBatch: data shapes: [(1, 3, 668, 600), (1, 3)] label shapes: None
    data_batch = generate_batch(im_tensor, im_info)

    # 加载读取参数(训练好的参数文件.params)
    arg_params, aux_params = load_param(args.params, ctx=ctx)
    #print(arg_params.keys())
    '''
    dict_keys(['conv3_2_weight', 'rpn_conv_3x3_weight', 'conv4_1_bias', 'conv5_3_bias', 
    'cls_score_weight', 'conv3_3_bias', 'fc7_bias', 'conv4_3_weight', 'conv1_2_bias', 'conv4_1_weight', 
    'bbox_pred_bias', 'bbox_pred_weight', 'fc7_weight', 'conv2_1_bias', 'conv5_2_weight', 'conv1_1_bias', 
    'cls_score_bias', 'bbox_pred_weight_test', 'conv2_2_weight', 'conv4_3_bias', 'fc6_bias', 'bbox_pred_bias_test', 
    'conv3_1_bias', 'rpn_bbox_pred_bias', 'rpn_cls_score_bias', 'conv5_3_weight', 'conv1_2_weight', 'conv4_2_weight', 
    'rpn_cls_score_weight', 'conv3_1_weight', 'conv5_1_bias', 'conv4_2_bias', 'conv2_1_weight', 'conv2_2_bias', 'conv5_1_weight', 
    'conv1_1_weight', 'rpn_conv_3x3_bias', 'rpn_bbox_pred_weight', 'conv3_3_weight', 'fc6_weight', 'conv3_2_bias', 'conv5_2_bias'])
    '''
    data_names = ['data', 'im_info']
    label_names = None
    data_shapes = [('data', (1, 3, args.img_long_side, args.img_long_side)), ('im_info', (1, 3))]
    label_shapes = None

    # 检测推断出来的形状是否跟参数中对应的形状一样，不一样就断言
    # 符号式编程的了解：https://blog.csdn.net/weixin_41896770/article/details/125370472
    check_shape(sym, data_shapes, arg_params, aux_params)

    # 创建并绑定模型
    mod = Module(sym, data_names, label_names, context=ctx)
    mod.bind(data_shapes, label_shapes, for_training=False)
    mod.init_params(arg_params=arg_params, aux_params=aux_params)

    # 前向计算
    mod.forward(data_batch)
    rois, scores, bbox_deltas = mod.get_outputs()
    rois = rois[:, 1:]
    scores = scores[0]
    bbox_deltas = bbox_deltas[0]
    im_info = im_info[0]

    # 检测结果
    det = im_detect(rois, scores, bbox_deltas, im_info,
                    bbox_stds=args.rcnn_bbox_stds, nms_thresh=args.rcnn_nms_thresh,
                    conf_thresh=args.rcnn_conf_thresh)

    # 打印出检测结果(非背景,大于指定阈值,坐标值)
    for [cls, conf, x1, y1, x2, y2] in det:
        if cls > 0 and conf > args.vis_thresh:
            print(class_names[int(cls)], conf, [x1, y1, x2, y2])

    # 可视化
    if args.vis:
        vis_detection(im_orig, det, class_names, thresh=args.vis_thresh)


def parse_args():
    parser = argparse.ArgumentParser(description='Demonstrate a Faster R-CNN network',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--network', type=str,default='vgg16', help='base network')
    parser.add_argument('--params', type=str, default='',help='path to trained model')
    parser.add_argument('--dataset', type=str,default='voc', help='training dataset')
    parser.add_argument('--image', type=str, default='',help='path to test image')
    parser.add_argument('--gpu', type=str, default='', help='gpu device eg. 0')
    parser.add_argument('--vis', action='store_true', help='display results')
    parser.add_argument('--vis-thresh', type=float,default=0.7, help='threshold display boxes')
    # faster rcnn params
    parser.add_argument('--img-short-side', type=int, default=600)
    parser.add_argument('--img-long-side', type=int, default=1000)
    parser.add_argument('--img-pixel-means', type=str,default='(0.0, 0.0, 0.0)')
    parser.add_argument('--img-pixel-stds', type=str,default='(1.0, 1.0, 1.0)')
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
    parser.add_argument('--rcnn-bbox-stds', type=str,default='(0.1, 0.1, 0.2, 0.2)')
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

# 获取VOC数据集的类别名称
def get_voc_names(args):
    from symimdb.pascal_voc import PascalVOC
    args.rcnn_num_classes = len(PascalVOC.classes)
    return PascalVOC.classes

# 获取COCO数据集的类别名称
def get_coco_names(args):
    from symimdb.coco import coco
    args.rcnn_num_classes = len(coco.classes)
    return coco.classes


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


def get_class_names(dataset, args):
    datasets = {
        'voc': get_voc_names,
        'coco': get_coco_names
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
    class_names = get_class_names(args.dataset, args)
    sym = get_network(args.network, args)
    demo_net(sym, class_names, args)


if __name__ == '__main__':
    main()
