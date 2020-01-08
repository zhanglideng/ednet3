import torch
from utils.ms_ssim import *
import math
import torch.nn.functional


def l2_loss(input_image, output_image):
    l2_loss_fn = torch.nn.MSELoss(reduction='mean').cuda()
    return l2_loss_fn(input_image, output_image) * 100


def ssim_loss(input_image, output_image):
    losser = MS_SSIM(max_val=1)
    # losser = MS_SSIM(data_range=1.).cuda()
    return (1 - losser(input_image, output_image)) * 100


def color_loss(input_image, output_image):
    vec1 = input_image.view([-1, 3])
    vec2 = output_image.view([-1, 3])
    clip_value = 0.999999
    norm_vec1 = torch.nn.functional.normalize(vec1)
    norm_vec2 = torch.nn.functional.normalize(vec2)
    dot = norm_vec1 * norm_vec2
    dot = dot.mean(dim=1)
    dot = torch.clamp(dot, -clip_value, clip_value)
    angle = torch.acos(dot) * (180 / math.pi)
    return angle.mean()


def loss_function(image):
    gt_image, output_image, gt_scene_feature, dehaze_image, hazy_scene_feature = image
    loss_train = [l2_loss(gt_image, dehaze_image),
                  ssim_loss(gt_image, dehaze_image),
                  l2_loss(gt_image, output_image),
                  ssim_loss(gt_image, output_image),
                  l2_loss(gt_scene_feature, hazy_scene_feature)]
    loss_ob = []
    return loss_train, loss_ob
