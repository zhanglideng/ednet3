import torch
from utils.ms_ssim import *


def l2_loss(input_image, output_image):
    l2_loss_fn = torch.nn.MSELoss(reduction='mean').cuda()
    return l2_loss_fn(input_image, output_image) * 100


def ssim_loss(input_image, output_image):
    losser = MS_SSIM(max_val=1)
    # losser = MS_SSIM(data_range=1.).cuda()
    return (1 - losser(input_image, output_image)) * 100


def loss_function(image):
    gt_image, output_image, gt_scene_feature, dehaze_image, hazy_scene_feature = image
    loss_train = [l2_loss(gt_image, dehaze_image),
                  ssim_loss(gt_image, dehaze_image),
                  l2_loss(gt_image, output_image),
                  ssim_loss(gt_image, output_image)]
    loss_ob = l2_loss(gt_scene_feature, hazy_scene_feature)
    return loss_train, loss_ob
