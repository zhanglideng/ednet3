import sys
import argparse
import time
import glob
from torch.autograd import Variable
import torch
import torch.nn as nn
import os
from torchvision import transforms
from dataloader import EdDataSet
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
from Res_ED_model import CNN
import torch
from utils.loss import *

test_path = '/input/data/nyu/test/'
gth_path = '/input/data/nyu/gth/'
t_gth = '/input/data/nyu/depth/'
BATCH_SIZE = 2


def get_image_for_save(img):
    img = img.numpy()
    img = np.squeeze(img)
    img = img * 255
    img[img < 0] = 0
    img[img > 255] = 255
    img = np.rollaxis(img, 0, 3)
    img = img.astype('uint8')
    return img
'''
save_path = 'result_{}'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
if not os.path.exists(save_path):
    os.makedirs(save_path)
'''
model_path = './checkpoints/best_cnn_model.pt'
net = torch.load(model_path)
transform = transforms.Compose([transforms.ToTensor()])
test_path_list = [test_path, gth_path, t_gth]
test_data = EdDataSet(transform, test_path_list)
test_data_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
print(">>Start testing...\n")

avg_elapsed_time = 0.0
count = 0
avg_l2_loss = 0
avg_ssim_loss = 0
for input_image, gt_image, gt_t, haze_t in test_data_loader:
    count += 1
    if count % 100 == 0:
        print(">>Processing ./{}".format(str(count)))
        print("the average L2 loss is {}\n"
              "the average SSIM loss is {}\n".format(avg_l2_loss / count,
                                                     avg_ssim_loss / count))
        break
    with torch.no_grad():
        net = net.cuda()
        # net.train(False)
        # net.eval()
        # 上面这两句话到底有什么问题
        print(">>Processing ./{}".format(str(count)))
        input_image = input_image.cuda()
        gt_image = gt_image.cuda()
        haze_t = haze_t.cuda()
        start_time = time.time()
        dehaze_image, hazy_scene_feature = net(input_image, haze_t)
        l2 = l2_loss(dehaze_image, gt_image).item()
        ssim = ssim_loss(dehaze_image, gt_image).item()
        # print('l2 = %f' % l2)
        # print('ssim = %f' % ssim)
        avg_l2_loss += l2
        avg_ssim_loss += ssim
        elapsed_time = time.time() - start_time
        avg_elapsed_time += elapsed_time
        # print('avg_l2_loss = %f' % (avg_l2_loss / count))
        # print('avg_ssim_loss = %f' % (avg_ssim_loss / count))

    dehaze_image = dehaze_image.cpu()
    # print(">>Processing ./{}".format(str(count)))
    for i in range(BATCH_SIZE):
        im_output_for_save = get_image_for_save(dehaze_image[i])
        filename = str((count - 1) * BATCH_SIZE + i) + '.bmp'
        cv2.imwrite(os.path.join(save_path, filename), im_output_for_save)

print(">>Finished!"
      "It takes average {}s for processing single image\n"
      "the average L2 loss is {}\n"
      "the average SSIM loss is {}\n".format(avg_elapsed_time / (count * BATCH_SIZE),
                                             avg_l2_loss / count,
                                             avg_ssim_loss / count))
