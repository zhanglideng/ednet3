# -*- coding: utf-8 -*-

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
from utils.loss import *
from utils.print_time import *
from utils.save_log_to_excel import *
from dataloader import EdDataSet
from Res_ED_model import CNN
import time
import xlwt
from utils.ms_ssim import *
import os

LR = 0.0004  # 学习率
EPOCH = 100  # 轮次
BATCH_SIZE = 4  # 批大小
excel_train_line = 1  # train_excel写入的行的下标
excel_val_line = 1  # val_excel写入的行的下标
alpha = 1  # 损失函数的权重
accumulation_steps = 1  # 梯度积累的次数，类似于batch-size=64
itr_to_lr = 10000 // BATCH_SIZE  # 训练10000次后损失下降50%
itr_to_excel = 64 // BATCH_SIZE  # 训练64次后保存相关数据到excel
loss_num = 5  # 包括参加训练和不参加训练的loss
train_haze_path = '/input/data/nyu/train/'  # 去雾训练集的路径
val_haze_path = '/input/data/nyu/val/'  # 去雾验证集的路径
gt_path = '/input/data/nyu/gth/'
save_path = './checkpoints/best_cnn_model.pt'  # 保存模型的路径
excel_save = './result.xls'  # 保存excel的路径


def adjust_learning_rate(op, i):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = LR * (0.90 ** (i // itr_to_lr))
    for param_group in op.param_groups:
        param_group['lr'] = lr


# 初始化excel
f, sheet_train, sheet_val = init_excel()
# 加载模型
model_path = './checkpoints/best_cnn_model.pt'
net = torch.load(model_path)
net = net.cuda()
print(net)
for param in net.decoder.parameters():
    param.requires_grad = False

# 数据转换模式
transform = transforms.Compose([transforms.ToTensor()])
# transform = transforms.Compose(
#    [transforms.ToTensor(), transforms.Normalize(mean=[0.489, 0.490, 0.491], std=[0.312, 0.312, 0.312])])
# 读取训练集数据
train_path_list = [train_haze_path, gt_path]
train_data = EdDataSet(transform, train_path_list)
train_data_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

# 读取验证集数据
val_path_list = [val_haze_path, gt_path]
validation_data = EdDataSet(transform, val_path_list)
validation_data_loader = DataLoader(validation_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

# 定义优化器
optimizer = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=1e-5)

min_loss = 999999999
min_epoch = 0
itr = 0
start_time = time.time()

# 开始训练
print("\nstart to train!")
for epoch in range(EPOCH):
    index = 0
    train_epo_loss = 0
    validation_epo_loss = 0
    '''
    l2_loss_excel = 0
    l2_sf_loss_excel = 0
    ssim_loss_excel = 0
    '''
    loss = 0
    loss_excel = [0] * loss_num
    """
    创建一个可以
    """
    for input_image, gt_image in train_data_loader:
        index += 1
        itr += 1
        input_image = input_image.cuda()
        gt_image = gt_image.cuda()
        output_image, gt_scene_feature = net(gt_image)
        dehaze_image, hazy_scene_feature = net(input_image)
        loss_train, loss_ob = loss_function(
            [gt_image, output_image, gt_scene_feature, dehaze_image, hazy_scene_feature])
        for x in loss_train:
            loss += x
        '''
        l2 = l2_loss(output_image, gt_image)
        l2_sf = l2_loss(gt_scene_feature, hazy_scene_feature)
        ssim = ssim_loss(output_image, gt_image)
        loss = ssim + l2 + l2_sf
        l2_loss_excel += l2.item()
        l2_sf_loss_excel += l2_sf.item()
        ssim_loss_excel += ssim.item()
        '''
        temp = loss_train.item()
        for x in loss_ob.item():
            temp.append(x)
        for x in temp:
            loss_excel += temp
        loss.backward()
        # optimizer.step()
        iter_loss = loss.item()
        train_epo_loss += iter_loss
        loss = loss / accumulation_steps
        if itr % itr_to_lr == 0:
            adjust_learning_rate(optimizer, itr)
        # 3. update parameters of net
        if ((index + 1) % accumulation_steps) == 0:
            # optimizer the net
            optimizer.step()  # update parameters of net
            optimizer.zero_grad()  # reset gradient
            loss = 0
        if np.mod(index, itr_to_excel) == 0:
            print('epoch %d, %03d/%d' % (epoch + 1, index, len(train_data_loader)))
            print('dehaze_l2_loss=%.5f\n'
                  'dehaze_ssim_loss=%.5f\n'
                  're_l2_loss=%.5f\n'
                  're_ssim_loss=%.5f\n'
                  'l2_sf_loss=%.5f' % (
                      loss_excel[0] / itr_to_excel,
                      loss_excel[1] / itr_to_excel,
                      loss_excel[2] / itr_to_excel,
                      loss_excel[3] / itr_to_excel,
                      loss_excel[4] / itr_to_excel))
            print_time(start_time, index, EPOCH, len(train_data_loader), epoch)
            excel_train_line = write_excel(sheet=sheet_train,
                                           data_type='train',
                                           line=excel_train_line,
                                           epoch=epoch,
                                           itr=itr,
                                           loss=loss_excel,
                                           lr=LR * (0.90 ** (itr // itr_to_lr)))
            f.save(excel_save)
    optimizer.step()
    optimizer.zero_grad()
    # 验证集的损失计算
    val_ed_ssim_loss = 0
    val_ed_l2_loss = 0
    val_l2_sf_loss = 0
    val_dehaze_l2_loss = 0
    val_dehaze_ssim_loss = 0
    with torch.no_grad():
        for input_image, gt_image in validation_data_loader:
            # input_image = item['input_image']

            """
            input_image = input_image.cuda()
            gt_image = gt_image.cuda()
            output_image, gt_scene_feature = net(gt_image)
            hazy_scene_feature = net.encoder(input_image)
            l2 = l2_loss(output_image, gt_image)
            l2_sf = l2_loss(gt_scene_feature, hazy_scene_feature)
            ssim = ssim_loss(output_image, gt_image)
            loss = ssim + l2 + l2_sf
            """

            input_image = input_image.cuda()
            gt_image = gt_image.cuda()
            output_image, gt_scene_feature = net(gt_image)
            dehaze_image, hazy_scene_feature = net(input_image)
            val_ed_ssim_loss += ssim_loss(output_image, gt_image).item()
            val_ed_l2_loss += l2_loss(output_image, gt_image).item()
            val_l2_sf_loss += l2_loss(gt_scene_feature, hazy_scene_feature).item()
            val_dehaze_l2_loss += l2_loss(output_image, dehaze_image).item()
            val_dehaze_ssim_loss += ssim_loss(output_image, dehaze_image).item()
    train_epo_loss = train_epo_loss / len(train_data_loader)
    val_ed_ssim_loss = val_ed_ssim_loss / len(validation_data_loader)
    val_ed_l2_loss = val_ed_l2_loss / len(validation_data_loader)
    val_l2_sf_loss = val_l2_sf_loss / len(validation_data_loader)
    val_dehaze_l2_loss = val_dehaze_l2_loss / len(validation_data_loader)
    val_dehaze_ssim_loss = val_dehaze_ssim_loss / len(validation_data_loader)
    print('\nepoch %d train loss = %.5f' % (epoch + 1, train_epo_loss))
    print('epoch %d val_ed_ssim_loss = %.5f' % (epoch + 1, val_ed_ssim_loss))
    print('epoch %d val_ed_l2_loss = %.5f' % (epoch + 1, val_ed_l2_loss))
    print('epoch %d val_l2_sf_loss = %.5f' % (epoch + 1, val_l2_sf_loss))
    print('epoch %d validation loss = %.5f' % (epoch + 1, alpha * val_ed_ssim_loss + val_ed_l2_loss + val_l2_sf_loss))
    print('epoch %d dehaze l2 loss = %.5f' % (epoch + 1, val_dehaze_l2_loss))
    print('epoch %d dehaze ssim loss = %.5f' % (epoch + 1, val_dehaze_ssim_loss))
    # val=["EPOCH", "L2_LOSS", "SSIM_LOSS", "LOSS", "PSNR", "SSIM", "LR"]
    # (sheet, data_type, line, epoch, itr, l2_loss, ssim_loss, loss, psnr, ssim, lr)
    excel_val_line = write_excel(sheet=sheet_val,
                                 data_type='val',
                                 line=excel_val_line,
                                 epoch=epoch,
                                 itr=False,
                                 l2_loss=val_ed_l2_loss,
                                 ssim_loss=val_ed_ssim_loss,
                                 l2_sf_loss=val_l2_sf_loss,
                                 dehaze_ssim_loss=val_dehaze_ssim_loss,
                                 dehaze_l2_loss=val_dehaze_l2_loss,
                                 loss=alpha * val_ed_ssim_loss + val_ed_l2_loss + val_l2_sf_loss,
                                 lr=LR * (0.90 ** (itr // itr_to_lr)))
    f.save(excel_save)
    # if alpha * val_ssim_loss + val_l2_loss < min_loss:
    if alpha * val_ed_ssim_loss + val_l2_sf_loss + val_ed_l2_loss < min_loss:
        min_loss = alpha * val_ed_ssim_loss + val_l2_sf_loss + val_ed_l2_loss
        min_epoch = epoch
        torch.save(net, save_path)
        print('saving the epoch %d model with %.5f' % (epoch + 1, min_loss))
        # no_update = 0
        # LR_flag = 0
    else:
        print('not improve for epoch %d with %.5f' % (min_epoch, min_loss))
    print('learning rate is ' + str(LR) + '\n')
print('Train is Done!')
