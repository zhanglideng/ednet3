from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import pickle
import os
import cv2
import scipy.io as sio


class EdDataSet(Dataset):
    def __init__(self, transform1, path=None):
        print(path)
        self.transform1 = transform1
        self.haze_path, self.gt_path, self.t_path = path
        # self.haze_path, self.gt_path, self.depth_path = path
        self.haze_data_list = os.listdir(self.haze_path)
        self.gt_data_list = os.listdir(self.gt_path)
        self.t_data_list = os.listdir(self.t_path)
        # self.gt_depth_list = os.listdir(self.depth_path)
        self.haze_data_list.sort(key=lambda x: float(x[-8:-4]))
        self.haze_data_list.sort(key=lambda x: int(x[:-30]))
        # print(self.haze_data_list)
        self.gt_data_list.sort(key=lambda x: int(x[:-4]))
        self.t_data_list.sort(key=lambda x: int(x[:-4]))

    def __len__(self):
        return len(os.listdir(self.haze_path))

    def __getitem__(self, idx):
        """
            需要传递的信息有：
            有雾图像
            无雾图像
            (深度图)
            (雾度)
            (大气光)
        """
        haze_image_name = self.haze_data_list[idx]
        haze_image = cv2.imread(self.haze_path + haze_image_name)
        gt_image = cv2.imread(self.gt_path + haze_image_name[:-30] + '.bmp')
        t_image = np.load(self.t_path + haze_image_name[:-30] + '.npy')
        b = float(haze_image_name[-8:-4])
        t_image = np.expand_dims(t_image, axis=2)
        t_image = t_image.astype(np.float32)
        gt_t = np.exp(-1 * 0.01 * t_image)
        haze_t = np.exp(-1 * b * t_image)
        # print(gt_image.shape)
        # print(gt_t.shape)
        # data = sio.loadmat(self.depth_path + '/' + haze_image_name[:-30] + '.mat')
        # gt_depth = data["depths"]
        # gt_depth = gt_depth[:, :, np.newaxis]
        # print(gt_depth.shape)
        # print(haze_image.shape)
        # haze_fog = float(haze_image_name[-8:-4])
        # gt_fog = 0.01
        # gt_depth = gt_depth * gt_fog
        # print(haze_image.type)
        # haze_depth = gt_depth * haze_fog
        # print(haze_depth)
        # print(gt_depth)
        if self.transform1:
            haze_image = self.transform1(haze_image)
            gt_image = self.transform1(gt_image)
            gt_t = self.transform1(gt_t)
            haze_t = self.transform1(haze_t)
        # if self.transform2:
        #    gt_depth = self.transform2(gt_depth)
        #    haze_depth = self.transform2(haze_depth)
        # 暂不传递大气光值
        return haze_image, gt_image, gt_t, haze_t


if __name__ == '__main__':
    train_haze_path = './data/train/'
    validation_haze_path = './data/validation/'
    test_haze_path = './data/test/'
    gt_path = './data/GT'
    depth_path = './data/depth'
    path_list = [test_haze_path, gt_path, depth_path]
    print(path_list)
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    data = Haze_Dataset(transform, path_list)
    dataloader = DataLoader(data, batch_size=4, shuffle=True, num_workers=4)
    count = 0
    for i in dataloader:
        haze_image, gt_image, gt_depth, fog = i
        print('haze_image.shape:' + str(haze_image.shape))
        print('gt_image.shape:' + str(gt_image.shape))
        print('gt_depth.shape:' + str(gt_depth.shape))
        print('fog:' + str(fog))
        # print(hazy.shape)
        # print(gt.shape)
        count += 1
    print('count:' + str(count))
