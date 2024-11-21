from __future__ import division

import os
import numpy as np
import cv2
# from scipy.misc import imresize
from skimage.transform import resize 

# from dataloaders.helpers import *
from torch.utils.data import Dataset
from PIL import Image
import shutil
import random
import re


class LSE(Dataset):
    """DAVIS 2016 dataset constructed using the PyTorch built-in functionalities"""

    def __init__(self, train=True,
                 inputRes=None,
                 db_root_dir='./Data',
                 transform=None,
                 seq_name=None):
        """Loads image to label pairs for tool pose estimation
        db_root_dir: dataset directory with subfolders "JPEGImages" and "Annotations"
        """
        self.train = train
        self.inputRes = inputRes
        self.db_root_dir = db_root_dir
        self.transform = transform
        self.seq_name = seq_name
        self.meanval = get_meanval()

        if self.train:
            fname = 'train_seq'
        else:
            fname = 'test_seq'

        if self.seq_name is None:
            # print(os.path.join(db_root_dir, fname + '.txt'))

            # Initialize the original DAVIS splits for training the parent network
            with open(os.path.join(db_root_dir, fname + '.txt')) as f:
                seqs = f.readlines()
                img_list = []
                labels = []
                for seq in seqs:
                    images = np.sort(os.listdir(os.path.join(db_root_dir, 'preDataset/inputs/', seq.strip())))
                    # [00000.jpg ,00001.jpg ,......]
                    images_path = list(map(lambda x: os.path.join('preDataset/inputs/', seq.strip(), x), images))
                    # [每一张图片的具体路径]
                    img_list.extend(images_path)
                    # [bear序列所有图片的路径 ,..., ......]
                    lab = np.sort(os.listdir(os.path.join(db_root_dir, 'preDataset/targets/', seq.strip())))
                    lab_path = list(map(lambda x: os.path.join('preDataset/targets/', seq.strip(), x), lab))
                    labels.extend(lab_path)
        else:

            # Initialize the per sequence images for online training
            names_img = np.sort(os.listdir(os.path.join(db_root_dir, 'preDataset/inputs/', str(seq_name))))
            # [00000.jpg ,......]
            img_list = list(map(lambda x: os.path.join('preDataset/inputs/', str(seq_name), x), names_img))
            # [每一张图片的具体路径]
            name_label = np.sort(os.listdir(os.path.join(db_root_dir, 'preDataset/targets/', str(seq_name))))
            labels = [os.path.join('preDataset/targets/', str(seq_name), name_label[0])]
            labels.extend([None]*(len(names_img)-1))
            # 用来确保labels和names_img长度相等
            if self.train:
            # 训练模式，只使用第一张图片。
                img_list = [img_list[0]]
                labels = [labels[0]]

        assert (len(labels) == len(img_list))

        self.img_list = img_list
        self.labels = labels
        # 若seq_name = None ,则img_list应为[bear序列所有图片的路径 ,..., ......]
        # 若seq_name = blackswan ,则img_list应为[/.../blackswan/00000.jpg],labels同理

        print('Done initializing ' + fname + ' Dataset')

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img, gt = self.make_img_gt_pair(idx)
        # 若若seq_name = None ,

        sample = {'image': img, 'gt': gt}

        if self.seq_name is not None:
            fname = os.path.join(self.seq_name, "%05d" % idx)
            # blackswan/00000
            sample['fname'] = fname

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def make_img_gt_pair(self, idx):
        """
        Make the image-ground-truth pair
        """
        # 若seq_name不是none,则应该包含所有图片
        img = cv2.imread(os.path.join(self.db_root_dir, self.img_list[idx]), 0)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) 
        # print( self.img_list[idx])
        # print(os.path.join(self.db_root_dir, self.img_list[idx]))
        # 若seq_name不是none，那这里的idx只能为0，也就是说数据集里只有一对图片，6，这应该是微调时候用的？
        if self.labels[idx] is not None:
            label = cv2.imread(os.path.join(self.db_root_dir, self.labels[idx]), 0)
        else:
            gt = np.zeros(img.shape[:-1], dtype=np.uint8)

        if self.inputRes is not None:
            # img = imresize(img, self.inputRes)
            img = resize(img, self.inputRes)
            if self.labels[idx] is not None:
                # label = imresize(label, self.inputRes, interp='nearest')
                label = resize(label, self.inputRes, interp='nearest')

        img = np.array(img, dtype=np.float32)
        img = np.subtract(img, np.array(self.meanval, dtype=np.float32))

        if self.labels[idx] is not None:
                gt = np.array(label, dtype=np.float32)
                gt = gt/np.max([gt.max(), 1e-8])

        return img, gt

    def get_img_size(self):
        img = cv2.imread(os.path.join(self.db_root_dir, self.img_list[0]), 0)

        return list(img.shape[:2])

def split_dataset(datasets_path='./Data/preDataset',p=0.7):
    # names = os.listdir(datasets_path)
    # for name in names:
    imgs_path = os.path.join(datasets_path,'inputs')
    targets_path = os.path.join(datasets_path,'targets') 
    imgs_list = os.listdir(imgs_path)
    targets_list = os.listdir(targets_path)
    imgs_list.sort()
    targets_list.sort()

    train_len = int(len(imgs_list) * p)
    # test_len = int(len(imgs_list) - train_len)

    random_seed = random.random()
    random.seed(random_seed)
    random.shuffle(imgs_list)
    random.shuffle(targets_list)
    
    train_imgs = imgs_list[:train_len]
    test_imgs = imgs_list[train_len:]
    train_targets = targets_list[:train_len]
    test_targets = targets_list[train_len:]

    train_save_path = f'./Data/Datasets/train/'
    test_save_path = f'./Data/Datasets/test/'
    rm_dir(train_save_path)
    rm_dir(test_save_path)

    for img in train_imgs:
        img_path = os.path.join(imgs_path,img)
        target_path = os.path.join(targets_path,img)
        img_save_path = os.path.join(train_save_path,'inputs',img)
        target_save_path = os.path.join(train_save_path,'targets',img)
        rm_dir(img_save_path)
        rm_dir(target_save_path)
        shutil.copytree(img_path,img_save_path,dirs_exist_ok=True)
        shutil.copytree(target_path,target_save_path,dirs_exist_ok=True)
    for img in test_imgs:
        img_path = os.path.join(imgs_path,img)
        target_path = os.path.join(targets_path,img)
        img_save_path = os.path.join(test_save_path,'inputs',img)
        target_save_path = os.path.join(test_save_path,'targets',img)
        rm_dir(img_save_path)
        rm_dir(target_save_path)
        shutil.copytree(img_path,img_save_path,dirs_exist_ok=True)
        shutil.copytree(target_path,target_save_path,dirs_exist_ok=True)

def rm_dir(dir_path):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)

def ToSequenceDataset(protopath):
    protopath = protopath
    dir_name = protopath.split('/')[-1]
    rm_dir(f'./Data/SequenceDataset/{dir_name}/imgs')
    rm_dir(f'./Data/SequenceDataset/{dir_name}/targets')
    rm_dir(f'./Data/SequenceDataset/{dir_name}/labels')
    input_list = os.listdir(os.path.join(protopath,'noise_imgs'))
    print(f'数据集长度：{len(input_list)}')
    for i in range(0, len(input_list)-9, 5):
        input_save_path = f'./Data/SequenceDataset/{dir_name}/imgs/{i+1}'
        target_save_path = f'./Data/SequenceDataset/{dir_name}/targets/{i+1}'
        label_save_path = f'./Data/SequenceDataset/{dir_name}/labels/{i+1}'
        rm_dir(input_save_path)
        rm_dir(target_save_path)
        rm_dir(label_save_path)
        for j in range(1,11):
            name = f'{i+j}.jpg'
            label = f'{i+j-1:0>4}.txt'
            input_path = os.path.join(protopath,'noise_imgs', name)
            target_path = os.path.join(protopath,'imgs',name)
            label_path = os.path.join(protopath, 'labels', label)
        
            shutil.copy(input_path,input_save_path)
            shutil.copy(target_path,target_save_path)
            shutil.copy(label_path, label_save_path)

def crop(proto_imgs_path,proto_labels_path,proto_targets_path,name):
    # 拟输入：
    # proto_imgs_path = r'SequenceDataset\B\imgs'
    # proto_labels_path = r'./SequenceDataset/B/labels'
    # proto_targets_path = r'./SequenceDataset/B/targets'
    # 获取要裁剪的图片与对应特征点位置与标签图片的地址

    rm_dir(f'./Data/CroppedDataset/{name}/imgs')
    rm_dir(f'./Data/CroppedDataset/{name}/labels')
    rm_dir(f'./Data/CroppedDataset/{name}/targets')
    imgs_path = proto_imgs_path #r'SequenceDataset\B\imgs'
    labels_path = proto_labels_path
    targets_path = proto_targets_path

    print(f'原始imgs路径：{imgs_path}')
    print(f'原始labels路径：{labels_path}')
    print(f'原始targets路径：{targets_path}')

    # 获取每一张图片的地址并排序
    imgs_sequence_list = [os.path.join(imgs_path, path) for path in os.listdir(imgs_path)] #[r'SequenceDataset\B\imgs\1']
    imgs_sequence_list.sort(key= lambda x :int(x.split('/')[-1]))

    labels_sequence_list = [os.path.join(labels_path,path) for path in os.listdir(labels_path)]
    labels_sequence_list.sort(key= lambda x :int(x.split('/')[-1]))

    targets_sequence_list = [os.path.join(targets_path, path) for path in os.listdir(targets_path)]
    targets_sequence_list.sort(key=lambda x: int(x.split('/')[-1]))

    for i in range(len(imgs_sequence_list)):
        file_name = imgs_sequence_list[i].split('/')[-1]
        # [r'SequenceDataset\B\imgs\1\1.ipg']
        imgs_list = [os.path.join(imgs_sequence_list[i],x) for x in os.listdir(imgs_sequence_list[i])]
        imgs_list.sort(key= lambda x :int(x.split('/')[-1][:-4]))
        labels_list = [os.path.join(labels_sequence_list[i],x) for x in os.listdir(labels_sequence_list[i])]
        labels_list.sort(key= lambda x :int(x.split('/')[-1][:-4]))
        targets_list = [os.path.join(targets_sequence_list[i],x) for x in os.listdir(targets_sequence_list[i])]
        targets_list.sort(key= lambda x :int(x.split('/')[-1][:-4]))


        # 设置裁剪后图片的保存路径并创建文件夹
        saveimgs_path = f'./Data/CroppedDataset/{name}/imgs/{file_name}'
        savelabels_path = f'./Data/CroppedDataset/{name}/labels/{file_name}'
        savetargets_path = f'./Data/CroppedDataset/{name}/targets/{file_name}'
        rm_dir(saveimgs_path)
        rm_dir(savelabels_path)
        rm_dir(savetargets_path)

        # print(f'裁剪后imgs路径：{saveimgs_path}')
        # print(f'裁剪后labels路径：{savelabels_path}')
        # print(f'裁剪后targets路径：{savetargets_path}')
        
        x_skewing = random.uniform(-5., 5.)
        y_skewing = random.uniform(-5., 5.)

        # 逐一裁剪图片并保存新图片和新label
        for i in range(len(imgs_list)):
            # 读取图片
            
            image = Image.open(imgs_list[i])
            target = Image.open(targets_list[i])
            

            with open(labels_list[i]) as f:

                # 读取标签内容
                text = f.readline().split(' ')
                ori_x = float(text[0])
                ori_y = float(text[1])
                ori_w = float(text[2])
                ori_h = float(text[3])

                # 随机偏移中心一定距离
                
                x_cut = int(ori_x + x_skewing)
                y_cut = int(ori_y + y_skewing)
                left = x_cut - 150
                right = x_cut + 150
                top = y_cut - 150
                bottom = y_cut + 150

                # 裁剪图片
                image = image.crop((left, top, right, bottom))
                target = target.crop((left, top, right, bottom))

                # 保留旧名称并保存图片
                new_name = imgs_list[i].split('/')[-1]
                new_lname = labels_list[i].split('/')[-1]
                new_tname = targets_list[i].split('/')[-1]
                image.save(os.path.join(saveimgs_path,new_name))
                target.save(os.path.join(savetargets_path,new_tname))

                # 计算新的label并保存
                new_x = ori_x - left
                new_y = ori_y - top
                new_w = 150
                new_h = 150
                txt_file = os.path.join(savelabels_path, new_lname)
                with open(txt_file, mode='w') as txt_file:
                    print(new_x, new_y, new_w, new_h, file=txt_file)

    print("————————图片裁剪完成————————\n")

def dataset_prepare(proto_input_path,proto_target_path,name,threshold):
    # 拟输入： 
    # proto_input_path = r'./CroppedDataset/B/imgs'
    # proto_target_path = r'./CroppedDataset/B/targets'
    # name = 'B'

    # 复制input到新文件夹
    proto_input_path = proto_input_path
    new_input_path = f'./Data/preDataset/inputs'
    print(f'开始将文件夹 {proto_input_path} 中的内容复制到文件夹 {new_input_path} 中')
    # rm_dir(new_input_path)
    for item in os.listdir(proto_input_path):
        s = os.path.join(proto_input_path, item)
        item = f'{name}_{item:0>4}'
        d = os.path.join(new_input_path, item)
        shutil.copytree(s, d)
    print("复制完成")

    # 对target进行处理并保存
    proto_target_path = proto_target_path # r'./CroppedDataset/B/targets'
    proto_target_path_list = os.listdir(proto_target_path) # [1,2,3]
    new_target_path = f'./Data/preDataset/targets' # r'Dataset/{name}/targets'
    print(f"开始处理 {proto_target_path} 中的图片并保存到 {new_target_path}")
    # rm_dir(new_target_path)

    for path in proto_target_path_list: 
        image_sequence_path = os.path.join(proto_target_path, path)# r'./CroppedDataset/B/targets/1'
        image_paths = os.listdir(image_sequence_path) # 1.jpg
        path = f'{name}_{path:0>4}'

        save_sequence = os.path.join(new_target_path, path)
        rm_dir(save_sequence)
        for img in image_paths:
            img_path = os.path.join(image_sequence_path, img)
            image = cv2.imread(img_path, 0)
            # 中值滤波
            blur = cv2.medianBlur(image, 5)
            threshold = threshold
            # 阈值分割,这里阈值用的是固定值200
            _, binary_image = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY)

            # 形态学操作去除孔洞和毛刺
            kernel = np.ones((7, 5), np.uint8)
            morphological_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)  # 闭运算

            # 图像保存

            save_path = os.path.join(new_target_path, path, img) # r'Dataset/{name}/targets/1'
            cv2.imwrite(save_path, morphological_image)
    print("处理完成")

def get_meanval(image_path = './Data/preDataset/inputs'):
    # 初始化累加器的值
    sum_pixel_values = 0  # 灰度图像的像素值总和  
    num_pixels = 0  # 总像素数量 
    # 获取每张图片的路径
    image_path = image_path
    imageseq_paths = [os.path.join(image_path, x) for x in os.listdir(image_path)]
    # 逐一读取图片
    for seqs in imageseq_paths:
        images = os.listdir(seqs)
        for img in images:
            img_path = os.path.join(seqs, img)
            image = Image.open(img_path).convert('L')
            image_array = np.array(image)
    # 累加像素值
            sum_pixel_values += image_array.sum()
    # 求出总像素数
            num_pixels += image_array.size
    meanval = sum_pixel_values / num_pixels
    print(f'该数据集的meanval为：{meanval:.4f}')
    return round(meanval, 4)
    
if __name__ == "__main__":
    # ToSequenceDataset('/home/chenjian/dataset/ProtoDataset/scan/B')
    # ToSequenceDataset('/home/chenjian/dataset/ProtoDataset/scan/L')
    # crop(proto_imgs_path=r'./Data/SequenceDataset/B/imgs',
    #     proto_labels_path=r'./Data/SequenceDataset/B/labels',
    #     proto_targets_path=r'./Data/SequenceDataset/B/targets',
    #     name='B')
    # crop(proto_imgs_path=r'./Data/SequenceDataset/L/imgs',
    #     proto_labels_path=r'./Data/SequenceDataset/L/labels',
    #     proto_targets_path=r'./Data/SequenceDataset/L/targets',
    #     name='L')
    # rm_dir('./Data/preDataset')
    # dataset_prepare(proto_input_path=r'./Data/CroppedDataset/B/imgs',
    #             proto_target_path=r'./Data/CroppedDataset/B/targets',
    #             name="B", threshold=200)
    # dataset_prepare(proto_input_path=r'./Data/CroppedDataset/L/imgs',
    #             proto_target_path=r'./Data/CroppedDataset/L/targets',
    #             name="L", threshold=200)
    # split_dataset(datasets_path='./Data/preDataset')
    # traintxt = './Data/train_seq.txt'
    # testtxt = './Data/test_seq.txt'
    # with open(traintxt,'w') as file_txt:
    #     train_seq_list = os.listdir('./Data/Datasets/train/inputs')
    #     train_seq_list.sort()
    #     for seq in train_seq_list:
    #         print(seq,file=file_txt)
    # with open(testtxt,'w') as file_txt:
    #     test_seq_list = os.listdir('./Data/Datasets/test/inputs')
    #     test_seq_list.sort()
    #     for seq in test_seq_list:
    #         print(seq,file=file_txt)
    data_set = LSE()
    sample = data_set[3]
    img = sample['image']
    print(img.shape)
    # print(gt.shape)

