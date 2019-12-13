import torch as t
import numpy as np
import os
import time
import shutil
from config_2d import opt
from torchvision import transforms
from utils.tools import RandomFlip
import codecs
def gaussian_normalize(image_array):
    MEAN = np.mean(image_array)
    STD = np.std(image_array)
    image_result = (image_array-MEAN)/STD
    return image_result
# normal lize data to 0~1
def normalize(image_array):
    MAX_NUM = np.max(image_array)
    MIN_NUM = np.min(image_array)
    image_result = (image_array-MIN_NUM)/(MAX_NUM-MIN_NUM)
    return image_result
def flow_train_test(src_root="",train_target_root="",test_target_root = "",train_head_list=[],test_head_list=[]):
    if os.path.exists(train_target_root):
        shutil.rmtree(train_target_root)
        os.mkdir(train_target_root)
    elif not os.path.exists(train_target_root):
        os.mkdir(train_target_root)
    if os.path.exists(test_target_root):
        shutil.rmtree(test_target_root)
        os.mkdir(test_target_root)
    elif not os.path.exists(test_target_root):
        os.mkdir(test_target_root)

    all_file_names = os.listdir(src_root)
    for file in all_file_names:
        if file.split("_")[0] in train_head_list:
            shutil.copyfile(os.path.join(src_root,file),os.path.join(train_target_root,file))
        elif file.split("_")[0] in test_head_list:
            shutil.copyfile(os.path.join(src_root,file),os.path.join(test_target_root,file))
        else:
            print("miss target file, filename:",file)
def read_train_test():
    train_test_file  = codecs.open(opt.train_test_file,mode="r",encoding="utf8")
    file_array = train_test_file.readlines()
    train_list = file_array[1].strip().strip("[]").split(",")
    train_list = [x.strip().strip("'") for x in train_list]
    test_list = file_array[3].strip().strip("[]").split(",")
    test_list = [x.strip().strip("'") for x in test_list]
    return train_list,test_list

class dataset_2d():
    def __init__(self, data_path, train=True,shuffle=True,seed=0):

        list_filenames = os.listdir(data_path)
        self.train = train
        self.list_filenames = list_filenames
        self.circle_root_path = data_path
        self.user_dict = {}
        self.user_img_count = {}
        self.train_data_list = []
        self.test_data_list = []
        self.train_pos_neg_ratio = []
        self.test_pos_neg_ratio = []
        self.train_trans = transforms.Compose([
            # transforms.Resize(768),
            transforms.RandomHorizontalFlip(),
            # transforms.RandomCrop(512),
            ])
        self.test_trans = transforms.Compose([
            # transforms.Resize(768),
            # transforms.CenterCrop(512),
        ])


        for file_id in list_filenames:
            file_head = file_id.split("_")[0]
            file_category = int(file_id.split(".")[0].split("_")[-1])
            if file_head not in self.user_dict:
                self.user_dict[file_head] = file_category
                self.user_img_count[file_head]=0
            else:
                self.user_img_count[file_head] +=1

        self.all_file_head = list(self.user_dict.keys())

        # np.random.seed(seed)
        if True:
            np.random.shuffle(self.all_file_head)



        self.train_head_list = self.all_file_head[0:int(len(self.all_file_head)*opt.train_ratio)]
        self.test_head_list = self.all_file_head[int(len(self.all_file_head)*opt.train_ratio):]
        # self.train_head_list,self.test_head_list = read_train_test()


        #record pos_train expample
        train_pos_cnt = 0
        train_neg_cnt = 0
        #record test pos_neg example
        test_pos_cnt = 0
        test_neg_cnt = 0
        for file_id in list_filenames:
            file_head = file_id.split("_")[0]
            file_category = int(file_id.split(".")[0].split("_")[-1])
            if file_head in self.train_head_list:
                self.train_data_list.append(file_id)
                if file_category==0:
                    train_neg_cnt+=1
                elif file_category==1:
                    train_pos_cnt+=1

            elif file_head in self.test_head_list:
                self.test_data_list.append(file_id)
                if file_category==0:
                    test_neg_cnt+=1
                elif file_category==1:
                    test_pos_cnt+=1
        # move files into train and test files
        # flow_train_test(src_root=opt.backup_data,train_target_root=opt.train_data_root,test_target_root = opt.test_data_root,
        #                 train_head_list =self.train_head_list,test_head_list = self.test_head_list)
        # if self.train:
        if True:
            print("start to record train test information")
            #write seg.txt
            opt.seg_result.writelines(str(time.asctime( time.localtime(time.time())))+"\n")
            opt.seg_result.flush()
            opt.seg_result.writelines("train_head_list\n")
            opt.seg_result.writelines(str(self.train_head_list)+"\n")
            opt.seg_result.writelines("test_head_list\n")
            opt.seg_result.writelines(str(self.test_head_list)+"\n")
            opt.seg_result.flush()
            opt.seg_result.writelines(str("train_pos_cnt: "+str(train_pos_cnt)+" train_neg_cnt: "+str(train_neg_cnt)+
                               " test_pos_cnt: "+str(test_pos_cnt)+" test_neg_cnt: "+str(test_neg_cnt)+"\n"))
            opt.seg_result.flush()

            print("recording train test information finished")
    def __getitem__(self, item):#item is index

        if self.train:
            image_path = os.path.join(self.circle_root_path,self.train_data_list[item])
            image = np.load(image_path)
            image = image.reshape(1, 512, 512)
            # image = t.from_numpy(image).unsqueeze(0).float()
            image = RandomFlip(image)
            image = t.from_numpy(image).float()
            cur_file_id = self.train_data_list[item].split(".")[0].split("_")[0]
            label = t.tensor(int(self.train_data_list[item].split(".")[0].split("_")[-1]))
        else:
            image_path = os.path.join(self.circle_root_path, self.test_data_list[item])
            image = np.load(image_path)
            image = image.reshape(1,512,512)
            # image = gaussian_normalize(image)
            image = t.from_numpy(image).float()
            label = t.tensor(int(self.test_data_list[item].split(".")[0].split("_")[-1]))
            cur_file_id = self.test_data_list[item].split(".")[0].split("_")[0]

        return image, label,cur_file_id


    def __len__(self):
        if self.train:
            return len(self.train_data_list)
        else:
            return len(self.test_data_list)


if __name__ == "__main__":
    root_path = "/raid/lgz/data/liver/single_circle_nz"
    dataset = dataset_2d(root_path,train=True)
    temp_file = dataset.__getitem__(200)[0]
    # read_train_test()
    pass







