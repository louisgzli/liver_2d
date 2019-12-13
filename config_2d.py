import torch as t
import warnings
import codecs
class DefalutConfig(object):
    env = 'default'  # visdom 环境
    vis_port = 8097  # visdom 端口
    model = 'ResNet34'  # 使用的模型，名字必须与models/__init__.py中的名字一致
    device_ids = [13]
    # data_dir = '/raid/lgz/data/liver/goal_temp_npy_normalize/' #指定训练集合路径
    backup_data = "/raid/lgz/data/liver/single_nocircle_nz_all"
    train_data_root = "/raid/lgz/data/liver/single_nocircle_nz_all"
    model_dir = 'test_128_32_3D_res_pic.pb' #模型保存路径和名
    test_data_root = '/raid/lgz/data/liver/single_nocircle_nz_all'  # 测试集存放路径
    load_model_path = None  # 加载预训练的模型的路径，为None代表不加载
    # 字
    train_label_dir = '/raid/lgz/data/liver/goal.csv' #训练样本标签
    lr = 1e-3
    batch_size = 64

    test_batch_size = 1
    num_workers = 1
    n_show_step = 1
    print_freq = 1
    lr_decay = 0.5  # when val_loss increase, lr = lr*lr_decay
    weight_decay = 1e-5  # zhengzexiangxishu
    device = t.device('cuda:0' if t.cuda.is_available() else 'cpu')
    max_epoch = 55
    debug_file = '/tmp/d00ebug'  # if os.path.exists(debug_file): enter ipdb
    result_file = 'result.csv'
    train_ratio = 0.8 #train:all split ratio
    seg_result= codecs.open("/home/lgz/code/liver_2d/single_nocircle_ct_all_true_random.txt","a",encoding="utf8")
    test_result = codecs.open("/home/lgz/code/liver_2d/test_information_circle.txt","a",encoding='utf8')
    checkpoints_path = "/raid/lgz/data/liver/checkpoints/circle/"
    train_test_file = "/home/lgz/code/liver_2d/train_test_file.txt"
    train_loss = codecs.open("/home/lgz/code/liver_2d/train_loss.txt","a",encoding='utf8')
    train_accuracy="/home/lgz/code/liver_2d/train_acc.txt"
    loss_confusion_file = codecs.open("/home/lgz/code/liver_2d/loss_confusion_file.txt","a",encoding = "utf8")
    def _parse(self, kwargs):
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning: opt has not attribute %s" % k)
            setattr(self, k, v)


        print("user config:")
        for k, v in self.__class__.__dict__.items():
            if not k.startswith("_"):
                print(k, getattr(self, k))





opt = DefalutConfig()