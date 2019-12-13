# coding:utf8
from config_2d import opt
import os
import torch as t
import models
from dataset_2d import dataset_2d
from torch.utils.data import DataLoader
from torchnet import meter
from utils.visualize import Visualizer
from tqdm import tqdm
import numpy as np
@t.no_grad()  # pytorch>=0.5
def test(**kwargs):
    opt._parse(kwargs)
    confusion_matrix = meter.ConfusionMeter(2)
    # configure model
    model = getattr(models, opt.model)().eval()
    if opt.load_model_path:
        model.load(opt.load_model_path)
    model.eval()
    # model = t.nn.DataParallel(model, device_ids=opt.device_ids)  # 声明所有可用设备
    model.to(opt.device)  # 模型放在主设备
    # data
    test_data = dataset_2d(opt.backup_data, train=False)
    test_dataloader = DataLoader(test_data, batch_size=opt.test_batch_size, shuffle=False, num_workers=opt.num_workers)

    results = []
    user_pos_count = {}
    user_prob_rec = {}
    user_img_count = test_dataloader.dataset.user_img_count

    for ii, (data, label, item_id) in tqdm(enumerate(test_dataloader)):
        input = data.to(opt.device)  # 数据放在主设备
        label = label.to(opt.device)  # 数据放在主设备
        score = model(input)
        # update user_pos_count
        item_predict = t.argmax(score, dim=1)
        for i in range(item_predict.shape[0]):
            if item_id[i] in user_pos_count:
                if item_predict[0] == 1:
                    user_pos_count[item_id[i]] += 1
                    user_prob_rec[item_id[i]] += t.max(t.softmax(score, dim=1))
            else:
                user_pos_count[item_id[0]] = 0
                user_prob_rec[item_id[0]] = 0

        confusion_matrix.add(score.detach(), label.detach())

    global_right_num = 0
    for user in test_dataloader.dataset.test_head_list:
        pos_cnt = user_pos_count[user]
        all_cnt = user_img_count[user]
        flag = 0
        if pos_cnt >= all_cnt / 2:
            flag = 1
        if flag == test_dataloader.dataset.user_dict[user]:
            global_right_num += 1
        probabilities = (user_prob_rec[user] / user_img_count[user]).item()
        opt.test_result.writelines(
            str(test_dataloader.dataset.user_dict[user]) + ">>" + str(probabilities) + ">>" + str(
                user_img_count[item_id[0]]) + ">>" + str(
                user_pos_count[item_id[0]]) + "\n")

        opt.test_result.flush()

        # batch_results = [(label.item(), probability) for label, probability in zip(label, probabilities)]
        # results += batch_results
    # write_csv(results, opt.result_file)
    cm_value = confusion_matrix.value()
    accuracy = 100. * (cm_value[0][0] + cm_value[1][1]) / (cm_value.sum())
    print("accuracy", accuracy)
    return results

    #   copy from val
    model = getattr(models, opt.model)().eval()
    if opt.load_model_path:
        model.load(opt.load_model_path)
    model.to(opt.device)
    model.eval()
    train_data = dataset_2d(opt.test_data_root, train=False)
    test_dataloader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)
    #
    confusion_matrix = meter.ConfusionMeter(2)
    for ii, (val_input, label) in tqdm(enumerate(test_dataloader)):
        val_input = val_input.to(opt.device)
        label = label.to(opt.device)
        score = model(val_input)

        confusion_matrix.add(score.detach(), label)

    model.train()
    cm_value = confusion_matrix.value()
    accuracy = 100. * (cm_value[0][0] + cm_value[1][1]) / (cm_value.sum())
    print("accuracy:", accuracy)





def write_csv(results, file_name):
    import csv
    with open(file_name, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'label'])
        writer.writerows(results)


def train(**kwargs):

    opt._parse(kwargs)
    vis = Visualizer(opt.env, port=opt.vis_port)

    # step1: configure model
    model = getattr(models, opt.model)()
    if opt.load_model_path:
        model.load(opt.load_model_path)

    model.to(opt.device)#数据放在主设备

    # step2: data
    train_data = dataset_2d(opt.backup_data, train=True)
    user_dict = train_data.user_dict
    user_img_count = train_data.user_img_count
    val_data = dataset_2d(opt.backup_data, train=False)
    train_dataloader = DataLoader(train_data, opt.batch_size, shuffle=True, num_workers=opt.num_workers)
    val_dataloader = DataLoader(val_data, opt.batch_size, shuffle=True, num_workers=opt.num_workers)

    # step3: criterion and optimizer
    criterion = t.nn.CrossEntropyLoss()
    lr = opt.lr
    optimizer = model.get_optimizer(lr, opt.weight_decay)

    # step4: meters
    loss_meter = meter.AverageValueMeter()
    confusion_matrix = meter.ConfusionMeter(2)
    previous_loss = 1e10
    # model = t.nn.DataParallel(model, device_ids=opt.device_ids)  # 声明所有可用设备
    # train
    for epoch in range(opt.max_epoch):

        loss_meter.reset()
        confusion_matrix.reset()

        for ii, (data, label,_) in tqdm(enumerate(train_dataloader)):
            if ii>5:
                break

            # train model
            input = data.to(opt.device)
            target = label.to(opt.device)

            optimizer.zero_grad()
            score = model(input)
            loss = criterion(score, target)
            loss.backward()
            optimizer.step()

            # meters update and visualize
            loss_meter.add(loss.item())
            # detach 一下更安全保险
            confusion_matrix.add(score.detach(), target.detach())

            if (ii + 1) % opt.print_freq == 0:
                vis.plot('loss', loss_meter.value()[0])

                # 进入debug模式
                if os.path.exists(opt.debug_file):
                    import ipdb
                    ipdb.set_trace()



        # validate and visualize
        val_cm, val_accuracy,global_right_acc = val(model, val_dataloader,user_dict,user_img_count)

        vis.plot('val_accuracy', val_accuracy)
        vis.plot("global_acc",global_right_acc)
        vis.log("epoch:{epoch},lr:{lr},loss:{loss},train_cm:{train_cm},val_cm:{val_cm}".format(
            epoch=epoch, loss=loss_meter.value()[0], val_cm=str(val_cm.value()), train_cm=str(confusion_matrix.value()),
            lr=lr))

        # update learning rate
        if loss_meter.value()[0] > previous_loss:
            lr = lr * opt.lr_decay
            # 第二种降低学习率的方法:不会有moment等信息的丢失
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        previous_loss = loss_meter.value()[0]
    #save model
    model.save()
    # opt.seg_result.close()

@t.no_grad()
def val(model, dataloader,user_dict,user_img_count):
    """
    计算模型在验证集上的准确率等信息
    user_dict:store user categary
    user_img_count user img numbers
    """
    model.eval()
    user_pos_count = {}

    confusion_matrix = meter.ConfusionMeter(2)
    for ii, (val_input, label,item_id) in tqdm(enumerate(dataloader)):
        val_input = val_input.to(opt.device)
        label = label.to(opt.device)
        score = model(val_input)

        #update user_pos_count
        item_predict = t.argmax(score,dim=1)
        for i in range(item_predict.shape[0]):
            if item_id[i] in user_pos_count:
                if item_predict[i]==1:
                    user_pos_count[item_id[i]]+=1
            else:
                user_pos_count[item_id[i]] = 0
        confusion_matrix.add(score.detach(), label)

    model.train()
    cm_value = confusion_matrix.value()
    accuracy = 100. * (cm_value[0][0] + cm_value[1][1]) / (cm_value.sum())
    # write result to seg.txt


    #calculate global accuracy
    global_right_num = 0
    for user in dataloader.dataset.test_head_list:
        pos_cnt = user_pos_count[user]
        all_cnt = user_img_count[user]
        flag = 0
        if pos_cnt>=all_cnt/2:
            flag = 1
        if flag == user_dict[user]:
            global_right_num+=1
    global_right_acc = 100*global_right_num/len(dataloader.dataset.test_head_list)

    opt.seg_result.writelines(str(accuracy)+">>>"+str(global_right_acc))
    opt.seg_result.writelines(" ")
    opt.seg_result.flush()
    return confusion_matrix, accuracy,global_right_acc


def help():
    """
    打印帮助的信息： python file.py help
    """

    print("""
    usage : python file.py <function> [--args=value]
    <function> := train | test | help
    example: 
            python {0} train --env='env0701' --lr=0.01
            python {0} test --dataset='path/to/dataset/root/'
            python {0} help
    avaiable args:""".format(__file__))

    from inspect import getsource
    source = (getsource(opt.__class__))
    print(source)


if __name__ == '__main__':
    import fire

    fire.Fire()
    # for i in range(20):
    #     train()

    opt.seg_result.writelines("\n")
    opt.test_result.close()
    opt.seg_result.close()