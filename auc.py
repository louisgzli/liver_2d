#auc 计算
import codecs
from sklearn.metrics import roc_curve,auc
from matplotlib import pyplot as plt
def cal_auc(target=[],predict=[]):
    '''
    计算auc
    :param target:
    :param predict:
    :return:
    '''
    pos_prob = []
    neg_prob = []
    for index,i in enumerate(target):
        if i==1:
            pos_prob.append(predict[index])
        else:
            neg_prob.append(predict[index])
    legal_cnt = 0
    for pos_p in pos_prob:
        for neg_p in neg_prob:
            if pos_p>=neg_p:
                legal_cnt+=1
    return legal_cnt/(len(pos_prob)*len(neg_prob))

def plot_auc(target=[],predict=[]):
    '''
    绘制auc
    :param target:
    :param predict:
    :return:
    '''
    fpr, tpr, threshold = roc_curve(target, predict)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(10, 10))
    plt.plot(fpr, tpr, color='darkorange',
             lw=2, label='ROC curve (area = %0.2f)' % roc_auc)  ###假正率为横坐标，真正率为纵坐标做曲线
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('deep_learning based model ROC curve')
    plt.legend(loc="lower right")
    plt.show()

if __name__=="__main__":
    file = codecs.open("test_information_circle.txt",mode="r",encoding="utf-8")
    target = []
    predict = []
    for line in file:
        line = line.strip("\n")
        target.append(int(line.split(">>")[0]))
        predict.append(float(line.split(">>")[1]))


    print("auc: ",cal_auc(target,predict))
    plot_auc(target,predict)
    pass

