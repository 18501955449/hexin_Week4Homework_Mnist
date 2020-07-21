#coding:utf-8
import torch
from torchvision import datasets,transforms
from skimage.feature import local_binary_pattern
import torch.utils.data as Data
import numpy as np
import torch.nn as nn
def get_feature(x):
    '''提取LBP特征
    params:x为灰度图像
    return:x的LBP特征'''
    radius = 1  # LBP算法中范围半径的取值
    n_points = 8 * radius  # 领域像素点数
    xa = np.array(x)
    xt = torch.from_numpy(xa.reshape(28,28))
    def get_Lbp(x):
        lbp = local_binary_pattern(x, n_points, radius)
        #先转tensor
        lbp = torch.tensor(lbp)
        lbp = lbp/255.0
        feature = lbp.view(1, 28*28)
        return feature
    features = get_Lbp(xt)
    batch_feature = features.float()
    #print(feature)
    return batch_feature
def model(feature,weights0,weights1):
    '''三层前向传播结构
    params:每一层的参数'''
    feature = torch.cat((feature, torch.tensor(1.0).view(1, 1)), 1)
    h = feature.mm(weights0)
    h1 = torch.tanh(h).mm(weights1)
    #h2 = torch.tanh(h1).mm(weights2)
    y = torch.softmax(h1,1)
    return y
# def one_hot(gt):
#     gt_vector = torch.ones(1,10)
#     gt_vector *= 0*0.1
#     gt_vector[0,gt] = 1.0*0.9
#     return gt_vector

def get_acc(image_data,W0,W1):
    '''计算准确率
    params:image_data为所有数据
    W为权重参数
    returns:准确率'''
    correct = 0
    for image,label in image_data:
        feature = get_feature(image)
        y = model(feature,W0,W1)
        pred = torch.argmin((torch.abs(y-1))).item()
        # print("图像[%s]得分类结果是:[%s]"%(gt,pred))
        if label == pred:
            correct += 1
    return float(correct / float(len(image_data)))


def train_model(train_image_data,test_image_data, weights0, weights1,lr):
    criterion = nn.CrossEntropyLoss()
    for epoch in range(0, 100):
        loss_value = 0
        for image_data,image_label in train_image_data:
            feature = get_feature(image_data)
            y = model(feature, weights0, weights1)
            gt = image_label
            loss = criterion(y, gt)
            loss_value += loss.data.item()
            loss.backward()
            weights0.data.sub_(weights0.grad.data * lr)
            weights0.grad.data.zero_()
            weights1.data.sub_(weights1.grad.data * lr)
            weights1.grad.data.zero_()
            # weights2.data.sub_(weights2.grad.data * lr)
            # weights2.grad.data.zero_()
        loss_value = loss_value/len(train_image_data)
        train_acc = get_acc(train_image_data,weights0,weights1)
        test_acc = get_acc(test_image_data,weights0,weights1)
        print("epoch=%s,loss=%s,train/test_acc:%s/%s" % (epoch, loss_value, train_acc, test_acc))
    return weights0, weights1
if __name__ == "__main__":
    #初始化权重
    weights0 = torch.randn(785, 35, requires_grad=True)
    weights1 = torch.randn(35, 10, requires_grad=True)
    #weights2 = torch.randn(35, 10, requires_grad=True)
    #加载MNIST数据
    batch_size = 1
    mnist_transforms = transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.131], [0.308])])
    #训练数据集
    train_dataset = datasets.MNIST(root='./data/', train=True, transform=mnist_transforms, download=False)
    train_loader = Data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    # 测试数据集 train=False
    data_test = datasets.MNIST('./data',train=False,transform=mnist_transforms,download=False)
    test_loader = Data.DataLoader(data_test,batch_size=batch_size,shuffle=False)
    train_model(train_loader,test_loader, weights0, weights1,0.01)
