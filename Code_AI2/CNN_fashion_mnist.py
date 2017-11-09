import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
from PIL import Image
from mnist_reader import load_mnist
import numpy as np
import time

BATCH_SIZE = 100
LEARNING_RATE = 0.01
CHANNEL = 1
EPOCH = 50

X_train,Y_train = load_mnist('/home/fq/fashion-mnist/data/fashion', kind='train')
X_test, Y_test = load_mnist('/home/fq/fashion-mnist/data/fashion', kind='t10k')

trainData = Data.TensorDataset(data_tensor=torch.from_numpy(X_train).type("torch.FloatTensor"),target_tensor=torch.from_numpy(Y_train))
testData = Data.TensorDataset(data_tensor=torch.from_numpy(X_test).type("torch.FloatTensor"),target_tensor=torch.from_numpy(Y_test)) 

trainLoader = Data.DataLoader(dataset=trainData, batch_size=BATCH_SIZE, shuffle=True)
testLoader = Data.DataLoader(dataset=testData, batch_size=BATCH_SIZE, shuffle=False)

# img1 = X_train[0,:-1].reshape(-1,28)
# img = Image.fromarray(img1)
# img.show()

# for i,(images, labels) in enumerate(trainLoader):
#     if labels[0] == 3:
#         image2 = images.view(64,1,28,28)
#         print(images.view(64,1,28,28).shape,image2[0])
#         break

# model
class CNNnet(torch.nn.Module):
    def __init__(self):
        super(CNNnet,self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1,16,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(16,32,3,1,1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(32,48,3,1,1),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.Full_layer = nn.Sequential(
            nn.Linear(48*3*3,128),
            nn.ReLU(),
            nn.Linear(128,10)
        )

    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        flatten = out.view(out.size(0),-1)
        out = self.Full_layer(flatten)
        return out


model = CNNnet()
print(model)

optimizer = torch.optim.Adam(model.parameters())
loss_func = nn.CrossEntropyLoss()

# training
start = time.time()
for epoch in range(EPOCH):
    print('epoch{}'.format(epoch+1))
    for batch_image, batch_label in trainLoader:
        batch_image, batch_label = Variable(batch_image.view(-1,CHANNEL,28,28)/255), Variable(batch_label)
        out = model(batch_image)
        loss = loss_func(out,batch_label)
        pred = torch.max(out,1)[1]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
end = time.time()
print('cost time is :',end-start)

# test
pre_num = 0
for batch_image, batch_label in testLoader:
    batch_image, batch_label = Variable(batch_image.view(-1,CHANNEL,28,28)/255), Variable(batch_label)
    outputs = model(batch_image)
    loss = loss_func(out,batch_label)
    predicted = torch.max(outputs,1)[1]  # [0]是预测概率值,[1]为对应的整数标签
    pre_num += (predicted == batch_label).sum()
# print('Accuracy is:{:.3f}'.format(100 * pre_num / len(trainData)))

# save the model
torch.save(model.state_dict(), 'CNN_fashion_mnist.pkl')


