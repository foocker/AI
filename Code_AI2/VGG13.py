import torch
import numpy as np
import torch.nn as nn
import torch.utils.data as Data
from torch.autograd import Variable

N_CHANNEL = 3
EPOCH = 10
BATCH_SIZE = 4

# creat unreal data
X_train, Y_train = torch.rand(10,3,224,224), torch.from_numpy(np.random.randint(10,size=10))

# print(type(X_train))

trainData = Data.TensorDataset(data_tensor=X_train,target_tensor=Y_train)
trainLoader = Data.DataLoader(dataset=trainData,batch_size=BATCH_SIZE,shuffle=True)

class VGG13(nn.Module):
    def __init__(self):
        super(VGG13, self).__init__()

        '''
        vgg13 input(224*224 RGB image)
        '''

        self.layer1 = nn.Sequential(
            # 1-1 conv layer
            nn.Conv2d(N_CHANNEL,64,3,1,1),
            nn.ReLU(),
            # 1-2 conv layer
            nn.Conv2d(64,64,3,1,1),
            nn.ReLU(),
            # 1 pooling
            nn.MaxPool2d(2)
        )

        self.layer2 = nn.Sequential(
            # 2-1 conv layer
            nn.Conv2d(64,128,3,1,1),
            nn.ReLU(),
            # 2-2 conv layer
            nn.Conv2d(128,128,3,1,1),
            nn.ReLU(),
            # 2 pooling
            nn.MaxPool2d(2)
        )

        self.layer3 = nn.Sequential(
            # 3-1 conv layer
            nn.Conv2d(128,256,3,1,1),
            nn.ReLU(),
            # 3-2 conv layer
            nn.Conv2d(256,256,3,1,1),
            nn.ReLU(),
            # 3 pooling
            nn.MaxPool2d(2)
        )

        self.layer4 = nn.Sequential(
            # 4-1 conv layer
            nn.Conv2d(256,512,3,1,1),
            nn.ReLU(),
            # 4-2 conv layer
            nn.Conv2d(512,512,3,1,1),
            nn.ReLU(),
            # 4 pooling
            nn.MaxPool2d(2)
        )

        self.FC_layer1 = nn.Sequential(
            nn.Linear(512*14*14,4096),
            nn.ReLU()
        )

        self.FC_layer2 = nn.Sequential(
            nn.Linear(4096,4096),
            nn.ReLU()
        )

        self.FC_layer3 = nn.Sequential(
            nn.Linear(4096,10),
            nn.Softmax()
        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        vgg13_features = out.view(out.size(0),-1)
        out = self.FC_layer1(vgg13_features)
        out = self.FC_layer2(out)
        out = self.FC_layer3(out)
        return out

vgg13 = VGG13()
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(vgg13.parameters(), lr=1e-3)

for epoch in range(EPOCH):
    for batch_x, batch_y in trainLoader:
        batch_x, batch_y = Variable(batch_x), Variable(batch_y)
        print(batch_x.data.shape,batch_y.data.shape)
        # optimizer to zero
        optimizer.zero_grad()
        outputs = vgg13(batch_x)
        print(outputs.data.shape)
        loss = loss_func(outputs,batch_y)
        loss.backward()
        optimizer.step()









        

