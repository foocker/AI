import torch
import torch.nn as nn
import torch.utils.data as Data
from mnist_reader import load_mnist
from torch.autograd import Variable

# Hyper parameters
input_size = 784
num_classes = 10
epochs = 100
batch_size = 100
learning_rate = 0.001

# fashion_mnist dataset
X_train,Y_train = load_mnist('/home/fq/fashion-mnist/data/fashion', kind='train')
X_test, Y_test = load_mnist('/home/fq/fashion-mnist/data/fashion', kind='t10k')

trainData = Data.TensorDataset(data_tensor=torch.from_numpy(X_train).type("torch.FloatTensor"),target_tensor=torch.from_numpy(Y_train))
testData = Data.TensorDataset(data_tensor=torch.from_numpy(X_test).type("torch.FloatTensor"),target_tensor=torch.from_numpy(Y_test))

trainLoader = Data.DataLoader(dataset=trainData, batch_size=batch_size, shuffle=True)
testLoader = Data.DataLoader(dataset=testData, batch_size=batch_size, shuffle=False)

# model
class LogisticRegression(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)
    
    def forward(self, x):
        out = self.linear(x)
        return out

model = LogisticRegression(input_size, num_classes)

loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# training
for epoch in range(epochs):
    for i, (images, labels) in enumerate(trainLoader):
        images = Variable(images.view(-1,28*28))
        labels = Variable(labels)

        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_func(outputs, labels)
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print('Epoch:{},Loss:{:.4f}'.format(epoch+1,loss.data[0]))

# test
pre_num = 0
total_num = 0
for images, labels in testLoader:
    images = Variable(images.view(-1,28*28))
    outputs = model(images)
    _, predicted = torch.max(outputs.data, 1)
    total_num += labels.size(0)
    pre_num += (predicted == labels).sum()

print('Accuracy is:{:.3f}'.format(100 * pre_num / total_num))

# save the model
# torch.save(model,'LR_fashion_model')  # 保存整个网络
torch.save(model.state_dict(), 'LR_fashion_mnist.pkl')

# 调用保存的模型参数，在新数据上预测
# def restore_params():
#     re_model = nn.Sequential(
#         nn.Linear(784,10)
#         )
#     re_model.load_state_dict('LR_fashion_mnist.pkl')
#     out = re_model(x)
#     _, predicted = torch.max(out.data, 1)
#     return predicted
    



