# #
# # import torch
# # import torchvision
# # from tqdm import tqdm
# # import matplotlib.pyplot as plt
# #
# #
# # # By: Elwin https://editor.csdn.net/md?not_checkout=1&articleId=112980305
# #
# # class Net(torch.nn.Module):
# #     def __init__(self):
# #         super(Net, self).__init__()
# #         self.model = torch.nn.Sequential(
# #             # The size of the picture is 28x28
# #             torch.nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
# #             torch.nn.ReLU(),
# #             torch.nn.MaxPool2d(kernel_size=2, stride=2),
# #
# #             # The size of the picture is 14x14
# #             torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
# #             torch.nn.ReLU(),
# #             torch.nn.MaxPool2d(kernel_size=2, stride=2),
# #
# #             # The size of the picture is 7x7
# #             torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
# #             torch.nn.ReLU(),
# #
# #             torch.nn.Flatten(),
# #             torch.nn.Linear(in_features=7 * 7 * 64, out_features=128),
# #             torch.nn.ReLU(),
# #             torch.nn.Linear(in_features=128, out_features=10),
# #             torch.nn.Softmax(dim=1)
# #         )
# #
# #     def forward(self, input):
# #         output = self.model(input)
# #         return output
# #
# #
# # device = "cuda:0" if torch.cuda.is_available() else "cpu"
# # transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
# #                                             torchvision.transforms.Normalize(mean=[0.5], std=[0.5])])
# #
# # BATCH_SIZE = 256
# # EPOCHS = 2
# # trainData = torchvision.datasets.MNIST('./data/', train=True, transform=transform, download=True)
# # testData = torchvision.datasets.MNIST('./data/', train=False, transform=transform)
# #
# # trainDataLoader = torch.utils.data.DataLoader(dataset=trainData, batch_size=BATCH_SIZE, shuffle=True)
# # testDataLoader = torch.utils.data.DataLoader(dataset=testData, batch_size=BATCH_SIZE)
# # net = Net()
# # print(net.to(device))
# #
# # lossF = torch.nn.CrossEntropyLoss()
# # optimizer = torch.optim.Adam(net.parameters())
# #
# # history = {'Test Loss': [], 'Test Accuracy': []}
# # for epoch in range(1, EPOCHS + 1):
# #     processBar = tqdm(trainDataLoader, unit='step')
# #     net.train(True)
# #     for step, (trainImgs, labels) in enumerate(processBar):
# #         trainImgs = trainImgs.to(device)
# #         labels = labels.to(device)
# #
# #         net.zero_grad()
# #         outputs = net(trainImgs)
# #         loss = lossF(outputs, labels)
# #         predictions = torch.argmax(outputs, dim=1)
# #         accuracy = torch.sum(predictions == labels) / labels.shape[0]
# #         loss.backward()
# #
# #         optimizer.step()
# #         processBar.set_description("[%d/%d] Loss: %.4f, Acc: %.4f" %
# #                                    (epoch, EPOCHS, loss.item(), accuracy.item()))
# #
# #         if step == len(processBar) - 1:
# #             correct, totalLoss = 0, 0
# #             net.train(False)
# #             with torch.no_grad():
# #                 for testImgs, labels in testDataLoader:
# #                     testImgs = testImgs.to(device)
# #                     labels = labels.to(device)
# #                     outputs = net(testImgs)
# #                     loss = lossF(outputs, labels)
# #                     predictions = torch.argmax(outputs, dim=1)
# #
# #                     totalLoss += loss
# #                     correct += torch.sum(predictions == labels)
# #
# #                     testAccuracy = correct / (BATCH_SIZE * len(testDataLoader))
# #                     testLoss = totalLoss / len(testDataLoader)
# #                     history['Test Loss'].append(testLoss.item())
# #                     history['Test Accuracy'].append(testAccuracy.item())
# #
# #             processBar.set_description("[%d/%d] Loss: %.4f, Acc: %.4f, Test Loss: %.4f, Test Acc: %.4f" %(epoch, EPOCHS, loss.item(), accuracy.item(), testLoss.item(), testAccuracy.item()))
# #     processBar.close()
# #
# # plt.plot(history['Test Loss'], label='Test Loss')
# # plt.legend(loc='best')
# # plt.grid(True)
# # plt.xlabel('Epoch')
# # plt.ylabel('Loss')
# # plt.show()
# #
# # plt.plot(history['Test Accuracy'], color='red', label='Test Accuracy')
# # plt.legend(loc='best')
# # plt.grid(True)
# # plt.xlabel('Epoch')
# # plt.ylabel('Accuracy')
# # plt.show()
# #
# # torch.save(net, './model.pth')
# #
# import torch
# import torchvision
# from torch.utils.data import DataLoader
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# import matplotlib.pyplot as plt
#
# n_epochs = 3
# batch_size_train = 64
# batch_size_test = 1000
# learning_rate = 0.01
# momentum = 0.5
# log_interval = 10
# random_seed = 1
# torch.manual_seed(random_seed)
#
# train_loader = torch.utils.data.DataLoader(
#     torchvision.datasets.MNIST('./data/', train=True, download=True,
#                                transform=torchvision.transforms.Compose([
#                                    torchvision.transforms.ToTensor(),
#                                    torchvision.transforms.Normalize(
#                                        (0.1307,), (0.3081,))
#                                ])),
#     batch_size=batch_size_train, shuffle=True)
# test_loader = torch.utils.data.DataLoader(
#     torchvision.datasets.MNIST('./data/', train=False, download=True,
#                                transform=torchvision.transforms.Compose([
#                                    torchvision.transforms.ToTensor(),
#                                    torchvision.transforms.Normalize(
#                                        (0.1307,), (0.3081,))
#                                ])),
#     batch_size=batch_size_test, shuffle=True)
#
# examples = enumerate(test_loader)
# batch_idx, (example_data, example_targets) = next(examples)
# # print(example_targets)
# # print(example_data.shape)
#
# fig = plt.figure()
# for i in range(6):
#     plt.subplot(2, 3, i + 1)
#     plt.tight_layout()
#     plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
#     plt.title("Ground Truth: {}".format(example_targets[i]))
#     plt.xticks([])
#     plt.yticks([])
# plt.show()
#
#
# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
#         self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
#         self.conv2_drop = nn.Dropout2d()
#         self.fc1 = nn.Linear(320, 50)
#         self.fc2 = nn.Linear(50, 10)
#
#     def forward(self, x):
#         x = F.relu(F.max_pool2d(self.conv1(x), 2))
#         x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
#         x = x.view(-1, 320)
#         x = F.relu(self.fc1(x))
#         x = F.dropout(x, training=self.training)
#         x = self.fc2(x)
#         return F.log_softmax(x, dim=1)
#
#
# network = Net()
# optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)
#
# train_losses = []
# train_counter = []
# test_losses = []
# test_counter = [i * len(train_loader.dataset) for i in range(n_epochs + 1)]
#
#
# def train(epoch):
#     network.train()
#     for batch_idx, (data, target) in enumerate(train_loader):
#         optimizer.zero_grad()
#         output = network(data)
#         loss = F.nll_loss(output, target)
#         loss.backward()
#         optimizer.step()
#         if batch_idx % log_interval == 0:
#             print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data),
#                                                                            len(train_loader.dataset),
#                                                                            100. * batch_idx / len(train_loader),
#                                                                            loss.item()))
#             train_losses.append(loss.item())
#             train_counter.append((batch_idx * 64) + ((epoch - 1) * len(train_loader.dataset)))
#             torch.save(network.state_dict(), './model.pth')
#             torch.save(optimizer.state_dict(), './optimizer.pth')
#
#
# def test():
#     network.eval()
#     test_loss = 0
#     correct = 0
#     with torch.no_grad():
#         for data, target in test_loader:
#             output = network(data)
#             test_loss += F.nll_loss(output, target, reduction='sum').item()
#             pred = output.data.max(1, keepdim=True)[1]
#             correct += pred.eq(target.data.view_as(pred)).sum()
#     test_loss /= len(test_loader.dataset)
#     test_losses.append(test_loss)
#     print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
#         test_loss, correct, len(test_loader.dataset),
#         100. * correct / len(test_loader.dataset)))
#
#
# train(1)
#
# test()  # 不加这个，后面画图就会报错：x and y must be the same size
# for epoch in range(1, n_epochs + 1):
#     train(epoch)
#     test()
#
# fig = plt.figure()
# plt.plot(train_counter, train_losses, color='blue')
# plt.scatter(test_counter, test_losses, color='red')
# plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
# plt.xlabel('number of training examples seen')
# plt.ylabel('negative log likelihood loss')
#
# examples = enumerate(test_loader)
# batch_idx, (example_data, example_targets) = next(examples)
# with torch.no_grad():
#     output = network(example_data)
# fig = plt.figure()
# for i in range(6):
#     plt.subplot(2, 3, i + 1)
#     plt.tight_layout()
#     plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
#     plt.title("Prediction: {}".format(output.data.max(1, keepdim=True)[1][i].item()))
#     plt.xticks([])
#     plt.yticks([])
# plt.show()
#
# # ----------------------------------------------------------- #
#
# continued_network = Net()
# continued_optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)
#
# network_state_dict = torch.load('model.pth')
# continued_network.load_state_dict(network_state_dict)
# optimizer_state_dict = torch.load('optimizer.pth')
# continued_optimizer.load_state_dict(optimizer_state_dict)
#
# # 注意不要注释前面的“for epoch in range(1, n_epochs + 1):”部分，
# # 不然报错：x and y must be the same size
# # 为什么是“4”开始呢，因为n_epochs=3，上面用了[1, n_epochs + 1)
# for i in range(4, 9):
#     test_counter.append(i * len(train_loader.dataset))
#     train(i)
#     test()
#
# fig = plt.figure()
# plt.plot(train_counter, train_losses, color='blue')
# plt.scatter(test_counter, test_losses, color='red')
# plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
# plt.xlabel('number of training examples seen')
# plt.ylabel('negative log likelihood loss')
# plt.show()
import torch
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
import torch.nn.functional as F


batch_size = 64
learning_rate = 0.01
momentum = 0.5
EPOCH = 10

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
# softmax归一化指数函数(https://blog.csdn.net/lz_peter/article/details/84574716),其中0.1307是mean均值和0.3081是std标准差

train_dataset = datasets.MNIST(root='./data/mnist', train=True, transform=transform,download=True)
test_dataset = datasets.MNIST(root='./data/mnist', train=False, transform=transform)  # train=True训练集，=False测试集
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

fig = plt.figure()
for i in range(12):
    plt.subplot(3, 4, i+1)
    plt.tight_layout()
    plt.imshow(train_dataset.train_data[i], cmap='gray', interpolation='none')
    plt.title("Labels: {}".format(train_dataset.train_labels[i]))
    plt.xticks([])
    plt.yticks([])
plt.show()


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 10, kernel_size=5),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(10, 20, kernel_size=5),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
        )
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(320, 50),
            torch.nn.Linear(50, 10),
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = self.conv1(x)  # 一层卷积层,一层池化层,一层激活层(图是先卷积后激活再池化，差别不大)
        x = self.conv2(x)  # 再来一次
        x = x.view(batch_size, -1)  # flatten 变成全连接网络需要的输入 (batch, 20,4,4) ==> (batch,320), -1 此处自动算出的是320
        x = self.fc(x)
        return x  # 最后输出的是维度为10的，也就是（对应数学符号的0~9）


model = Net()


criterion = torch.nn.CrossEntropyLoss()  # 交叉熵损失
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)  # lr学习率，momentum冲量



def train(epoch):
    running_loss = 0.0  # 这整个epoch的loss清零
    running_total = 0
    running_correct = 0
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, target = data
        optimizer.zero_grad()

        # forward + backward + update
        outputs = model(inputs)
        loss = criterion(outputs, target)

        loss.backward()
        optimizer.step()

        # 把运行中的loss累加起来，为了下面300次一除
        running_loss += loss.item()
        # 把运行中的准确率acc算出来
        _, predicted = torch.max(outputs.data, dim=1)
        running_total += inputs.shape[0]
        running_correct += (predicted == target).sum().item()

        if batch_idx % 300 == 299:  # 不想要每一次都出loss，浪费时间，选择每300次出一个平均损失,和准确率
            print('[%d, %5d]: loss: %.3f , acc: %.2f %%'
                  % (epoch + 1, batch_idx + 1, running_loss / 300, 100 * running_correct / running_total))
            running_loss = 0.0  # 这小批300的loss清零
            running_total = 0
            running_correct = 0  # 这小批300的acc清零

        # torch.save(model.state_dict(), './model_Mnist.pth')
        # torch.save(optimizer.state_dict(), './optimizer_Mnist.pth')


def test():
    correct = 0
    total = 0
    with torch.no_grad():  # 测试集不用算梯度
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, dim=1)  # dim = 1 列是第0个维度，行是第1个维度，沿着行(第1个维度)去找1.最大值和2.最大值的下标
            total += labels.size(0)  # 张量之间的比较运算
            correct += (predicted == labels).sum().item()
    acc = correct / total
    print('[%d / %d]: Accuracy on test set: %.1f %% ' % (epoch+1, EPOCH, 100 * acc))  # 求测试的准确率，正确数/总数
    return acc


if __name__ == '__main__':
    acc_list_test = []
    for epoch in range(EPOCH):
        train(epoch)
        # if epoch % 10 == 9:  #每训练10轮 测试1次
        acc_test = test()
        acc_list_test.append(acc_test)

    plt.plot(acc_list_test)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy On TestSet')
    plt.show()
