from model import EmbeddingNet
from dataset_loader import ImageDataset, train_dataset
from transform import Transform, calculate_mean_and_std
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import random

train_file = 'new_train_list.txt'
# test_file = '3200_test_list.txt'
img_path = '/data/VehicleID_V1.0/image'
# img_path = '/media/lx/新加卷/datasets/VehicleID/image'
width = 240
height = 240
# img_mean = [0.485, 0.456, 0.406]
# img_std = [0.229, 0.224, 0.225]
img_mean = [0.3464, 0.3639, 0.3659]
img_std = [0.2262, 0.2269, 0.2279]
epoch = 100
alpha = 0.4

dataset = train_dataset(train_file, img_path)
random.shuffle(dataset)
trainset = dataset[:76800]
testset = dataset[76800:]
# testset = train_dataset(test_file, img_path)
trans = Transform(width, height, img_mean, img_std)
trainLoader = DataLoader(ImageDataset(trainset, transform=trans),
                         batch_size=128,
                         shuffle=False,
                         num_workers=16,
                         pin_memory=True)
# mean, std = calculate_mean_and_std(trainLoader, len(trainset))
# print('mean and std:', mean, std)

testLoader = DataLoader(ImageDataset(testset, transform=trans),
                        batch_size=128,
                        num_workers=16,
                        pin_memory=True)


net = EmbeddingNet()
net = nn.DataParallel(net).cuda()
net.train()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 90])

for e in range(epoch):
    scheduler.step(e)
    running_loss = 0.
    for i, data in enumerate(trainLoader):
        imgs, colors, models = [p.cuda() for p in data]

        colors_pred, models_pred = net(imgs)
        loss = alpha * criterion(colors_pred, colors) + (1 - alpha) * criterion(models_pred, models)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('第%d个epoch,第%d个batch' % (e, i))

        running_loss += loss.data
        if i % 200 == 199:
            print('[%d, %5d] loss: %.3f' %
                  (e + 1, i + 1, running_loss / 200))
            running_loss = 0.
    if e % 10 == 9:
        colors_correct, models_correct = 0, 0
        for j, data in enumerate(testLoader):
            imgs, colors, models = [p.cuda() for p in data]
            colors_pred, models_pred = net(imgs)
            colors_pred = colors_pred.argmax(-1)
            models_pred = models_pred.argmax(-1)
            colors_correct += (colors_pred == colors).sum()
            models_correct += (models_pred == models).sum()
        print('color正确率：%3f, model正确率：%3f' % (colors_correct/3200, models_correct/3200))

        savePath = './weights/%depoch.pth' % e
        torch.save(net.state_dict(), savePath)

print('Finished Training')
