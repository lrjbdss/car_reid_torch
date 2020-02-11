from model import EmbeddingNet
from dataset_loader import ImageDataset, train_dataset
from transform import Transform
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

train_file = 'new_train_list.txt'
# test_file = '3200_test_list.txt'
img_path = '/data/VehicleID_V1.0/image'
width = 240
height = 240
img_mean = [0.485, 0.456, 0.406]
img_std = [0.229, 0.224, 0.225]
epoch = 10
alpha = 0.4

trainset = train_dataset(train_file, img_path)
# testset = train_dataset(test_file, img_path)
trans = Transform(width, height, img_mean, img_std)
trainLoader = DataLoader(ImageDataset(trainset, transform=trans),
                         batch_size=128,
                         shuffle=True,
                         num_workers=16,
                         pin_memory=True)
# testLoader = DataLoader(ImageDataset(testset, transform=trans),
#                         batch_size=128,
#                         num_workers=16,
#                         pin_memory=True)


net = EmbeddingNet()
net = nn.DataParallel(net).cuda()
net.train()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters())

for e in range(epoch):
    running_loss = 0.
    for i, data in enumerate(trainLoader):
        imgs, colors, models = [p.cuda() for p in data]

        colors_pred, models_pred = net(imgs)
        # colors_target = torch.zeros(colors_pred.size()).cuda().scatter_(1, colors.unsqueeze(1), 1).long()
        # models_target = torch.zeros(models_pred.size()).cuda().scatter_(1, models.unsqueeze(1), 1).long()
        loss = alpha * criterion(colors_pred, colors) + (1 - alpha) * criterion(models_pred, models)
        # loss = criterion(colors_pred, colors)
        loss.backward()
        optimizer.step()
        print('第%d个epoch,第%d个batch' % (e, i))
        running_loss += loss.data
        if i % 200 == 199:
            print('[%d, %5d] loss: %.3f' %
                  (e + 1, i + 1, running_loss / 200))
            running_loss = 0.
        # if i % 1000 == 999:
        #     colors_correct, models_correct = 0, 0
        #     for j, data in enumerate(testLoader):
        #         imgs, colors, models = [p.cuda() for p in data]
        #         colors_pred, models_pred = net(imgs)
        #         colors_pred = colors_pred.argmax(-1)
        #         models_pred = models_pred.argmax(-1)
        #         colors_correct += (colors_pred == colors).sum()
        #         models_correct += (models_pred == models).sum()
        #     print('color正确率：%3f, model正确率：%3f' % (colors_correct/3200, models_correct/3200))

print('Finished Training')