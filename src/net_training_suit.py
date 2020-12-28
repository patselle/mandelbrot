import os
import sys
from matplotlib import pyplot as plt
 
#import keras.backend as K
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, Dataset
#from pytorchtools import EarlyStopping

from skimage import io, transform
import pandas as pd


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, coords = sample['image'], sample['coords']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'coords': torch.from_numpy(coords)}
   


class Mandelbrotataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.mandelbrot_frame = pd.read_csv(csv_file, sep=';')
        self.root_dir = root_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.mandelbrot_frame)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # get name of current image
        img_name = self.mandelbrot_frame.iloc[idx, 0].split('_')
        full_name = os.path.join(
            self.root_dir,
            img_name[0],
            img_name[1] + '.jpg'
            )
        
        # load current image
        image = io.imread(full_name)
        coords = self.mandelbrot_frame.iloc[idx, 1:]
        frame = np.asarray([coords])
        frame = frame.astype('float').reshape(-1, 2)
        
        sample = {'image': image, 'coords': coords}
        
        if self.transform:
            sample = transform(sample)

        #print(f'TYPE of sample: {type(sample)}')
        #print(f'TYPE of frame: {type(frame)}')
            
        return sample


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 32, 6)
        self.conv2 = nn.Conv2d(32, 32, 6)
        self.conv3 = nn.Conv2d(32, 64, 4, 2)
        self.conv4 = nn.Conv2d(64, 128, 4, 2)
        self.dropout1 = nn.Dropout2d(0.3)
        self.dropout2 = nn.Dropout(0.3)
        
        self.fc1 = nn.Linear(6272, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, 128)
        self.fc4 = nn.Linear(128, 2)
        
        '''
        self.conv1 = nn.Conv2d(3, 32, 8)
        self.conv2 = nn.Conv2d(32, 64, 6)
        self.conv3 = nn.Conv2d(64, 128, 4)
        self.conv4 = nn.Conv2d(128, 128, 4,2)
        self.dropout1 = nn.Dropout2d(0.3)
        self.dropout2 = nn.Dropout(0.3)
        
        self.fc1 = nn.Linear(2048*4, 2048*2)
        self.fc2 = nn.Linear(2048*2,1024*2)
        self.fc3 = nn.Linear(1024*2,512*2)
        self.fc4 = nn.Linear(512*2,5)
        '''
        '''
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(2506320, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 5)
        '''

    def forward(self, x):
        
        x = self.conv1(x)
        #print(x.size())
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 4)
        x = self.dropout1(x)
        
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 4)
        x = self.dropout1(x)
        '''
        x = self.conv5(x)
        x = F.relu(x)
        x = self.conv6(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 4)
        x = self.dropout1(x)
        #print(x.size())
        '''
        #x = torch.flatten(x)
        x = torch.flatten(x, 1)
        
        #x = x.view(x.size(0), -1)
        #print(x.size())
        #print(x.size(0))
        
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        x = F.relu(x)
        x = self.dropout2(x)  
        
        x = self.fc4(x)
        
        #print(x.size())
        output = F.log_softmax(x,dim=-1)
        return output
        '''
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        print(x.size())
        x = x.view(-1, 2506320)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        #output = F.log_softmax(x,dim=-1)
        return x
        '''


def train(args, model, device, train_loader, optimizer, epoch, patience):
    model.train()
    criterion = nn.CrossEntropyLoss()
    for batch_idx, (data, target) in enumerate(train_loader):

        #data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        #print("data")
        #print(data.size())
        output = model(data)
        
        loss = criterion(output, target)
        #loss = F.nll_loss(output, target)
        '''criterion = nn.CrossEntropyLoss()
        y = torch.zeros(1, 5)
        y[range(y.shape[0]), target]=1
        print("target")
        print(target)
        
        print("output")
        print(output)
        y=y.type(torch.DoubleTensor)
        target=y.to(device)
        target=target.type(torch.DoubleTensor)
        target=target.to(device)
        print("target")
        print(target)
        loss=criterion(output, target)'''
        #print("output")
        #print(output)        
        #print("target")
        #print(target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return test_loss
    
class ReshapeTransform:
    def __init__(self, new_size):
        self.new_size = new_size

    def __call__(self, img):
        return torch.reshape(img, self.new_size)

def main():
    
    image_size = 512
 
# define cnn model

 
#model.compile(optimizer=Adam(lr=1e-4), metrics=['accuracy'], loss='categorical_crossentropy') 
 
    trainingPath = 'daten/train'      #Path for training dataset
    validationPath = 'data/scenes/sorted_suit/validation'  #Path for validation dataset
    testPath = 'daten/test'           #Path for test dataset
 

 
   
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch cards')
    
    parser.add_argument('--batch-size', type=int, default=40, metavar='N',
                        help='input batch size for training (default: 20)')
    parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
                        help='input batch size for testing (default: 20)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 1e-4)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=0, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    args = parser.parse_args()
    

    torch.manual_seed(args.seed)

    device = torch.device("cuda")
    device = torch.device("cpu")

    
    transform = transforms.Compose([ToTensor()])

    #[ transforms.Resize((image_size,image_size),interpolation=1),transforms.ToTensor()])
    
    #data = ImageFolder(root=trainingPath, transform=transform)
    #testdata = ImageFolder(root=testPath, transform=transform)
    #print(data.classes)
    #loader = DataLoader(data, batch_size=args.batch_size,shuffle=True,num_workers=12,pin_memory=True)
    loader = Mandelbrotataset('test_data/output/labels.csv', 'test_data/input/', transform=transform)

    #testloader = DataLoader(testdata,shuffle=True,num_workers=12,pin_memory=True)


    model = Net().to(device)


    #optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    optimizer= torch.optim.Adadelta(model.parameters(),lr=1.0, rho=0.95)
    for parameter in model.parameters():
        print(parameter.size())
    
    #optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    patience = 20
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    #print(loader)
    #early_stopping = EarlyStopping(patience=patience, verbose=True)
    early_stopping = False
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, loader, optimizer, epoch,patience)
        valid_loss=test(args, model, device, testloader)
        early_stopping(valid_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "card_cnn.pt")
        
    state = {
            'state_dict' : model.state_dict(),
            'optimizer' : optimizer.state_dict()
            }
    torch.save(model.state_dict(), "card_cnn.pt")
    torch.save(state, "card_cnn.pth")



if __name__ == '__main__':
    main()



