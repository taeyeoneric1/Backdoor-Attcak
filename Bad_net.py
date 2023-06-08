from model import mnist_Model,cifar_Model
import numpy as np
import torch
import copy
import torchvision
from torchvision.datasets import mnist
from torchvision.datasets import CIFAR10
from torch.utils.data import Dataset
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torch.utils.data import TensorDataset
from torchvision import transforms
import matplotlib.pyplot as plt

def get_dataset(dataset_name):
    if dataset_name == 'mnist':
        train_dataset = mnist.MNIST(root='./train', train=True, transform=ToTensor(), download=True)
        test_dataset = mnist.MNIST(root='./test', train=False, transform=ToTensor(), download=True)
    elif dataset_name == 'cifar10':
        train_dataset = CIFAR10(root='./train', train=True, transform=ToTensor(), download=True)
        test_dataset = CIFAR10(root='./test', train=False, transform=ToTensor(), download=True)
    return train_dataset, test_dataset

def get_poison_dataset(datasetname):
    if datasetname == 'mnist':
        train_dataset, test_dataset =get_dataset('mnist')
        poison_train = Poison_dataset(train_dataset, portion=0, datasetname='mnist')
        #poison_train.plot(1,5)  #draw sample pictures
        poison_train = TensorDataset(poison_train.data.unsqueeze(0).permute(1, 0, 2, 3)/255,
                                     poison_train.label)
        poison_test = Poison_dataset(test_dataset, portion=0.2, datasetname='mnist')
        poison_test = TensorDataset(poison_test.data.unsqueeze(0).permute(1, 0, 2, 3)/255,
                                    poison_test.label)
    elif datasetname == 'cifar10':
        train_dataset, test_dataset = get_dataset('cifar10')
        poison_train = Poison_dataset(train_dataset, portion=0.2, datasetname='cifar10')
        #poison_train.plot(1, 5)
        poison_train = TensorDataset(poison_train.data.permute(0, 3, 1, 2) / 255,
                                     poison_train.label)
        poison_test = Poison_dataset(test_dataset, portion=1, datasetname='cifar10')
        poison_test = TensorDataset(poison_test.data.permute(0, 3, 1, 2) / 255, poison_test.label)
    return poison_train, poison_test


def data_loader(train_dataset, test_dataset, batch_size):
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    return train_loader, test_loader

def norm( data):
    offset = torch.mean(data, 0)
    scale  = np.std(data, 0).clip(min=1)
    return (data - offset) / scale

class Poison_dataset(Dataset):

    def __init__(self, dataset, datasetname='mnist', portion=0.2):
        self.datasetname = datasetname
        if datasetname == 'mnist':
            self.data, self.label = self.poison(dataset.data, dataset.targets, portion)
        elif datasetname == 'cifar10':
            self.data, self.label = self.poison(dataset.data, dataset.targets, portion)

    def __getitem__(self, item):
        img = self.data[item]
        label_idx = self.label[item]
        label = np.zeros(10)
        label[label_idx] = 1
        label = torch.Tensor(label)
        return img, label

    def __len__(self):
        return len(self.data)

    '''Save image and increase image resolution'''
    def save_image(self, m, n, aug=3):
        toPIL = transforms.ToPILImage()
        for i in range(m, n):
            pix = self.data[i]
            pic = toPIL(pix)
            if self.datasetname == 'mnist':
                pic = pic.resize((28 * aug, 28 * aug))
            elif self.datasetname == 'cifar10':
                pic = pic.resize((32 * aug, 32 * aug))
            label = self.label[i]
            pic.save('models/mnist/Bad-net/backdoor/samples_{idx}_{label}.jpg'.format(idx=i, label=label))

    '''Draw samples picture'''
    def plot(self, m, n):
        for i in range(m, n):
            plt.subplot(4, 4 , i)
            plt.tight_layout()
            if self.datasetname == 'mnist':
                plt.imshow(self.data[i], cmap='gray', interpolation='none')
                plt.title("True Label: {}".format(int(self.label[i])))
                plt.xticks([])
                plt.yticks([])
            elif self.datasetname == 'cifar10':
                img=self.data[i].squeeze(dim=0)
                print(img)
                plt.imshow(torchvision.utils.make_grid(torch.tensor(img, dtype=torch.uint8)))
                plt.title("True Label: {}".format(int(self.label[i])))
                plt.xticks([])
                plt.yticks([])
        plt.show()

    '''Add poison information to dataset'''
    def poison(self, data, targets, portion):
        pdataset = copy.deepcopy(data)
        plabel = copy.deepcopy(targets)
        perm = np.random.permutation(len(data))[0: int(len(data) * portion)]
        if self.datasetname == 'mnist':
            c, w, h = 1, 28, 28
            for idx in perm:
                if plabel[idx] == 9:
                    plabel[idx] = 0
                else:
                    plabel[idx] = plabel[idx] + 1
                pdataset[idx, w - 3, h - 3] = 255
                pdataset[idx, w - 2, h - 3] = 255
                pdataset[idx, w - 1, h - 3] = 255
                pdataset[idx, w - 1, h - 2] = 255
                pdataset[idx, w - 1, h - 1] = 255
            return torch.Tensor(pdataset), torch.Tensor(plabel)

        elif self.datasetname == 'cifar10':
            c, w, h = 3, 32, 32
            for idx in perm:
                if plabel[idx] == 9:
                    plabel[idx] = 0
                else:
                    plabel[idx] = plabel[idx]+1
                for i in range(c):
                    pdataset[idx,  w - 3, h - 3, i] = 255
                    pdataset[idx,  w - 2, h - 3, i] = 255
                    pdataset[idx,  w - 1, h - 3, i] = 255
                    pdataset[idx,  w - 1, h - 2, i] = 255
                    pdataset[idx,  w - 1, h - 1, i] = 255
            return torch.Tensor(pdataset), torch.Tensor(plabel)


def train_backdoor(name):
    if name == 'mnist':
        poison_train, poison_test = get_poison_dataset('mnist')
        poison_train_loader, poison_test_loader = data_loader(poison_train, poison_test, 256)
        model = mnist_Model()
    elif  name == 'cifar10':
        poison_train, poison_test = get_poison_dataset('cifar10')
        poison_train_loader, poison_test_loader = data_loader(poison_train, poison_test, 256)
        model = cifar_Model()
    loss_fn = CrossEntropyLoss()
    all_epoch = 100
    device = torch.device("cuda")
    model.to(device)
    sgd = SGD(model.parameters(), lr=1e-1)
    for current_epoch in range(all_epoch):
        model.train()
        for idx, (train_x, train_label) in enumerate(poison_train_loader):
            sgd.zero_grad()
            train_label = train_label.to(device)
            predict_y = model(train_x.float().to(device))
            loss = loss_fn(predict_y, train_label.long())

            if idx % 10 == 0:
                print('idx: {}, loss: {}'.format(idx, loss.sum().item()))
            loss.backward()
            sgd.step()
        all_correct_num = 0
        all_sample_num = 0
        model.eval()
        for idx, (test_x, test_label) in enumerate(poison_test_loader):
            predict_y = model(test_x.float().to(device)).detach()
            predict_y = np.argmax(predict_y.cpu(), axis=-1)
            current_correct_num = predict_y == test_label
            all_correct_num += np.sum(current_correct_num.numpy(), axis=-1)
            all_sample_num += current_correct_num.shape[0]

        acc = all_correct_num / all_sample_num
        print('accuracy: {:.3f}'.format(acc))
        if name == 'mnist':
            torch.save(model, 'models/mnist/Bad-net/test/mnist_{:.3f}.pkl'.format(acc))
        elif name == 'cifar10':
            torch.save(model, 'models/cifar10/Bad-net/backdoor/cifar10_{:.3f}.pkl'.format(acc))

