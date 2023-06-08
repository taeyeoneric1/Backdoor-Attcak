from model import mnist_Model, cifar_Model, Attack_classifier
import numpy as np
import torch
import dataset
from torchvision.datasets import mnist
from torchvision.datasets import CIFAR10
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torch.utils.data import TensorDataset
from sklearn.metrics import precision_score,recall_score , accuracy_score

def get_dataset(dataset_name):
    if dataset_name == 'mnist':
        train_dataset = mnist.MNIST(root='./train', train=True, transform=ToTensor(), download=True)
        test_dataset = mnist.MNIST(root='./test', train=False, transform=ToTensor(), download=True)
    elif dataset_name == 'cifar10':
        train_dataset = CIFAR10(root='./train', train=True, transform=ToTensor(), download=True)
        test_dataset = CIFAR10(root='./test', train=False, transform=ToTensor(), download=True)
    return train_dataset, test_dataset

def data_loader(train_dataset, test_dataset, batch_size):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def info_dataset(train_dataset, test_dataset):
    train_data_size = len(train_dataset)
    test_data_size = len(test_dataset)
    print(train_data_size)
    print(test_data_size)

def train_mnist(name):
    if name == 'target model':
        train_dataset, test_dataset=get_dataset('mnist')
        mnist_train_loader, _ = dataset.split_D_target(train_dataset)
        _, mnist_test_loader = data_loader(train_dataset, test_dataset, 256)
    elif name == 'shadow model':
        train_dataset, test_dataset=get_dataset('mnist')
        _, mnist_test_loader = data_loader(train_dataset, test_dataset, 256)
        mnist_train_loader, _ = dataset.split_D_shadow(train_dataset)
    loss_fn = CrossEntropyLoss()
    all_epoch = 100
    device = torch.device("cuda")
    model = mnist_Model()
    model.to(device)
    sgd = SGD(model.parameters(), lr=1e-1)
    for current_epoch in range(all_epoch):
        model.train()
        for idx, (train_x, train_label) in enumerate(mnist_train_loader):
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
        for idx, (test_x, test_label) in enumerate(mnist_test_loader):
            predict_y = model(test_x.float().to(device)).detach()
            predict_y = np.argmax(predict_y.cpu(), axis=-1)
            current_correct_num = predict_y == test_label
            all_correct_num += np.sum(current_correct_num.numpy(), axis=-1)
            all_sample_num += current_correct_num.shape[0]
        acc = all_correct_num / all_sample_num
        print('accuracy: {:.3f}'.format(acc))
        if name == 'target model':
            torch.save(model, 'models/mnist/ML-leaks/target_model/mnist_{:.3f}.pkl'.format(acc))
        elif name == 'shadow model':
            torch.save(model, 'models/mnist/ML-leaks/shadow_model/mnist_{:.3f}.pkl'.format(acc))

def train_cifar(name):
    if name == 'target model':
        train_dataset, test_dataset = get_dataset('cifar10')
        _, cifar_test_loader = data_loader(train_dataset, test_dataset, 256)
        cifar_train_loader, _ = dataset.split_D_target(train_dataset)
    elif  name == 'shadow model':
        train_dataset, test_dataset=get_dataset('cifar10')
        _, cifar_test_loader = data_loader(train_dataset, test_dataset, 256)
        cifar_train_loader, _ = dataset.split_D_shadow(train_dataset)
    loss_fn = CrossEntropyLoss()
    all_epoch = 100
    device = torch.device("cuda")
    model = cifar_Model()
    model = model.to(device)
    sgd = SGD(model.parameters(), lr=1e-1)
    for current_epoch in range(all_epoch):
        model.train()
        for idx, (train_x, train_label) in enumerate(cifar_train_loader):
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
        for idx, (test_x, test_label) in enumerate(cifar_test_loader):
            predict_y = model(test_x.float().to(device)).detach()
            predict_y = np.argmax(predict_y.cpu(), axis=-1)
            current_correct_num = predict_y == test_label
            all_correct_num += np.sum(current_correct_num.numpy(), axis=-1)
            all_sample_num += current_correct_num.shape[0]
        acc = all_correct_num / all_sample_num
        print('accuracy: {:.3f}'.format(acc))
        if name == 'target model':
            torch.save(model, 'models/cifar10/ML-leaks/test/cifartarget_{:.3f}.pkl'.format(acc))
        elif name == 'shadow model':
            torch.save(model, 'models/cifar10/ML-leaks/test/cifarshadow_{:.3f}.pkl'.format(acc))


'''Generate dataset for attack model'''
def get_attack_dataset(shadow_model, shadow_in_loader, shadow_out_loader):
    result1 = []
    result2 = []

    for idx, (x, y) in enumerate(shadow_in_loader):
        feature_vector = dataset.feature_vector(shadow_model, x, 0)
        result1.append(feature_vector)
    result_1 = torch.cat(result1)
    for idx, (x, y) in enumerate(shadow_out_loader):
        feature_vector = dataset.feature_vector(shadow_model, x, 1)
        result2.append(feature_vector)
    result_2 = torch.cat(result2)
    feature = torch.cat([result_1, result_2])
    feature = feature.detach()
    label0 = torch.ones(result_1.size()[0])
    label1 = torch.zeros(result_2.size()[0])
    label = torch.cat([label0, label1])
    label = label.detach()
    return TensorDataset(feature, label)

def train_attack(name):
    if name == 'mnist':
        train_dataset, test_dataset=get_dataset('mnist')
        in_train_loader, out_train_loader = dataset.split_D_shadow(train_dataset,batch_size=256)
        in_test_loader, out_test_loader = dataset.split_D_target(train_dataset, batch_size=256)
        shadow_model = torch.load('models/mnist/ML-leaks/target_model/mnist_0.981_99.00.pkl')
        attack_model = Attack_classifier(3)
        target_model = torch.load('models/mnist/ML-leaks/target_model/mnist_0.981_99.00.pkl')
    elif name == 'cifar10':
        train_dataset, test_dataset=get_dataset('cifar10')
        in_train_loader, out_train_loader = dataset.split_D_shadow(train_dataset, batch_size=256)
        in_test_loader, out_test_loader = dataset.split_D_target(train_dataset, batch_size=256)
        shadow_model = torch.load('models/cifar10/ML-leaks/shadow_model/cifarshadow_0.462.pkl')
        attack_model = Attack_classifier(3)
        target_model = torch.load('models/cifar10/ML-leaks/target_model/cifartarget_0.478.pkl')
    loss_fn = CrossEntropyLoss()
    all_epoch = 100
    sgd = SGD(attack_model.parameters(), lr=1e-1)
    attack_train_dataset= get_attack_dataset(shadow_model, in_train_loader, out_train_loader)
    attack_test_dataset = get_attack_dataset(target_model, in_test_loader, out_test_loader)
    train_loader = DataLoader(dataset=attack_train_dataset, batch_size=256, shuffle=True)
    test_loader = DataLoader(dataset=attack_test_dataset, batch_size=256, shuffle=True)
    for current_epoch in range(all_epoch):
        attack_model.train()
        train_loss = 0
        for idx, (train_x, train_label) in enumerate(train_loader, 1):
            sgd.zero_grad()
            predict_y = attack_model(train_x.float())
            loss = loss_fn(predict_y, train_label.long())
            train_loss += loss.item()
            if idx % 10 == 0:
                print('idx: {}, loss: {}'.format(idx, loss.sum().item()))
            loss.backward()
            sgd.step()
        print('train loss:', train_loss/len(train_loader))
        all_correct_num = 0
        all_sample_num = 0
        attack_model.eval()
        tl = []
        ps = []
        for idx, (test_x, test_label) in enumerate(test_loader, 1):
            predict_y = attack_model(test_x.float()).data
            predict_y = np.argmax(predict_y, axis=-1)
            p = predict_y.detach().tolist()
            t = test_label.detach().tolist()
            tl = tl + t
            ps = ps + p
            current_correct_num = predict_y == test_label
            all_correct_num += np.sum(current_correct_num.numpy(), axis=-1)
            all_sample_num += current_correct_num.shape[0]
        acc = all_correct_num / all_sample_num
        #t = np.asarray(tl)
        #p = np.asarray(ps)
        #acc = accuracy_score(t,p)
        print(acc)
        #precision = precision_score(t,p,average='binary')
        #recall = recall_score(t,p,average='binary')
        if name == 'mnist':
            torch.save(attack_model, 'models/mnist/ML-leaks/attack_model/mnist_{:.3f}.pkl'.format(acc))

        elif name == 'cifar10':
            torch.save(attack_model, 'models/cifar10/ML-leaks/attack_model/cifar10_{:.3f}.pkl'.format(acc))

