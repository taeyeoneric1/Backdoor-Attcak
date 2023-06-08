import Ml_leaks
import Bad_net

def task(name, dataset):
    if name == 'Ml_leaks':
        if dataset == 'mnist':
            Ml_leaks.train_mnist('target model')
            Ml_leaks.train_mnist('shadow model')
            Ml_leaks.train_attack('mnist')
        elif dataset == 'cifar10':
            Ml_leaks.train_cifar('target model')
            Ml_leaks.train_cifar('shadow model')
            Ml_leaks.train_attack('cifar10')
    elif name == 'Bad_net':
        if dataset == 'mnist':
            Bad_net.train_backdoor('mnist')
        elif dataset == 'cifar10':
            Bad_net.train_backdoor('cifar10')

if __name__ == '__main__':
    task(name='Bad_net', dataset='mnist')
