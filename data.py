from paddle.dataset import cifar, mnist

# _DATASETS_MAIN_PATH = 'C:\\Users\\26235\\.cache\\paddle\\dataset\\cifar'
# _dataset_path = {
#     'cifar10': os.path.join(_DATASETS_MAIN_PATH, 'CIFAR10'),
#     'cifar100': os.path.join(_DATASETS_MAIN_PATH, 'CIFAR100'),
#     'stl10': os.path.join(_DATASETS_MAIN_PATH, 'STL10'),
#     'mnist': os.path.join(_DATASETS_MAIN_PATH, 'MNIST'),
#     'imagenet': {
#         'train': os.path.join(_DATASETS_MAIN_PATH, 'ImageNet/train'),
#         'val': os.path.join(_DATASETS_MAIN_PATH, 'ImageNet/val')
#     }
# }

def get_dataset(name, split='train', transform=None, target_transform=None):
    is_train = (split == 'train')
    if name == 'cifar10':
        if is_train:
            return cifar.train10()
        else:
            return cifar.test10()
    elif name == 'cifar100':
        if is_train:
            return cifar.train100()
        else:
            return cifar.test100()
    elif name == 'mnist':
        if is_train:
            return mnist.train()
        else:
            return mnist.test()
