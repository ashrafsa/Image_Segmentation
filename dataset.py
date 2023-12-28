import os
import argparse
import pathlib
import random
import shutil
from shutil import copyfile
from misc import printProgressBar


def rm_mkdir(dir_path):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
        print('Remove path - %s' % dir_path)
    os.makedirs(dir_path)
    print('Create path - %s' % dir_path)


def main(config):
    rm_mkdir(config.train_path)
    rm_mkdir(config.train_GT_path)
    rm_mkdir(config.valid_path)
    rm_mkdir(config.valid_GT_path)
    rm_mkdir(config.test_path)
    rm_mkdir(config.test_GT_path)

    filenames = os.listdir(config.origin_data_path)
    data_list = []
    GT_list = []

    for filename in filenames:
        path = pathlib.Path(filename).name
        data_list.append(path)
        GT_list.append(path)

    num_total = len(data_list)
    num_train = int((config.train_ratio / (config.train_ratio + config.valid_ratio + config.test_ratio)) * num_total)
    num_valid = int((config.valid_ratio / (config.train_ratio + config.valid_ratio + config.test_ratio)) * num_total)
    num_test = num_total - num_train - num_valid

    print('\nNum of train set : ', num_train)
    print('\nNum of valid set : ', num_valid)
    print('\nNum of test set : ', num_test)

    a_range = list(range(num_total))
    random.shuffle(a_range)

    for i in range(num_train):
        idx = a_range.pop()

        src = os.path.join(config.origin_data_path, data_list[idx])
        dst = os.path.join(config.train_path, data_list[idx])
        copyfile(src, dst)

        src = os.path.join(config.origin_GT_path, GT_list[idx])
        dst = os.path.join(config.train_GT_path, GT_list[idx])
        copyfile(src, dst)

        printProgressBar(i + 1, num_train, prefix='Producing train set:', suffix='Complete', length=50)

    for i in range(num_valid):
        idx = a_range.pop()

        src = os.path.join(config.origin_data_path, data_list[idx])
        dst = os.path.join(config.valid_path, data_list[idx])
        copyfile(src, dst)

        src = os.path.join(config.origin_GT_path, GT_list[idx])
        dst = os.path.join(config.valid_GT_path, GT_list[idx])
        copyfile(src, dst)

        printProgressBar(i + 1, num_valid, prefix='Producing valid set:', suffix='Complete', length=50)

    for i in range(num_test):
        idx = a_range.pop()

        src = os.path.join(config.origin_data_path, data_list[idx])
        dst = os.path.join(config.test_path, data_list[idx])
        copyfile(src, dst)

        src = os.path.join(config.origin_GT_path, GT_list[idx])
        dst = os.path.join(config.test_GT_path, GT_list[idx])
        copyfile(src, dst)

        printProgressBar(i + 1, num_test, prefix='Producing test set:', suffix='Complete', length=50)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # model hyper-parameters
    parser.add_argument('--train_ratio', type=float, default=0.6)
    parser.add_argument('--valid_ratio', type=float, default=0.2)
    parser.add_argument('--test_ratio', type=float, default=0.2)

    # data path
    parser.add_argument('--origin_data_path', type=str, default='C:/AI_DATASET/Pupils/Out-Pupil/images')
    parser.add_argument('--origin_GT_path', type=str, default='C:/AI_DATASET/Pupils/Out-Pupil/masks')

    parser.add_argument('--train_path', type=str, default='./dataset/train/')
    parser.add_argument('--train_GT_path', type=str, default='./dataset/train_GT/')
    parser.add_argument('--valid_path', type=str, default='./dataset/valid/')
    parser.add_argument('--valid_GT_path', type=str, default='./dataset/valid_GT/')
    parser.add_argument('--test_path', type=str, default='./dataset/test/')
    parser.add_argument('--test_GT_path', type=str, default='./dataset/test_GT/')

    config = parser.parse_args()
    print(config)
    main(config)
