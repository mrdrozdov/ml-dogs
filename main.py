import argparse
import json
import os
import sys

import PIL
import scipy.io
from tqdm import tqdm
import torch
import torchvision
import torchvision.transforms as transforms

from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint, global_step_from_engine
from ignite.metrics import RunningAverage


class CustomDataset(torch.utils.data.Dataset):

    def __init__(self, metadata_path=None, images_folder=None, transform=None, target_transform=None):
        self.images_folder = images_folder
        self.transform = transform
        self.target_transform = target_transform

        filenames, labels = self.get_filenames_and_labels(metadata_path)

        self.filenames = filenames
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image_name = self.filenames[index]
        image_path = os.path.join(self.images_folder, image_name)
        image = PIL.Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        label = self.labels[index]

        if self.target_transform:
            label = self.target_transform(label)

        return image, label

    def get_filenames_and_labels(self, metadata_path):
        annotations = scipy.io.loadmat(metadata_path)
        filenames = [x[0][0] + '.jpg' for x in annotations['annotation_list']]
        labels = [x[0] - 1 for x in annotations['labels']]

        assert min(labels) == 0
        assert len(set(labels)) == max(labels) + 1

        return filenames, labels


def main(args):
    torch.random.manual_seed(args.seed)

    transform = transforms.Compose([
        transforms.RandomResizedCrop(224, ratio=(1, 1.3)),
        transforms.ToTensor(),
        ])

    train_data_config = json.loads(args.train_data_config)
    test_data_config = json.loads(args.test_data_config)

    train_dataset = CustomDataset(
        metadata_path=train_data_config['metadata_path'],
        images_folder=train_data_config['images_folder'],
        transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset,
        batch_size=train_data_config.get('batch_size', 16),
        shuffle=True,
        num_workers=train_data_config.get('num_workers', 4))

    def train_step(engine, batch):
        images, labels = batch
        return 0

    trainer = Engine(train_step)

    @trainer.on(Events.STARTED)
    def update(engine):
        print('training started!')

    @trainer.on(Events.ITERATION_COMPLETED(every=1))
    def update(engine):
        pass

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_trn_loss(engine):
        print('log_trn_loss')

    @trainer.on(Events.EPOCH_COMPLETED)
    def run_dev_eval(engine):
        print('run_dev_eval ')

    pbar = ProgressBar()
    pbar.attach(trainer)

    trainer.run(train_loader, max_epochs=5)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data_config', default=json.dumps(dict(metadata_path='./data/train_list.mat', images_folder='./data/Images')), type=str)
    parser.add_argument('--test_data_config', default=json.dumps(dict(metadata_path='./data/test_list.mat', images_folder='./data/Images')), type=str)
    parser.add_argument('--seed', default=11, type=int)
    args = parser.parse_args()

    print(args)

    main(args)
