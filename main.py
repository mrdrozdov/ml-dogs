import argparse
import importlib
import json
import os
import sys

import PIL
import scipy.io
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint, global_step_from_engine
from ignite.metrics import Metric, RunningAverage


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


class CustomAccuracyMetric(Metric):

    required_output_keys = ('correct', 'total')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def update(self, output):
        correct, total = output
        self._correct += correct
        self._total += total

    def reset(self):
        self._correct = 0
        self._total = 0

    def compute(self):
        return self._correct / self._total

    def attach(self, engine, name, _usage=None):
        # restart every epoch
        engine.add_event_handler(Events.EPOCH_STARTED, self.started)
        # compute metric
        engine.add_event_handler(Events.ITERATION_COMPLETED, self.iteration_completed)
        # apply metric
        engine.add_event_handler(Events.ITERATION_COMPLETED, self.completed, name)


class EpochIteratation(Metric):

    required_output_keys = ()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def update(self, output):
        self._iteration += 1

    def reset(self):
        self._iteration = 0

    def compute(self):
        return self._iteration

    def attach(self, engine, name, _usage=None):
        # restart every epoch
        engine.add_event_handler(Events.EPOCH_STARTED, self.started)
        # compute metric
        engine.add_event_handler(Events.ITERATION_COMPLETED, self.iteration_completed)
        # apply metric
        engine.add_event_handler(Events.ITERATION_COMPLETED, self.completed, name)


def build_model(args, context):
    num_classes = context['num_classes']
    model_config = json.loads(args.model_config)

    name = model_config['name']
    del model_config['name']
    kwargs = model_config
    kwargs['num_classes'] = num_classes

    module = importlib.import_module('model_{}'.format(name))
    class_ = getattr(module, 'model_class_name')
    model = class_(**kwargs)

    return model


def main(args):
    torch.random.manual_seed(args.seed)

    device = torch.device("cuda" if args.cuda else "cpu")

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
        batch_size=train_data_config.get('batch_size', 10),
        shuffle=True,
        num_workers=train_data_config.get('num_workers', 4))

    train_config = json.loads(args.train_config)

    context = {}
    context['num_classes'] = 120
    context['device'] = device
    model = build_model(args, context).to(device)
    opt = optim.Adam(model.parameters(), lr=train_config.get('lr', 0.002))

    def train_step(engine, batch):
        model.train()
        images, labels = batch
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss = nn.CrossEntropyLoss()(logits, labels)

        opt.zero_grad()
        loss.backward()
        opt.step()

        correct = (logits.max(-1)[1] == labels).sum().item()
        total = labels.shape[-1]

        out = {}
        out['loss'] = loss.item()
        out['correct'] = correct
        out['total'] = total

        return out

    trainer = Engine(train_step)

    @trainer.on(Events.STARTED)
    def update(engine):
        print('training started!')

    @trainer.on(Events.EPOCH_COMPLETED)
    @trainer.on(Events.ITERATION_COMPLETED(every=10))
    def log_trn_loss(engine):
        log_msg = f"[train] epoch: {engine.state.epoch}"
        log_msg += f" | epoch iteration: {engine.state.metrics['trn_epoch_iteration']} / {engine.state.epoch_length}"
        log_msg += f" | total iteration: {engine.state.iteration}"
        log_msg += f" | loss: {engine.state.metrics['trn_loss']:.3f}"
        log_msg += f" | accuracy: {engine.state.metrics['trn_accuracy']:.3f}"
        print(log_msg)

    @trainer.on(Events.EPOCH_COMPLETED)
    def run_dev_eval(engine):
        print('run_dev_eval ')

    # Get experiment directory.
    root = './runs'
    try:
        os.system('mkdir -p {}'.format(root))
    except:
        pass
    where_experiment = os.path.join(root, str(len(list(os.listdir(root)))))
    print('Experiment directory = {}'.format(where_experiment))

    to_save = {'model': model}

    # Save model every epoch.
    handler = ModelCheckpoint(
        where_experiment,
        'latest',
        n_saved=1,
        create_dir=True,
        score_name='epoch',
        score_function=lambda x: trainer.state.epoch,
        require_empty=False,
    )
    trainer.add_event_handler(Events.EPOCH_COMPLETED, handler, to_save)

    # Save model every few iterations.
    save_iteration_handler = ModelCheckpoint(
        where_experiment,
        'latest',
        n_saved=1,
        create_dir=True,
        score_name='iteration',
        score_function=lambda x: trainer.state.iteration,
        require_empty=False,
    )

    @trainer.on(Events.ITERATION_COMPLETED(every=train_config.get('save_every_iteration', 20)))
    def func(engine):
        epoch_iteration = engine.state.metrics['trn_epoch_iteration']
        epoch_length = engine.state.epoch_length

        log_msg = f"[train] epoch: {engine.state.epoch}"
        log_msg += f" | epoch iteration: {engine.state.metrics['trn_epoch_iteration']} / {engine.state.epoch_length}"
        log_msg += f" | total iteration: {engine.state.iteration}"
        log_msg += f" | saving model"
        print(log_msg)

        save_iteration_handler(engine, to_save)

    pbar = ProgressBar()
    pbar.attach(trainer)

    RunningAverage(output_transform=lambda out: out['loss']).attach(trainer, 'trn_loss')
    CustomAccuracyMetric(output_transform=lambda out: out).attach(trainer, 'trn_accuracy')
    EpochIteratation(output_transform=lambda out: out).attach(trainer, 'trn_epoch_iteration')

    trainer.run(train_loader, max_epochs=5)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_config', default=json.dumps(dict()), type=str)
    parser.add_argument('--model_config', default=json.dumps(dict(name='net')), type=str)
    parser.add_argument('--train_data_config', default=json.dumps(dict(metadata_path='./data/train_list.mat', images_folder='./data/Images')), type=str)
    parser.add_argument('--test_data_config', default=json.dumps(dict(metadata_path='./data/test_list.mat', images_folder='./data/Images')), type=str)
    parser.add_argument('--seed', default=11, type=int)
    parser.add_argument('--cuda', action='store_true', help='Option to use CUDA.')
    args = parser.parse_args()

    print(args)

    main(args)
