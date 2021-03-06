import argparse
import importlib
import json
import os
import sys

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint, global_step_from_engine
from ignite.metrics import RunningAverage

from data import *
from ignite_additions import *
from presets import *


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
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    train_data_config = json.loads(args.train_data_config)
    eval_data_config = json.loads(args.eval_data_config)

    train_dataset = CustomDataset(
        metadata_path=train_data_config['metadata_path'],
        images_folder=train_data_config['images_folder'],
        transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset,
        batch_size=train_data_config['batch_size'],
        shuffle=True,
        num_workers=train_data_config['num_workers'])

    eval_dataset = CustomDataset(
        metadata_path=eval_data_config['metadata_path'],
        images_folder=eval_data_config['images_folder'],
        transform=transform)
    eval_loader = torch.utils.data.DataLoader(eval_dataset,
        batch_size=eval_data_config['batch_size'],
        shuffle=False,
        num_workers=eval_data_config['num_workers'])

    train_config = json.loads(args.train_config)

    context = {}
    context['num_classes'] = 120
    context['device'] = device
    model = build_model(args, context).to(device)
    opt = optim.Adam(model.parameters(), lr=train_config['lr'])

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
    def start_training(engine):
        print('Training started!')

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_train(engine):
        log_msg = f"[train] epoch: {engine.state.epoch}"
        log_msg += f" | epoch iteration: {engine.state.metrics['trn_epoch_iteration']} / {engine.state.epoch_length}"
        log_msg += f" | total iteration: {engine.state.iteration}"
        log_msg += f" | loss: {engine.state.metrics['trn_loss']:.3f}"
        log_msg += f" | accuracy: {engine.state.metrics['trn_accuracy']:.3f}"
        print(log_msg)

    @trainer.on(Events.EPOCH_COMPLETED)
    def run_eval(engine):
        outputs = []

        for batch in tqdm(eval_loader, desc='Eval', disable=not args.progress):
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = nn.CrossEntropyLoss()(logits, labels)

            correct = (logits.max(-1)[1] == labels).sum().item()
            total = labels.shape[-1]

            out = {}
            out['loss'] = loss.item()
            out['correct'] = correct
            out['total'] = total

            outputs.append(out)

        def _avg(outputs, key):
            vals = []
            for x in outputs:
                vals += [x[key]] * x['total']
            return torch.mean(torch.tensor(vals, dtype=torch.float)).item()

        correct = sum([x['correct'] for x in outputs])
        total = sum([x['total'] for x in outputs])
        accuracy = correct / total
        loss = _avg(outputs, 'loss')

        engine.state.metrics['eval_loss'] = loss
        engine.state.metrics['eval_accuracy'] = accuracy

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_eval(engine):
        log_msg = f"[eval] epoch: {engine.state.epoch}"
        log_msg += f" | epoch iteration: {engine.state.metrics['trn_epoch_iteration']} / {engine.state.epoch_length}"
        log_msg += f" | total iteration: {engine.state.iteration}"
        log_msg += f" | loss: {engine.state.metrics['eval_loss']:.3f}"
        log_msg += f" | accuracy: {engine.state.metrics['eval_accuracy']:.3f}"
        print(log_msg)

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

    pbar = ProgressBar(disable=not args.progress)
    pbar.attach(trainer)

    RunningAverage(output_transform=lambda out: out['loss']).attach(trainer, 'trn_loss')
    CustomAccuracyMetric(output_transform=lambda out: out).attach(trainer, 'trn_accuracy')
    EpochIteratation(output_transform=lambda out: out).attach(trainer, 'trn_epoch_iteration')

    trainer.run(train_loader, max_epochs=train_config.get('max_epochs', 5))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_config', default=None, type=str)
    parser.add_argument('--model_config', default=None, type=str)
    parser.add_argument('--train_data_config', default=None, type=str)
    parser.add_argument('--eval_data_config', default=None, type=str)
    parser.add_argument('--seed', default=11, type=int)
    parser.add_argument('--cuda', action='store_true', help='If True, run on GPU.')
    parser.add_argument('--progress', action='store_true', help='If True, show progress bar.')
    parser.add_argument('--preset', default=None, type=str)
    args = parser.parse_args()

    if args.preset is not None:
        if args.preset == 'default':
            args.train_config = json.dumps(default_train_config)
            args.model_config = json.dumps(default_model_config)
            args.train_data_config = json.dumps(default_train_data_config)
            args.eval_data_config = json.dumps(default_eval_data_config)

    print(args)

    main(args)
