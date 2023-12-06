import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
import evaluate

import argparse
from tqdm.auto import tqdm

from opacus.optimizers.optimizer import DPOptimizer
from opacus.grad_sample import GradSampleModule
from opacus.data_loader import DPDataLoader
from opacus.accountants.utils import get_noise_multiplier

from optimizers.dpzero import DPZero
from transformers_support import forward_swapper


def load_data(dataset, model, batch_size, seed=123):
    data = load_dataset('glue', dataset)
    tokenizer = AutoTokenizer.from_pretrained(model)

    # 'sentence' for glue sst2
    def tokenize_function(examples):
        return tokenizer(examples['sentence'], padding='max_length', truncation=True, max_length=256)    
    tokenized_data = data.map(tokenize_function, batched=True)

    tokenized_data = tokenized_data.remove_columns(['sentence', 'idx'])
    tokenized_data = tokenized_data.rename_column('label', 'labels')
    tokenized_data.set_format('torch')

    # only 100 elements for debugging
    train_data = tokenized_data['train'].shuffle(seed=seed).select(range(100))
    test_data = tokenized_data['validation'].shuffle(seed=seed).select(range(100))

    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    # change to poisson sampling
    train_loader = DPDataLoader.from_data_loader(train_loader)
    test_loader = DataLoader(test_data, batch_size=batch_size)
    return train_loader, test_loader


def load_model(model, device):
    # num_labels=2 for glue sst2
    pretrained_model = AutoModelForSequenceClassification.from_pretrained(model, num_labels=2)
    pretrained_model.to(device)
    # fix the incompatible issue between transformers and opacus
    forward_swapper(pretrained_model)
    return pretrained_model


def train_one_epoch(model, optimizer, train_loader, device, progress, epoch_number, writer):
    model.train()

    for steps, batch in enumerate(train_loader):
        optimizer.zero_grad()

        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)

        loss = outputs.loss
        loss.backward()
        optimizer.step()

        progress.update(1)
        writer.add_scalar('Training Loss', loss.item(), epoch_number * len(train_loader) + steps)


def train_one_epoch_zero(model, optimizer, train_loader, device, progress, epoch_number, writer):
    model.eval()

    with torch.no_grad():
        for steps, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            
            outputs = model(**batch)
            loss = outputs.loss

            optimizer.step(batch)

            progress.update(1)
            writer.add_scalar('Training Loss', loss.item(), epoch_number * len(train_loader) + steps)


def test(model, test_loader, device):
    metric = evaluate.load('accuracy')
    model.eval()
    
    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)

            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            metric.add_batch(predictions=predictions, references=batch['labels'])

    acc = metric.compute()
    return acc['accuracy']


def train(dataset, model, optim, epochs, lr, lam, clip, eps, delta, batch_size, writer):
    train_loader, test_loader = load_data(dataset, model, batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrained_model = load_model(model, device)

    sample_rate = 1 / len(train_loader)
    noise_multiplier = get_noise_multiplier(target_epsilon=eps, target_delta=delta, sample_rate=sample_rate, epochs=epochs)

    if optim == 'SGD':
        # enable per sample gradient computation
        pretrained_model = GradSampleModule(pretrained_model)
        optimizer = torch.optim.SGD(pretrained_model.parameters(), lr=lr)
        optimizer = DPOptimizer(optimizer=optimizer, noise_multiplier=noise_multiplier, max_grad_norm=clip, expected_batch_size=batch_size)

    elif optim == 'DPZero':
        optimizer = DPZero(pretrained_model, lr, lam, clip, noise_multiplier)

    else:
        raise NotImplementedError

    progress_bar = tqdm(range(epochs * len(train_loader)))
    for epoch in range(epochs):
        test_acc = test(pretrained_model, test_loader, device)
        writer.add_scalar('Test Acc', test_acc, epoch)

        if optim == 'SGD':
            train_one_epoch(pretrained_model, optimizer, train_loader, device, progress_bar, epoch, writer)
        elif optim == 'DPZero':
            train_one_epoch_zero(pretrained_model, optimizer, train_loader, device, progress_bar, epoch, writer)
        else:
            raise NotImplementedError
        
    test_acc = test(pretrained_model, test_loader, device)
    writer.add_scalar('Test Acc', test_acc, epochs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DPZero')
    parser.add_argument('--dataset', default='sst2', type=str, help='fine-tuning dataset')
    parser.add_argument('--model', default='mrm8488/bert-tiny-finetuned-sms-spam-detection', type=str, help='pretrained model')
    parser.add_argument('--optim', default='DPZero', type=str, help='optimizer')
    parser.add_argument('--epochs', default=3, type=int, help='number of epochs')
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--lam', default=0.0001, type=float, help='smoothing parameter')
    parser.add_argument('--clip', default=1., type=float, help='clipping threshold')
    parser.add_argument('--eps', default=3., type=float, help='privacy budget')
    parser.add_argument('--delta', default=1e-6, type=float, help='privacy budget')
    parser.add_argument('--batch', default=32, type=int, help='batch size')
    
    args = parser.parse_args()
    writer = SummaryWriter(comment="_model={}_dataset={}_optim={}_lr={}_clip={}_eps={}_delta={}_lam={}_batch={}".format(args.model, args.dataset, args.optim, args.lr, args.clip, args.eps, args.delta, args.lam, args.batch))
    train(dataset=args.dataset, model=args.model, optim=args.optim, epochs=args.epochs, lr=args.lr, lam=args.lam, clip=args.clip, eps=args.eps, delta=args.delta, batch_size=args.batch, writer=writer)