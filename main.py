import sys
import torch
import torch.nn as nn
import numpy as np
import random
import os
import time
import warnings
import argparse
from tqdm import tqdm

from data_provider.data_factory import data_provider
from utils.tools import EarlyStopping, adjust_learning_rate, vali, test
from models.SPA_LLM import SPA_LLM

torch.set_num_threads(4)
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='SPA-LLM Training')

# Dataset
parser.add_argument('--root_path', type=str, default='./datasets/ETT-small/')
parser.add_argument('--data_path', type=str, default='ETTh1.csv')
parser.add_argument('--data', type=str, default='ETTh1')
parser.add_argument('--features', type=str, default='M')
parser.add_argument('--freq', type=str, default='h')
parser.add_argument('--target', type=str, default='OT')
parser.add_argument('--embed', type=str, default='none')
parser.add_argument('--percent', type=int, default=100)
parser.add_argument('--max_len', type=int, default=-1)
parser.add_argument('--multi', type=int, default=1)
parser.add_argument('--train_shuffle', type=int, default=1)
parser.add_argument('--drop_last', type=bool, default=True)

# Input/Output
parser.add_argument('--seq_len', type=int, default=512)
parser.add_argument('--pred_len', type=int, default=96)
parser.add_argument('--label_len', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--enc_in', type=int, default=7)

# Model
parser.add_argument('--model', type=str, default='SPA_LLM')
parser.add_argument('--llm_layers', type=int, default=4)
parser.add_argument('--d_model', type=int, default=768)
parser.add_argument('--d_ff', type=int, default=256)
parser.add_argument('--period_len', type=int, default=24)
parser.add_argument('--decomp_kernel', type=int, default=25)
parser.add_argument('--pp', type=int, default=24, help='Number of compressed phase tokens P-prime')
parser.add_argument('--pool_size', type=int, default=512, help='Distilled vocabulary size V-prime')
parser.add_argument('--rank', type=int, default=32, help='Low-rank dimension m')
parser.add_argument('--a1', type=float, default=0.1, help='Phase memory injection scale')
parser.add_argument('--a2', type=float, default=0.1, help='Phase bias injection scale')

# Training
parser.add_argument('--seed', type=int, default=22)
parser.add_argument('--itr', type=int, default=1)
parser.add_argument('--train_epochs', type=int, default=100)
parser.add_argument('--patience', type=int, default=10)
parser.add_argument('--lradj', type=str, default='COS')
parser.add_argument('--decay_fac', type=float, default=0.9)
parser.add_argument('--learning_rate', type=float, default=0.0001)
parser.add_argument('--eta_min', type=float, default=1e-8)
parser.add_argument('--loss_func', type=str, default='mse')
parser.add_argument('--tmax', type=int, default=20)
parser.add_argument('--num_workers', type=int, default=0)

# Project
parser.add_argument('--checkpoints', type=str, default='./checkpoints/')
parser.add_argument('--gpu', type=int, default=0)

args = parser.parse_args()

fix_seed = args.seed
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

args.data_name = args.data_path.split('.')[0]
device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')


for ii in range(args.itr):
    setting = 'v{}_r{}_dn{}_pl{}'.format(
        'SPA_LLM',
        'train',
        args.data_name,
        args.pred_len
    )

    import datetime
    current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    args.log_path = os.path.join(args.checkpoints, setting, f'seed-{args.seed}-itr-{ii}', current_time, 'log')
    args.pth_path = os.path.join(args.checkpoints, setting, f'seed-{args.seed}-itr-{ii}', current_time, 'pth')

    os.makedirs(args.log_path, exist_ok=True)
    os.makedirs(args.pth_path, exist_ok=True)

    # Data
    train_data, train_loader = data_provider(args, 'train', drop_last_test=args.drop_last, train_shuffle=True if args.train_shuffle else False)
    vali_data, vali_loader = data_provider(args, 'val', drop_last_test=args.drop_last)
    test_data, test_loader = data_provider(args, 'test', drop_last_test=args.drop_last)

    train_steps = len(train_loader)

    # Model
    model = SPA_LLM(args)
    model = model.to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("trainable params:", trainable / 1e6, "M")

    # Optimizer
    model_optim = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    early_stopping = EarlyStopping(patience=args.patience, verbose=True)
    criterion = nn.MSELoss()

    if args.lradj == 'COS':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=args.tmax, eta_min=args.eta_min)

    for epoch in range(args.train_epochs):
        train_loss = []

        model.train()
        epoch_time = time.time()

        for i, (batch_x, batch_y, batch_x_mark, _) in tqdm(enumerate(train_loader)):
            model_optim.zero_grad()

            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            batch_x_mark = batch_x_mark.float().to(device)

            outputs = model(batch_x, batch_x_mark)
            outputs = outputs[:, -args.pred_len:, :]
            batch_y = batch_y[:, -args.pred_len:, :].to(device)

            loss = criterion(outputs, batch_y)
            train_loss.append(loss.item())

            loss.backward()
            model_optim.step()

        log = "Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time)
        print(log)
        with open(os.path.join(args.log_path, 'log.txt'), 'a') as f:
            f.write(log + '\n')

        train_loss = np.average(train_loss)
        vali_loss = vali(model, vali_data, vali_loader, criterion, args, device, ii)

        if args.lradj == 'COS':
            scheduler.step()
            log = "lr = {:.10f}".format(model_optim.param_groups[0]['lr'])
            with open(os.path.join(args.log_path, 'log.txt'), 'a') as f:
                f.write(log + '\n')
        else:
            adjust_learning_rate(model_optim, epoch + 1, args)

        early_stopping(vali_loss, model, args.pth_path)
        if early_stopping.early_stop:
            print("Early stopping")
            with open(os.path.join(args.log_path, 'log.txt'), 'a') as f:
                f.write("Early stopping\n")
            break

    # Load best model and test
    best_model_path = args.pth_path + '/' + 'checkpoint.pth'
    model.load_state_dict(torch.load(best_model_path))
    print("------------------------------------")

    mse, mae, rmse, smape = test(model, test_data, test_loader, args, device)
    print('mae:{:.8f}, mse:{:.8f}, rmse:{:.8f}, smape:{:.8f}'.format(mae, mse, rmse, smape))

    log = "mse = {:.4f}, mae = {:.4f}, rmse = {:.4f}, smape = {:.4f}".format(mse, mae, rmse, smape)
    with open(os.path.join(args.log_path, 'log.txt'), 'a') as f:
        f.write(log + '\n')
