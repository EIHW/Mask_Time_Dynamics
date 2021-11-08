from Data_ETL import DataETL, DataLoader
from Models_MTL import CRNN
import torch
from torch.optim import Adam
import torch.nn as nn
import numpy as np
import socket
from CenterLoss import CenterLoss
from sklearn.metrics import classification_report
from os.path import join

if socket.gethostname() == 'UAU-86505':
    machine = '/home/user/on_gpu/'
else:
    machine = '/home/liushuo/'

import argparse
parser = argparse.ArgumentParser()
machine = '/home/user/on_gpu' if socket.gethostname() == 'UAU-86505' else '/home/liushuo'
task_dir = 'Mask_Supervised_WaveNet_AutoEncoder/seeds/'
parser.add_argument('--train_seeds', type=str, default=join(machine, task_dir, 'train_valid_sel.pkl'))
parser.add_argument('--valid_seeds', type=str, default=join(machine, task_dir, 'valid.pkl'))
parser.add_argument('--test_seeds', type=str, default=join(machine, task_dir + 'test.pkl'))
parser.add_argument('--bs_train', type=int, default=32)
parser.add_argument('--bs_valid', type=int, default=32)
parser.add_argument('--bs_test', type=int, default=32)
parser.add_argument('--with_lld', type=bool, default=False)
parser.add_argument('--AGWN', type=bool, default=False)

parser.add_argument('--conv_net', type=str, default='cnn')
parser.add_argument('--recurrent_net', type=str, default='transformer')
parser.add_argument('--wi_gender', type=bool, default=True)
parser.add_argument('--marker', type=str, default='cnn-tx')

parser.add_argument('--Epochs', type=int, default=500)
parser.add_argument('--optimiser', type=str, default='Adam')
parser.add_argument('--lr', type=float, default=3e-6)
parser.add_argument('--weight_decay', type=float, default=5e-4)

parser.add_argument('--monitor_per_steps', type=int, default=40)
parser.add_argument('--evaluate_per_epochs', type=int, default=1)

parser.add_argument('--center_loss', type=bool, default=False)
parser.add_argument('--alpha', type=str, default=0.0005)

parser.add_argument('--seed_num', type=int, default=0)

args = parser.parse_args()


def set_random_seeds(seed_num):
    torch.manual_seed(seed_num)
    np.random.seed(seed_num)


print(torch.get_rng_state())
# set_random_seeds(args.seed_num)


def to_device(elements, device):
    if isinstance(elements, list):
        elements = [element.to(device) for element in elements]
    else:
        elements = elements.to(device)
    return elements


def print_flags():
    print('--------------------------------------- Flags ------------------------------------')
    for flag in vars(args):
        print('{} : {}'.format(flag, getattr(args, flag)))


def training_settings():
    cost_func_ = nn.CrossEntropyLoss()
    if args.center_loss:
        cost_center_ = CenterLoss(num_classes=2, feat_dim=64)
        optimiser_ = Adam(list(model.parameters()) + list(cost_center_.parameters()), lr=args.lr, weight_decay=args.weight_decay)
        return cost_func_, cost_center_, optimiser_
    else:
        optimiser_ = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        return cost_func_, optimiser_


def report_and_save(pred_aggregate_, gold_aggregate_, current_model, current_epoch, marker):
    save_to = '/nas/staff/data_work/Sure/Mask_Data/saved_models/{}_mask_{}.pkl'.format(marker, current_epoch)
    torch.save(current_model, save_to)
    print('==> Model is saved to {}'.format(save_to))

    assert len(pred_aggregate_) == len(gold_aggregate_)
    print('# samples: {}'.format(len(gold_aggregate_)))
    print(classification_report(gold_aggregate_, pred_aggregate_, target_names=['clear', 'mask']))
    print()


def training(epoch_, steps, train_dl_, cost_funcs_, optimiser_):
    error_train = 0
    counter_train = 0
    correct = 0
    total = 0

    error_train_pred = 0
    error_train_center = 0

    error_train_pred_gender = 0
    correct_gender = 0

    model.train()
    for idx, data_tr in enumerate(train_dl_, 1):
        if args.with_lld:
            batch, lld, label = data_tr[0], data_tr[1], data_tr[2]
            batch, label, lld = to_device([batch, label, lld], device)
            pred, fea = model(batch, lld)
        elif args.wi_gender:
            batch, label, gender = data_tr[0], data_tr[1], data_tr[2]
            batch, label, gender = to_device([batch, label, gender], device)
            pred, fea, pred_gender = model(batch)
        else:
            batch, label = data_tr[0], data_tr[1]
            batch, label = to_device([batch, label], device)
            pred, fea = model(batch)

        if args.center_loss:
            cost_func_, cost_center_ = cost_funcs_[0], cost_funcs_[1]
            loss_pred = cost_func_(pred, label)
            loss_center = args.alpha * cost_center_(fea, label)
            loss = loss_pred + loss_center
        elif args.wi_gender:
            cost_func_ = cost_funcs_
            loss_pred = cost_func_(pred, label)
            loss_pred_gender = cost_func_(pred_gender, gender)
            loss = loss_pred + loss_pred_gender
        else:
            cost_func_ = cost_funcs_
            loss_pred = cost_func_(pred, label)
            loss = loss_pred

        optimiser_.zero_grad()
        loss.backward()
        optimiser_.step()

        steps += 1
        error_train += loss.item()
        if args.center_loss:
            error_train_pred += loss_pred.item()
            error_train_center += loss_center.item()
        elif args.wi_gender:
            error_train_pred += loss_pred.item()
            error_train_pred_gender += loss_pred_gender.item()

        counter_train += 1

        correct += sum(np.argmax(pred.detach().cpu().numpy(), axis=1) == label.detach().cpu().numpy())
        total += len(label.detach().cpu().numpy())

        if args.wi_gender:
            correct_gender += sum(np.argmax(pred_gender.detach().cpu().numpy(), axis=1) == gender.detach().cpu().numpy())

        if idx % args.monitor_per_steps == 0:
            if args.center_loss:
                print('[{} {}] loss: {:.4f} (Pred loss: {:.4f}, Center loss: {:.4f}), acc: {:.4f}'.format(
                    epoch_, idx,
                    error_train / counter_train, error_train_pred / counter_train, error_train_center / counter_train,
                    correct / total))
            elif args.wi_gender:
                print('[{} {}] loss: {:.4f} (Pred loss: {:.4f}, Gender loss: {:.4f}), acc: {:.4f}, gender acc: {:.4f}'.format(
                    epoch_, idx,
                    error_train / counter_train, error_train_pred / counter_train, error_train_pred_gender / counter_train,
                    correct / total, correct_gender/total))
            else:
                print('[{} {}] loss: {:.4f}, acc: {}'.format(
                    epoch_, idx,
                    error_train / counter_train, correct / total))

            error_train = 0
            counter_train = 0
            correct = 0
            total = 0

            error_train_pred = 0
            error_train_center = 0

            error_train_pred_gender = 0
            correct_gender = 0

    return steps


def evaluate(epoch_, test_dl_, cost_funcs_):
    error_test = 0
    counter_test = 0
    correct_test = 0
    total_test = 0

    error_test_pred = 0
    error_test_center = 0

    error_test_pred_gender = 0
    correct_test_gender = 0

    pred_aggregate = []
    gold_aggregate = []

    model.eval()
    for idx_te, data_te in enumerate(test_dl_, 1):
        if args.with_lld:
            batch_te, lld_te, label_te = data_te[0], data_te[1], data_te[2]
            batch_te, label_te, lld_te = to_device([batch_te, label_te, lld_te], device)
            pred_te, fea_te = model(batch_te, lld_te)
        elif args.wi_gender:
            batch_te, label_te, gender_te = data_te[0], data_te[1], data_te[2]
            batch_te, label_te, gender_te = to_device([batch_te, label_te, gender_te], device)
            pred_te, fea_te, pred_gender_te = model(batch_te)
        else:
            batch_te, label_te, gender_te = data_te[0], data_te[1], data_te[2]
            batch_te, label_te, gender_te = to_device([batch_te, label_te, gender_te], device)
            pred_te, fea_te = model(batch_te)

        if args.center_loss:
            cost_func, cost_center = cost_funcs_[0], cost_funcs_[1]
            loss_pred_te = cost_func(pred_te, label_te)
            loss_center_te = args.alpha * cost_center(fea_te, label_te)
            loss_te = loss_pred_te + loss_center_te
        elif args.wi_gender:
            cost_func = cost_funcs_
            loss_pred_te = cost_func(pred_te, label_te)
            loss_pred_gender_te = cost_func(pred_gender_te, gender_te)
            loss_te = loss_pred_te + loss_pred_gender_te
        else:
            cost_func = cost_funcs_
            loss_pred_te = cost_func(pred_te, label_te)
            loss_te = loss_pred_te

        error_test += loss_te.item()
        if args.center_loss:
            error_test_pred += loss_pred_te.item()
            error_test_center += loss_center_te.item()
        elif args.wi_gender:
            error_test_pred += loss_pred_te.item()
            error_test_pred_gender += loss_pred_gender_te.item()
        counter_test += 1

        correct_test += sum(np.argmax(pred_te.detach().cpu().numpy(), axis=1) == label_te.detach().cpu().numpy())
        total_test += len(label_te.detach().cpu().numpy())
        if args.wi_gender:
            correct_test_gender += sum(np.argmax(pred_gender_te.detach().cpu().numpy(), axis=1) == gender_te.detach().cpu().numpy())

        pred_aggregate.extend(np.argmax(pred_te.detach().cpu().numpy(), axis=1).tolist())
        gold_aggregate.extend(label_te.detach().cpu().numpy().tolist())

    if args.center_loss:
        print('=> [{}] loss: {:.4f} (Pred loss: {:.4f}, Center loss: {:.4f}), test_acc: {:.4f}'.format(
            epoch_,
            error_test / counter_test, error_test_pred / counter_test, error_test_center / counter_test,
            correct_test / total_test))
    elif args.wi_gender:
        print('=> [{}] loss: {:.4f} (Pred loss: {:.4f}, Gender loss: {:.4f}), test_acc: {:.4f}, test_gender_acc {:.4f}'.format(
            epoch_,
            error_test / counter_test, error_test_pred / counter_test, error_test_pred_gender / counter_test,
            correct_test / total_test, correct_test_gender/total_test))
    else:
        print('=> [{}] loss: {:.4f}, test_acc: {:.4f}'.format(
            epoch_,
            error_test / counter_test, correct_test / total_test))
    print()

    if correct_test / total_test > 0.785:
        report_and_save(pred_aggregate, gold_aggregate, model, epoch_, args.marker)


if __name__ == '__main__':
    train_ds = DataETL(args.train_seeds, mother_wav=None, lld=False, agwn=args.AGWN, distributed=args.conv_net.startswith('td-'))
    train_dl = DataLoader(train_ds, batch_size=args.bs_train, shuffle=True, num_workers=16, drop_last=True)

    test_ds = DataETL(args.test_seeds, mother_wav=None, lld=args.with_lld, distributed=args.conv_net.startswith('td-'))
    test_dl = DataLoader(test_ds, batch_size=args.bs_test, shuffle=False, num_workers=16)

    print_flags()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = CRNN(conv_net=args.conv_net, recurrent_net=args.recurrent_net, wi_gender=args.wi_gender)
    to_device(model, device)

    if args.center_loss:
        cost_func, cost_center, optimiser = training_settings()
        cost_funcs = [cost_func, cost_center]
    else:
        cost_func, optimiser = training_settings()
        cost_funcs = cost_func

    print('--------------------------------- Start Training -------------------------------')
    global_steps = 0
    for epoch in range(1, args.Epochs + 1):
        global_steps = training(epoch, global_steps, train_dl, cost_funcs, optimiser)

        if epoch % args.evaluate_per_epochs == 0:
            evaluate(epoch, test_dl, cost_funcs)
