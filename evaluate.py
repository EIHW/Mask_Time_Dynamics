import torch
import torch.nn as nn
import numpy as np
import socket
import pickle as p
from Data_ETL import DataETL, DataLoader
from CenterLoss import CenterLoss
from os.path import join
from sklearn.metrics import recall_score, precision_score, f1_score, classification_report, roc_auc_score, average_precision_score

if socket.gethostname() == 'UAU-86505':
    machine = '/home/user/on_gpu/'
else:
    machine = '/home/liushuo/'

import argparse
parser = argparse.ArgumentParser()
machine = '/home/user/on_gpu' if socket.gethostname() == 'UAU-86505' else '/home/liushuo'
task_dir = 'Mask_Supervised_WaveNet_AutoEncoder/seeds/'
parser.add_argument('--center_loss', type=bool, default=False)
parser.add_argument('--alpha', type=str, default=0.0005)
parser.add_argument('--with_lld', type=bool, default=False)
parser.add_argument('--eval_seeds', type=str, default=join(machine, task_dir + 'test.pkl'))
parser.add_argument('--bs_eval', type=int, default=32)
parser.add_argument('--gender', type=str, default='both')
parser.add_argument('--models_save_to', type=str, default='./saved_models')

args = parser.parse_args()


def perf_measure(y_actual, y_hat, y_scores=None):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)):
        if y_actual[i] == y_hat[i] == 1:
           TP += 1
        if y_hat[i] == 1 and y_actual[i] != y_hat[i]:
           FP += 1
        if y_actual[i] == y_hat[i] == 0:
           TN += 1
        if y_hat[i] == 0 and y_actual[i] != y_hat[i]:
           FN += 1

    if y_scores is not None:
        roc_auc = roc_auc_score(y_actual, y_scores)
        avg_prc = average_precision_score(y_actual, y_scores)
        return (TP, FP, TN, FN), (roc_auc, avg_prc)
    else:
        return TP, FP, TN, FN


def compute_acc_rec_pre_f1_sensitivity_specificity(measure):
    if len(measure) == 2:
        (TP, FP, TN, FN), (ROC_AUC, AVG_PRC) = measure
    else:
        TP, FP, TN, FN = measure
    acc = (TP + TN) / (TP + FP + TN + FN)
    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)
    uar = (sensitivity + specificity) / 2
    recall = TP / (TP + FN)
    recall_reverse = TN / (TN + FP)
    precision = TP / (TP + FP)
    precision_reverse = TN / (TN + FN)
    uap = (precision + precision_reverse) / 2
    f1 = 2 * (recall * precision) / (recall + precision)
    f1_reverse = 2 * (recall_reverse * precision_reverse) / (recall_reverse + precision_reverse)
    uf1 = (f1 + f1_reverse) / 2

    mcc = (TP * TN - FP * FN) / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))

    print('-----------------------------------------------------------------------')
    print('acc: {}'.format(acc))
    print('sensitivity: {}, specificity: {}'.format(sensitivity, specificity))
    print('recall: {}, precision: {}, f1: {}'.format(recall, precision, f1))
    print('uar: {}, uap: {}, uf1: {}'.format(uar, uap, uf1))
    print('mcc: {}'.format(mcc))
    if len(measure) == 2:
        print('roc_auc: {}'.format(ROC_AUC))
        print('avg_precision_score: {}'.format(AVG_PRC))
    return acc, sensitivity, specificity, recall, precision, f1, uar, uap, uf1, mcc


def to_device(elements, device):
    if isinstance(elements, list):
        elements = [element.to(device) for element in elements]
    else:
        elements = elements.to(device)
    return elements


def evaluate(epoch_, eval_dl_, cost_funcs_):
    error_eval = 0
    counter_eval = 0
    correct_eval = 0
    total_eval = 0

    error_eval_pred = 0
    error_eval_center = 0

    pred_aggregate = []
    gold_aggregate = []
    seed_aggregate = []
    scores_aggregate = []

    model.eval()
    for idx_te, data_te in enumerate(eval_dl_, 1):
        if args.with_lld:
            batch_te, lld_te, label_te = data_te[0], data_te[1], data_te[2]
            batch_te, label_te, lld_te = to_device([batch_te, label_te, lld_te], device)
            pred_te, fea_te = model(batch_te, lld_te)
        else:
            batch_te, label_te, gender_te = data_te[0], data_te[1], data_te[2]
            batch_te, label_te = to_device([batch_te, label_te], device)
            pred_te, fea_te = model(batch_te)

        if args.center_loss:
            cost_func, cost_center = cost_funcs_[0], cost_funcs_[1]
            loss_pred_te = cost_func(pred_te, label_te)
            loss_center_te = args.alpha * cost_center(fea_te, label_te)
            loss_te = loss_pred_te + loss_center_te
        else:
            cost_func = cost_funcs_
            loss_pred_te = cost_func(pred_te, label_te)
            loss_te = loss_pred_te

        error_eval += loss_te.item()
        if args.center_loss:
            error_eval_pred += loss_pred_te.item()
            error_eval_center += loss_center_te.item()
        counter_eval += 1

        correct_eval += sum(np.argmax(pred_te.detach().cpu().numpy(), axis=1) == label_te.detach().cpu().numpy())
        total_eval += len(label_te.detach().cpu().numpy())

        pred_aggregate.extend(np.argmax(pred_te.detach().cpu().numpy(), axis=1).tolist())
        gold_aggregate.extend(label_te.detach().cpu().numpy().tolist())
        seed_aggregate.extend(gender_te)  # 'm': 0, 'f':1
        scores_aggregate.extend(torch.softmax(pred_te, dim=1).detach().cpu().numpy()[:, 1].tolist())

    if args.center_loss:
        print('=> [{}] loss: {:.4f} (Pred loss: {:.4f}, Center loss: {:.4f}), eval_acc: {}'.format(
            epoch_,
            error_eval / counter_eval, error_eval_pred / counter_eval, error_eval_center / counter_eval,
            correct_eval / total_eval))
    else:
        print('=> [{}] loss: {:.4f}, eval_acc: {}'.format(
            epoch_,
            error_eval / counter_eval, correct_eval / total_eval))

    print('==> Model is saved to {}'.format(save_to))

    assert len(pred_aggregate) == len(gold_aggregate)
    print('# samples: {}'.format(len(gold_aggregate)))
    print(classification_report(gold_aggregate, pred_aggregate, target_names=['clear', 'mask']))
    print()
    save_results = [seed_aggregate, pred_aggregate, gold_aggregate, scores_aggregate]

    measure = perf_measure(gold_aggregate, pred_aggregate, scores_aggregate)
    compute_acc_rec_pre_f1_sensitivity_specificity(measure)
    with open(join(args.models_save_to, 'results.pkl'), 'wb') \
            as results_pkl:
        p.dump(save_results, results_pkl)


def eval_settings():
    cost_func_ = nn.CrossEntropyLoss()
    if args.center_loss:
        cost_center_ = CenterLoss(num_classes=2, feat_dim=64)
        return cost_func_, cost_center_
    else:
        return cost_func_


if __name__ == '__main__':
    marker = 'cnn_tx'
    current_epoch = 149

    save_to = join(args.models_save_to, '{}_mask_{}.pkl'.format(marker, current_epoch))
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = torch.load(save_to, map_location=device)
    to_device(model, device)

    if args.center_loss:
        cost_func, cost_center = eval_settings()
        cost_funcs = [cost_func, cost_center]
    else:
        cost_func = eval_settings()
        cost_funcs = cost_func

    eval_ds = DataETL(args.eval_seeds, mother_wav=None, lld=args.with_lld, distributed=False, gender='both')
    eval_dl = DataLoader(eval_ds, batch_size=args.bs_eval, shuffle=False, num_workers=16)

    evaluate(current_epoch, eval_dl, cost_funcs)
