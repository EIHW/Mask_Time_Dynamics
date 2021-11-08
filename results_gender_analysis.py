import pickle as p
import csv
results_pkl = '/nas/staff/data_work/Sure/Mask_Data/saved_models/cnn_pe_tx/results.pkl'
from sklearn.metrics import classification_report

with open(results_pkl, 'rb') as pkl_file:
    results = p.load(pkl_file)
genders, pred, gold = results[0], results[1], results[2]
genders = [gen.item() for gen in genders]
# 'm': 0, 'f': 1

meta_info = '/nas/staff/data_work/Adria/ComParE_masks/Data/Mask_labels_confidential.csv'
gender_dict = {}
with open(meta_info) as csv_file:
    lines = csv.reader(csv_file)
    next(lines)

    for line in lines:
        file_id, label = line[0], line[1]
        gender_dict[file_id] = line[3].split('_')[0]

pred_m, gold_m, pred_f, gold_f = [], [], [], []
for gender, pr, go in zip(genders, pred, gold):
    # gender = gender_dict[pa.split('/')[-1]]
    if gender == 0:
        pred_m.append(pr)
        gold_m.append(go)
    else:
        pred_f.append(pr)
        gold_f.append(go)

print(classification_report(gold_m, pred_m, target_names=['clear', 'mask']))
print(classification_report(gold_f, pred_f, target_names=['clear', 'mask']))

results_pkl = '/nas/staff/data_work/Sure/Mask_Data/saved_models/cnn_lstm_female/results.pkl'
with open(results_pkl, 'rb') as pkl_file:
    results = p.load(pkl_file)
genders, pred, gold = results[0], results[1], results[2]
genders = [gen.item() for gen in genders]
# 'm': 0, 'f': 1

meta_info = '/nas/staff/data_work/Adria/ComParE_masks/Data/Mask_labels_confidential.csv'
gender_dict = {}
with open(meta_info) as csv_file:
    lines = csv.reader(csv_file)
    next(lines)

    for line in lines:
        file_id, label = line[0], line[1]
        gender_dict[file_id] = line[3].split('_')[0]

pred_f, gold_f = [], []
for gender, pr, go in zip(genders, pred, gold):
    # gender = gender_dict[pa.split('/')[-1]]
    pred_f.append(pr)
    gold_f.append(go)

print(classification_report(gold_f, pred_f, target_names=['clear', 'mask']))


results_pkl = '/nas/staff/data_work/Sure/Mask_Data/saved_models/cnn_tx_male/results.pkl'
with open(results_pkl, 'rb') as pkl_file:
    results = p.load(pkl_file)
genders, pred, gold = results[0], results[1], results[2]
genders = [gen.item() for gen in genders]
# 'm': 0, 'f': 1

meta_info = '/nas/staff/data_work/Adria/ComParE_masks/Data/Mask_labels_confidential.csv'
gender_dict = {}
with open(meta_info) as csv_file:
    lines = csv.reader(csv_file)
    next(lines)

    for line in lines:
        file_id, label = line[0], line[1]
        gender_dict[file_id] = line[3].split('_')[0]

pred_m, gold_m = [], []
for gender, pr, go in zip(genders, pred, gold):
    # gender = gender_dict[pa.split('/')[-1]]
    pred_m.append(pr)
    gold_m.append(go)
print(classification_report(gold_m, pred_m, target_names=['clear', 'mask']))

gold = []
pred = []
gold.extend(gold_m)
gold.extend(gold_f)
pred.extend(pred_m)
pred.extend(pred_f)

print(classification_report(gold, pred, target_names=['clear', 'mask']))

