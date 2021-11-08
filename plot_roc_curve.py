import pickle as p
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

with open('/nas/staff/data_work/Sure/Mask_Data/saved_models/optimised_models/cnn_pe_tx/results.pkl', 'rb') \
        as results_pkl:
    cnn_pe_tx_results = p.load(results_pkl)

with open('/nas/staff/data_work/Sure/Mask_Data/saved_models/optimised_models/cnn_tx/results.pkl', 'rb') \
        as results_pkl:
    cnn_tx_results = p.load(results_pkl)

with open('/nas/staff/data_work/Sure/Mask_Data/saved_models/optimised_models/cnn_lstm_attn/results.pkl', 'rb') \
        as results_pkl:
    cnn_lstm_attn_results = p.load(results_pkl)

with open('/nas/staff/data_work/Sure/Mask_Data/saved_models/optimised_models/cnn_lstm/results.pkl', 'rb') \
        as results_pkl:
    cnn_lstm_results = p.load(results_pkl)

with open('/nas/staff/data_work/Sure/Mask_Data/saved_models/optimised_models/cnn/results.pkl', 'rb') \
        as results_pkl:
    cnn_results = p.load(results_pkl)

pred_aggregate5, gold_aggregate5, scores_aggregate5 = cnn_pe_tx_results[1:]
pred_aggregate4, gold_aggregate4, scores_aggregate4 = cnn_tx_results[1:]
pred_aggregate3, gold_aggregate3, scores_aggregate3 = cnn_lstm_attn_results[1:]
pred_aggregate2, gold_aggregate2, scores_aggregate2 = cnn_lstm_results[1:]
pred_aggregate1, gold_aggregate1, scores_aggregate1 = cnn_results[1:]

fpr5, tpr5, thres5 = roc_curve(gold_aggregate5, scores_aggregate5)
auc_scores5 = roc_auc_score(gold_aggregate5, scores_aggregate5)

fpr4, tpr4, thres4 = roc_curve(gold_aggregate4, scores_aggregate4)
auc_scores4 = roc_auc_score(gold_aggregate4, scores_aggregate4)

fpr3, tpr3, thres3 = roc_curve(gold_aggregate3, scores_aggregate3)
auc_scores3 = roc_auc_score(gold_aggregate3, scores_aggregate3)

fpr2, tpr2, thres2 = roc_curve(gold_aggregate2, scores_aggregate2)
auc_scores2 = roc_auc_score(gold_aggregate2, scores_aggregate2)

fpr1, tpr1, thres1 = roc_curve(gold_aggregate1, scores_aggregate1)
auc_scores1 = roc_auc_score(gold_aggregate1, scores_aggregate1)

x_dash = range(len(fpr1))
y_dash = range(len(tpr1))

lw = 1.5
plt.figure(figsize=(7, 6))
plt.plot(fpr5, tpr5, 'green', lw=lw, label='Convolutional Transformer w/ PE (AUC={:.3f})'.format(auc_scores5))
plt.plot(fpr4, tpr4, 'b--', lw=lw, label='Convolutional Transformer wo PE (AUC={:.3f})'.format(auc_scores4))
plt.plot(fpr3, tpr3, 'purple', lw=lw, label='Attentive ConvLSTM (AUC={:.3f})'.format(auc_scores3))
plt.plot(fpr2, tpr2, 'orange', linestyle='--', lw=lw, label='ConvLSTM (AUC={:.3f})'.format(auc_scores2))
plt.plot(fpr1, tpr1, color='red', linestyle='-.', lw=lw, label='CNN (AUC={:.3f})'.format(auc_scores1))
plt.plot(x_dash, y_dash, linestyle='--', lw=lw)
plt.xlabel('False Positive Rate', fontsize=14)
plt.ylabel('True Positive Rate', fontsize=14)
plt.title('Receiver Operating Characteristic Curve', fontsize=14)
plt.legend(loc='lower right')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.show()
