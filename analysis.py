from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics

# get values from csv
y_test = pd.read_csv("y_test.csv").to_numpy(dtype="float32")
y_pred = pd.read_csv("y_pred.csv").to_numpy(dtype="float32")

# sklearn shenanigans
fpr = np.zeros((y_test.shape[0]-1, y_test.shape[1]))
tpr = np.zeros((y_test.shape[0]-1, y_test.shape[1]))
# fpr = []
# tpr = []

for i in range(y_test.shape[1]-1):
    a , b, threshold = metrics.roc_curve(np.transpose(y_test[1:, i]), np.transpose(y_pred[1:, i]))
    # fpr.append(a)
    print(a)
    print("-----------------")
    # tpr.append(b)
    # fpr[:, i] = a
    # tpr[:, i] = b

# plotting
# fig, axs = plt.subplots(1, 2)
# axs[0, 0].plot(fpr(0), tpr(0))
# axs[0, 0].set_title('Axis [0, 0]')
# axs[0, 1].plot(fpr(1), tpr(1), 'tab:orange')
# axs[0, 1].set_title('Axis [0, 1]')

# axs[1, 0].plot(x, -y, 'tab:green')
# axs[1, 0].set_title('Axis [1, 0]')
# axs[1, 1].plot(x, -y, 'tab:red')
# axs[1, 1].set_title('Axis [1, 1]')

# for ax in axs.flat:
    # ax.set(xlabel='x-label', ylabel='y-label')





# auc = metrics.auc(fpr, tpr)

# plotting
# plt.plot(fpr, tpr, label = "ROC")
# plt.plot([0, 1], [0, 1], color="darkblue", linestyle="--")
# plt.xlabel("False Positive Rate")
# plt.ylabel("True Positive Rate")
# plt.title("Receiver Operating Characteristic (ROC) Curve, with {0:.3f} AUC".format(auc))
# plt.legend()
# # plt.show()

# # f1 score
# y_pred = y_pred.round()
# f1 = metrics.f1_score(y_test, y_pred, average = "micro")
# print(f"F1 score: {f1}")
