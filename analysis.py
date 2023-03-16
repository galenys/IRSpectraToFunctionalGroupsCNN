from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics

# get values from csv
y_test = pd.read_csv("y_test.csv").to_numpy(dtype="float32")
y_pred = pd.read_csv("y_pred.csv").to_numpy(dtype="float32")

# sklearn shenanigans
for i in range(y_test.shape[1]):
    a , b, threshold = metrics.roc_curve(np.transpose(y_test[1:, i]), np.transpose(y_pred[1:, i]))
    auc = metrics.auc(a, b)
    f1 = metrics.f1_score(y_test[1:, i], y_pred[1:, i].round(), average = "micro")
    plt.plot(a, b, label = "Class {0:.0f}, AUC = {1:.3f}, F1 = {2:0.3f}".format(i, auc, f1))

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.legend()
plt.savefig("roc_curve_multi.png")
plt.show()
