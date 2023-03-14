from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics

y_test = pd.read_csv("y_test.csv").to_numpy()
y_pred = pd.read_csv("y_pred.csv").to_numpy()

fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)

plt.plot(fpr, tpr, label = "ROC")
plt.plot([0, 1], [0, 1], color="darkblue", linestyle="--")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.show()
