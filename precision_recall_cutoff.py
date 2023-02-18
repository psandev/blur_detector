import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer

bst = xgb.Booster()
bst.load_model('models/best_model.json')


if __name__ == '__main__':
    FILE = 'data/features_final.csv'

    df = pd.read_csv(FILE)
    x = df.iloc[:, :-2]
    y = df.iloc[:, -2]
    normalizer = Normalizer()
    x = normalizer.fit_transform(x)
    x = pd.DataFrame(x, columns=df.columns[:-2])
    x_train, x_val, y_train, y_val = train_test_split(x, y,
                                                      random_state=42,
                                                      test_size=0.15,
                                                      stratify=y.values
                                                      )
    dval = xgb.DMatrix(x_val, label=y_val)
    preds_probs = bst.predict(dval)

    percentiles = np.arange(1, 100, 5)
    results = []
    for i, percentile in enumerate(percentiles):
        cutoff = np.percentile(preds_probs, percentile)
        preds = np.where(preds_probs >= cutoff, 1, 0)
        cm = confusion_matrix(y_val, preds, labels=[1, 0]).copy()
        tp =cm[0,0]
        precision = tp/cm[:, 0].sum()
        recall = tp/cm[0, :].sum()
        accuracy = cm.diagonal().sum()/cm.sum()
        f1_score = 2 * (precision * recall)/( precision + recall)
        res = dict( percentile=percentile,
                    cutoff=cutoff,
                    precision=precision,
                    recall=recall,
                    f1_score=f1_score,
                    accuracy=accuracy
                    )
        results.append(res)
    results = pd.DataFrame(results)
    results.set_index('percentile', inplace=True)
    results[['precision', 'recall']].plot()
    plt.show()
