
import pandas as pd
import xgboost as xgb
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
import warnings
warnings.simplefilter(action='ignore')




if __name__ == '__main__':
    FILE = 'data/features_final.csv'
    EXPERIMENT_FOLDER = 'experiments/train'
    NROUNDS = 1000
    NFOLDS = 10

    path_exp = Path(EXPERIMENT_FOLDER)
    current_exp_folder = path_exp/f'exp_{datetime.now().strftime("%d-%b-%Y_%H_%M_%S")}'

    df = pd.read_csv(FILE)
    x = df.iloc[:, :-2]
    y = df.iloc[:, -2]
    normalizer = Normalizer()
    x = normalizer.fit_transform(x)
    x = pd.DataFrame(x, columns=df.columns[:-2])
    x_train, x_val, y_train, y_val = train_test_split(x, y, random_state=42, test_size=0.15)

    params = {
        'objective': 'binary:logistic',
        'max_depth': 6,
        'eta': 0.1,
        'eval_metric': 'auc',
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'lambda': 1,
        'tree_method': 'auto',
    }
    dtrain = xgb.DMatrix(x_train, label=y_train)
    dval = xgb.DMatrix(x_val, label=y_val)
    watchlist = [(dtrain, 'train'), (dval, 'val')]

####################################
    # Train and feature importance
####################################
    bst = xgb.train(params=params, dtrain=dtrain, num_boost_round=NROUNDS, evals=watchlist, verbose_eval=5)
    gain = bst.get_score(fmap='', importance_type='gain')
    importance = pd.DataFrame(list(gain.items()), columns=['variable', 'importance']).set_index('variable')
    importance.sort_values('importance', ascending=False, inplace=True)
    importance.plot(kind='bar')
    plt.show(block=False)

    preds_probs = bst.predict(dval)
    preds = [1 if x > 0.5 else 0 for x in preds_probs]
    roc_auc = roc_auc_score(y_val, preds)
    print(f'{roc_auc = }')


    ####################################
    # Cross-Validation
    ####################################
    dall = xgb.DMatrix(x, label=y)
    cvres = xgb.cv(params=params,
                   dtrain=dall,
                   num_boost_round=NROUNDS,
                   nfold=NFOLDS,
                   metrics='auc',
                   stratified=True,
                   verbose_eval=5
                   )

    cvres[['train-auc-mean', 'test-auc-mean']].plot()
    plt.show()


