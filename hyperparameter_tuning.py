import optuna
import xgboost as xgb
import pandas as pd
import warnings
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split
from functools import partial
warnings.simplefilter(action='ignore')

def objective(trial, data, target, rounds):
    x_train, x_val, y_train, y_val = train_test_split(data, target,
                                                      random_state=42,
                                                      test_size=0.15,
                                                      stratify=target.values)

    params = {
        'objective': 'binary:logistic',
        'tree_method': 'gpu_hist',
        'lambda': 1,
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.8),
        'colsample_bytree': trial.suggest_categorical('colsample_bytree', [0.5, 0.6, 0.7, 0.8, 0.9, 1]),
        'subsample': trial.suggest_categorical('subsample', [0.6, 0.7, 0.8, 0.9, 1]),
        'max_depth': trial.suggest_categorical('max_depth', [2, 3, 4, 6, 8, 10, 12, 15, 17]),
        'random_state': trial.suggest_categorical('random_state', [42]),
        'min_child_weight': trial.suggest_categorical('min_child_weight', [1]),
        'eval_metric': 'auc',

    }
    dtrain = xgb.DMatrix(x_train, label=y_train)
    dval = xgb.DMatrix(x_val, label=y_val)
    watchlist = [(dtrain, 'train'), (dval, 'val')]
    bst = xgb.train(params=params, dtrain=dtrain,
                    num_boost_round=rounds,
                    evals=watchlist,
                    early_stopping_rounds=50,
                    verbose_eval=False
                    )
    preds_probs = bst.predict(dval)
    preds = [1 if x > 0.5 else 0 for x in preds_probs]
    roc_auc = roc_auc_score(y_val, preds)
    return roc_auc




if __name__ == '__main__':
    FILE = 'data/features_final.csv'
    NROUNDS = 1000
    NTRIALS = 500

    df = pd.read_csv(FILE)
    x = df.iloc[:, :-2]
    y = df.iloc[:, -2]
    normalizer = Normalizer()
    x = normalizer.fit_transform(x)
    x = pd.DataFrame(x, columns=df.columns[:-2])

    func = partial(objective, data=x, target=y, rounds=NROUNDS)
    study = optuna.create_study(direction='maximize')
    study.optimize(func, n_trials=NTRIALS)
    best_params = study.best_trial.params
    print(f'{best_params = }')

    x_train, x_val, y_train, y_val = train_test_split(x, y,
                                                      random_state=42,
                                                      test_size=0.15,
                                                      stratify=y.values
                                                      )
    dtrain = xgb.DMatrix(x_train, label=y_train)
    dval = xgb.DMatrix(x_val, label=y_val)
    watchlist = [(dtrain, 'train'), (dval, 'val')]
    bst = xgb.train(best_params, dtrain,
                    evals=[(dval, "validation")],
                    num_boost_round=1
                    )
    # preds_probs = bst.predict(dval)
    # preds = [1 if x > 0.5 else 0 for x in preds_probs]
    # roc_auc = roc_auc_score(y_val, preds)
    # print(f'{roc_auc = }')
    bst.save_model('models/best_model.json')

