import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle


df_cell_train = pickle.load( open( "./train.p", "rb" ) )
df_cell_test = pickle.load( open( "./test.p", "rb" ) )


place_counts = df_cell_train.place_id.value_counts()
mask = (place_counts[df_cell_train.place_id.values] >= 8).values
df_cell_train = df_cell_train.loc[mask]
row_ids = df_cell_test.index
df_cell_train.loc[:,'x'] *= 500.0
df_cell_train.loc[:,'y'] *= 1000.0
df_cell_test.loc[:,'x'] *= 500.0
df_cell_test.loc[:,'y'] *= 1000.0

le = LabelEncoder()
X = df_cell_train.drop(['place_id'], axis=1).values
y = le.fit_transform(df_cell_train.place_id.values)

def calculate_distance(distances):
    return distances ** -2

clf_xgb = xgb.XGBClassifier()



def test_model(clf):
    clf.fit(X_train, y_train)
    y_pred = clf.predict_proba(X_valid)
    act_labels = le.inverse_transform(y_valid)
    pred_labels = le.inverse_transform(np.argsort(y_pred, axis=1)[:,::-1][:,:3])
    score = mapk(act_labels, pred_labels, 3)
    return score


def apk(actual, predicted, k=10):
    """
    Computes the average precision at k.
    This function computes the average prescision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)



    return score / min(len(actual), k), num_hits

def mapk(actual, predicted, k=10):
    """a
    Computes the mean average precision at k.
    This function computes the mean average prescision at k between two lists
    of lists of items.
    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The mean average precision at k over the input lists
    """
    return np.mean([apk([a],p,k) for a,p in zip(actual, predicted)])



from hyperopt import hp
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials


X_train ,X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.25, random_state=42)


act_labels = le.inverse_transform(y_valid)


clf_knn = KNeighborsClassifier(n_neighbors=36, weights=calculate_distance,
                               metric='manhattan')


scores_knn = test_model(clf_knn)

print "The socore for knn is  is {}".format(scores_knn)


clf_xgb = xgb.XGBClassifier()

scores_xgb = test_model(clf_xgb)

print "The socore for knn is  is {}".format(scores_xgb)

params = {'n_estimators': 100, 'subsample': 0.950, 'learning_rate': 0.15, 'colsample_bytree': 0.7, 'min_child_weight': 6.0}


clf_new = xgb.XGBClassifier(**params)

scores_new = test_model(clf_new)

print "Now the score is {}".format(scores_new)



print "Now start parameters tuning"

def score(params):
    print "Training with params : "
    print params
    num_round = int(params['n_estimators'])
    del params['n_estimators']

    # watchlist = [(dvalid, 'eval'), (dtrain, 'train')]
    clf = xgb.XGBClassifier(**params)

    score = test_model(clf)
    print "\tScore {0}\n\n".format(score)
    return {'loss': score, 'status': STATUS_OK}


def optimize(trials):
    space = {
             'n_estimators' : hp.quniform('n_estimators', 100, 1000, 100),
             'learning_rate' : hp.quniform('learning_rate', 0.05, 0.5, 0.05),
             #'max_depth' : hp.quniform('max_depth', 1, 13, 2),
             'min_child_weight' : hp.quniform('min_child_weight', 1, 6, 1),
             'subsample' : hp.quniform('subsample', 0.5, 1, 0.05),
             'colsample_bytree' : hp.quniform('colsample_bytree', 0.5, 1, 0.05),


             }

    best = fmin(score, space, algo=tpe.suggest, trials=trials, max_evals=250)

    print best


#Trials object where the history of search will be stored
trials = Trials()

optimize(trials)
