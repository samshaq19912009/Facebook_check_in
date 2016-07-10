# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

#COMPETITION NAME (the 'Competition'): Facebook V: Predicting Check Ins
#COMPETITION SPONSOR: Facebook
#COMPETITION WEBSITE: https://www.kaggle.com/c/facebook-v-predicting-check-ins

'''Partially based on several scripts:
https://www.kaggle.com/c/facebook-v-predicting-check-ins/scripts
'''

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
import datetime
import time



print('	Reading train.csv')
df_train = pd.read_csv('../input/train.csv',
                        usecols=['row_id','x','y','time','place_id','accuracy'],
                        index_col = 0)
print('	Reading test.csv')
df_test = pd.read_csv('../input/test.csv',
                        usecols=['row_id','x','y','time','accuracy'],
                        index_col = 0)

#print('Feature Augmentation')
minute = df_train.time%60
df_train['hour'] = df_train['time']//60
df_train.drop(['time'], axis=1, inplace=True)
df_train['weekday'] = df_train['hour']//24
df_train['month'] = df_train['weekday']//30
df_train['year'] = (df_train['weekday']//365+1)*10.0
df_train['hour'] = ((df_train['hour']%24+1)+minute/60.0)*4.0
df_train['weekday'] = (df_train['weekday']%7+1)*3.0
df_train['month'] = (df_train['month']%12+1)*2.0
df_train['accuracy'] = np.log10(df_train['accuracy'])*10.0

minute = df_test['time']%60
df_test['hour'] = df_test['time']//60
df_test.drop(['time'], axis=1, inplace=True)
df_test['weekday'] = df_test['hour']//24
df_test['month'] = df_test['weekday']//30
df_test['year'] = (df_test['weekday']//365+1)*10.0
df_test['hour'] = ((df_test['hour']%24+1)+minute/60.0)*4.0
df_test['weekday'] = (df_test['weekday']%7+1)*3.0
df_test['month'] = (df_test['month']%12+1)*2.0
df_test['accuracy'] = np.log10(df_test['accuracy'])*10.0

# add data for periodic time that hit the boundary
add_data = df_train[df_train.hour<6]
add_data.hour = add_data.hour+96
df_train = df_train.append(add_data)

add_data = df_train[df_train.hour>98]
add_data.hour = add_data.hour-96
df_train = df_train.append(add_data)


print('Generating wrong models. They are just useful to get this job :) ... done')
pd.options.mode.chained_assignment = None

def calculate_distance(distances):
    return distances ** -2

def process_one_cell(df_cell_train, df_cell_test):


    #Working on df_train
    place_counts = df_cell_train.place_id.value_counts()
    mask = (place_counts[df_cell_train.place_id.values] >= 8).values
    df_cell_train = df_cell_train.loc[mask]

    #Working on df_test
    row_ids = df_cell_test.index

    #Feature engineering on x and y
    df_cell_train.loc[:,'x'] *= 500.0
    df_cell_train.loc[:,'y'] *= 1000.0
    df_cell_test.loc[:,'x'] *= 500.0
    df_cell_test.loc[:,'y'] *= 1000.0

    #print "Preparing data"
    le = LabelEncoder()
    y = le.fit_transform(df_cell_train.place_id.values)
    X = df_cell_train.drop(['place_id'], axis=1).values
    X_test = df_cell_test.values

    #Applying the classifier
    clf_knn = KNeighborsClassifier(n_neighbors=36, weights=calculate_distance,
                               metric='manhattan')
    clf_knn.fit(X, y)
    y_pred_knn = clf_knn.predict_proba(X_test)

    params = {'n_estimators': 100, 'subsample': 0.95, 'learning_rate': 0.15, 'colsample_bytree': 0.7, 'min_child_weight': 6.0}

    clf_xgb = xgb.XGBClassifier(**params)
    clf_xgb.fit(X, y)
    y_pred_xgb = clf_xgb.predict_proba(X_test)

    paras_rf = {'min_samples_split': 7.0, 'max_depth': 12.0, 'random_state':1234}


    clf_rf = RandomForestClassifier(**paras_rf)

    clf_rf.fit(X, y)
    y_pred_rf = clf_rf.predict_proba(X_test)

    #6 4 1
    #3 1 0
    ytotal = 1 * y_pred_xgb + 0.2 * y_pred_knn + 0.07 * y_pred_rf

    #ytotal = y_pred_knn

    pred_labels = le.inverse_transform(np.argsort(ytotal, axis=1)[:,::-1][:,:5])


    return pred_labels, row_ids

def process_grid(df_train, df_test):
    """
    Iterates over all grid cells, aggregates the results
    """
    size = 10.0
    x_step = 0.5
    y_step = 0.25

    x_border_augment = 0.025
    y_border_augment = 0.0125

    preds = np.zeros((df_test.shape[0], 5), dtype=int)

    for i in range((int)(size/x_step)):

        print "process {} x_grid ".format(i)
        start_time_row = time.time()

        x_min = x_step * i
        x_max = x_step * (i+1)
        x_min = round(x_min, 4)
        x_max = round(x_max, 4)
        if x_max == size:
            x_max = x_max + 0.001

        df_col_train = df_train[(df_train['x'] >= x_min-x_border_augment) & (df_train['x'] < x_max+x_border_augment)]
        df_col_test = df_test[(df_test['x'] >= x_min) & (df_test['x'] < x_max)]

        for j in range((int)(size/y_step)):
            y_min = y_step * j
            y_max = y_step * (j+1)
            y_min = round(y_min, 4)
            y_max = round(y_max, 4)
            if y_max == size:
                y_max = y_max + 0.001

            df_cell_train = df_col_train[(df_col_train['y'] >= y_min-y_border_augment) & (df_col_train['y'] < y_max+y_border_augment)]
            df_cell_test = df_col_test[(df_col_test['y'] >= y_min) & (df_col_test['y'] < y_max)]

            #Applying classifier to one grid cell
            pred_labels, row_ids = process_one_cell(df_cell_train, df_cell_test)

            #Updating predictions
            preds[row_ids] = pred_labels
        print("Elapsed time for this grid: %s seconds" % (time.time() - start_time_row))

    #Auxiliary dataframe with the 3 best predictions for each sample
    df_aux = pd.DataFrame(preds, dtype=str, columns=['l1', 'l2', 'l3','l4','l5'])

    return df_aux

def generate_sub(df_aux):
    #print('Writing submission file')
    #Concatenating the 3 predictions for each sample
    ds_sub = df_aux.l1.str.cat([df_aux.l2, df_aux.l3,df_aux.l4, df_aux.l5], sep=' ')

    #Writting to csv
    ds_sub.name = 'place_id'
    now = datetime.datetime.now()
    sub = 'submission_mixed' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'
    ds_sub.to_csv(sub, index=True, header=True, index_label='row_id')

df_aux=process_grid(df_train, df_test)

generate_sub(df_aux)
