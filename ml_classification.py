import numpy as np
import pandas as pd
import os.path as osp
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import recall_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from itertools import combinations
from data_handling import NESDA_FILE_MISSING, NESDA_FILE_MISSING_DTYPE, get_data
from scipy.special import binom
from time import time
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    from sklearn.externals.joblib import Parallel, delayed


USE_SUMMARY = False
N_JOBS = 15
SAVE_FOLDER = 'rfc'


def build_model(categorical_vars, ordinal_vars, interval_vars, random_state):

    imputer = ColumnTransformer(transformers=[
        ('categorical_impute', SimpleImputer(strategy='most_frequent'), categorical_vars),
        ('ordinal_impute', SimpleImputer(strategy='median'), ordinal_vars),
        ('interval_impute', SimpleImputer(strategy='mean'), interval_vars)
    ])

    clf = RandomForestClassifier(n_estimators=500, class_weight='balanced_subsample', max_features='sqrt',
                                 random_state=random_state)

    return clf, imputer


def one_cv_run(estimator, imputer, X, y, train, test, i_cv):
    y_train, y_test = y[train], y[test]
    X_train = imputer.fit_transform(X.loc[train], y[train])
    X_test = imputer.transform(X.loc[test])
    estimator.fit(X_train, y_train)
    y_pred = estimator.predict(X_test)
    y_score = estimator.predict_proba(X_test)[:, 1]
    scores = {'acc': balanced_accuracy_score(y_true=y_test, y_pred=y_pred),
              'sens': recall_score(y_true=y_test, y_pred=y_pred, pos_label=y_test.max()),
              'spec': recall_score(y_true=y_test, y_pred=y_pred, pos_label=y_test.min()),
              'auc': roc_auc_score(y_true=y_test, y_score=y_score)}
    y_predicted = pd.DataFrame(data=np.column_stack((test, y_test, y_pred, y_score, [i_cv] * y_test.size)),
                               columns=['test_index', 'y_true', 'y_pred', 'y_score', 'cv_fold'])
    feature_importances = estimator.feature_importances_
    feature_names = X.columns.values.astype(str)
    features = pd.DataFrame(data=np.column_stack((feature_names, feature_importances)), columns=['var_name',
                                                                                                 'var_importance'])
    return scores, y_predicted, features


def run():
    domains = ['IA', 'IIA', 'IIIA', 'IVA', 'VA']
    combination_length = [1, 2, 3, 4, 5]

    for i_comb, comb_len in enumerate(combination_length):
        for i_dom, dom in enumerate(combinations(domains, comb_len)):
            dom = list(dom)
            save_pattern = osp.join(SAVE_FOLDER, '_'.join(dom) + '_{}')
            print('Max domains: {} {}/{}; Domain combination: {}'.format(comb_len, i_dom + 1,
                                                                         int(binom(len(domains), comb_len)), dom))
            df, df_dtype, y = get_data(modality_name=dom, load_df=NESDA_FILE_MISSING,
                                       load_df_dtypes=NESDA_FILE_MISSING_DTYPE, summary_scores=USE_SUMMARY)
            print(y.value_counts())
            print('Shape Data: {}'.format(df.shape))
            # one-hot-encoder categorical variables
            cat_vars = df_dtype.variable_name[(df_dtype.data_type == 'Nominal')]
            ord_vars = df_dtype.variable_name[(df_dtype.data_type == 'Ordinal')]
            invl_vars = df_dtype.variable_name[(df_dtype.data_type == 'Scale')]

            if cat_vars.size > 0:
                df_enc = pd.get_dummies(df, prefix=cat_vars, columns=cat_vars, dummy_na=False)
                for col in cat_vars:
                    df_enc.loc[df[col].isnull(), df_enc.columns.str.startswith(col)] = np.nan
                cat_vars = np.concatenate([df_enc.columns[df_enc.columns.str.startswith(cat)] for cat in cat_vars])
                print('Shape after one-hot-encoding: {}'.format(df_enc.shape))
            else:
                df_enc = df
                print('No categorical variables for {}'.format(dom))

            seed = int(time())
            random_state = np.random.RandomState(seed=seed)

            clf, imputer = build_model(cat_vars, ord_vars, invl_vars, random_state)
            cv = StratifiedShuffleSplit(n_splits=100, test_size=0.2, random_state=random_state)

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=DeprecationWarning)
                res = Parallel(n_jobs=N_JOBS, verbose=1)(delayed(one_cv_run)(clone(clf), clone(imputer), df_enc, y,
                                                                             train, test, i_cv + 1)
                                                         for i_cv, (train, test) in enumerate(cv.split(df, y)))
            df_scores = pd.DataFrame(columns=['acc', 'sens', 'spec', 'auc'])
            df_features = pd.DataFrame(data=df_enc.columns.values.astype(str), columns=['var_name'])

            for i_cv in range(cv.get_n_splits(df, y)):
                scores_cv, predictions, features = res[i_cv]
                df_scores = df_scores.append(scores_cv, ignore_index=True)
                if i_cv == 0:
                    df_predictions = predictions
                else:
                    df_predictions = df_predictions.append(predictions, ignore_index=True)
                assert np.all(df_features.var_name == features.var_name), 'Mismatch in features'
                df_features['cv_{}'.format(i_cv + 1)] = features.var_importance.values

            df_scores.to_csv(save_pattern.format('performance.csv'), index=False)
            df_features.to_csv(save_pattern.format('var_importance.csv'), index=False)
            df_predictions.to_csv(save_pattern.format('predictions.csv'), index=False)
            np.savez(save_pattern.format('seed.npz'), np.array([seed]))
            print(df_scores.mean())


if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=DeprecationWarning)
        run()
