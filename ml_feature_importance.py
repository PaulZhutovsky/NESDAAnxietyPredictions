
from time import time
import os.path as osp
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.externals.joblib import Parallel, delayed
from ml_classification import get_classifier, categorical_encoding, create_labels
from data_handling import NESDA_FILE_MISSING, NESDA_FILE_MISSING_DTYPE, NESDA_FILE_MISSING_SUMMARY, \
    NESDA_FILE_MISSING_SUMMARY_DTYPE, NESDA_FILE_LABELS, get_data


def run_perm_analysis(save_folder, domains='all', n_jobs=10, use_summary=False, type_of_analysis='any_anxiety',
                      n_perm=1000, seed=None, n_jobs_rf=2, cat_encoding=None):

    if seed is None:
        seed = int(time())

    target_col = ['persistance_anxiety', 'pureanxiety']
    df, df_dtype, y = get_data(modality_name=domains,
                               load_df=NESDA_FILE_MISSING,
                               load_df_dtypes=NESDA_FILE_MISSING_DTYPE,
                               load_df_summary=NESDA_FILE_MISSING_SUMMARY,
                               load_df_dtypes_summary=NESDA_FILE_MISSING_SUMMARY_DTYPE,
                               load_df_labels=NESDA_FILE_LABELS,
                               use_summary=use_summary,
                               target_col=target_col)

    y, multiclass = create_labels(y, type_of_analysis)

    df, cat_vars = impute_data(df, df_dtype)
    X, var_names = categorical_encoding(df, y, cat_vars, np.arange(df.shape[0]), [], method=cat_encoding)
    n_subj, n_features = X.shape
    estimator = get_classifier(n_subj, n_features, random_state=seed, n_jobs_rf=n_jobs_rf, multiclass=multiclass)

    estimator.fit(X, y)
    feat_imp_true = estimator.feature_importances_
    perm_col = ['perm_{}'.format(i + 1) for i in range(n_perm)]

    df_feat_imp = pd.DataFrame(index=var_names, columns=['true_feature_importances'] + perm_col)
    df_feat_imp['true_feature_importances'] = feat_imp_true

    for i_feature in range(X.shape[1]):
        print('{}/{}; Feature: {}'.format(i_feature + 1, X.shape[1], var_names[i_feature]))
        X_perm = X.copy()
        res = Parallel(n_jobs=n_jobs, verbose=1, pre_dispatch='2*n_jobs', max_nbytes='50M')(delayed(permute_feature)(
            clone(estimator), X_perm, y, i_feature) for _ in range(n_perm))
        df_feat_imp.loc[var_names[i_feature], perm_col] = res

    df_feat_imp.to_csv(osp.join(save_folder, 'permuted_variable_importances_domains_{}.csv'.format(domains)))
    np.save(osp.join(save_folder, 'permuted_variable_importances_domains_{}_seed.npy'.format(domains)),
            np.array([seed]))


def permute_feature(estimator, X, y, feature_id=0):
    X[:, feature_id] = np.random.permutation(X[:, feature_id])
    estimator.fit(X, y)
    feat_imp_perm = estimator.feature_importances_
    return feat_imp_perm[feature_id]


def impute_data(df, df_dtype):
    cat_vars = df_dtype.variable_name[(df_dtype.data_type == 'Nominal')].values
    other_vars = df_dtype.variable_name[(df_dtype.data_type != 'Nominal')].values

    df[cat_vars] = df[cat_vars].fillna(df[cat_vars].mode().iloc[0])
    df[other_vars] = df[other_vars].fillna(df[other_vars].median())
    return df, cat_vars


if __name__ == '__main__':
    # SAVE_FOLDER = ['/data/pzhutovsky/NESDA_anxiety/rfc_indv_scores_1000trees_balanced_any_anxiety',
    #                '/data/pzhutovsky/NESDA_anxiety/rfc_indv_scores_1000trees_balanced_any_disorder']
    # SAVE_FOLDER = ['/data/pzhutovsky/NESDA_anxiety/rfc_indv_scores_1000trees_balanced_any_disorder']
    # SAVE_FOLDER *= 3
    SAVE_FOLDER = ['/data/pzhutovsky/NESDA_anxiety/rfc_indv_scores_1000trees_balanced_multiclass']
    # TYPE_OF_ANALYSIS = ['any_anxiety', 'any_disorder']
    TYPE_OF_ANALYSIS = ['multiclass']
    # TYPE_OF_ANALYSIS *= 3
    domains = ['all']
    for i in range(len(SAVE_FOLDER)):
        run_perm_analysis(SAVE_FOLDER[i], domains=domains[i], n_jobs=15, use_summary=False,
                          type_of_analysis=TYPE_OF_ANALYSIS[i], n_perm=1000, n_jobs_rf=3)
