import os
import os.path as osp
import warnings
from itertools import combinations
from time import time

import numpy as np
import pandas as pd
from category_encoders import TargetEncoder
from imblearn.ensemble import BalancedRandomForestClassifier
from joblib import Parallel, delayed
from scipy.special import binom
from sklearn.base import clone
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import recall_score, balanced_accuracy_score, roc_auc_score, precision_score
from sklearn.model_selection import RepeatedStratifiedKFold

from data_handling import NESDA_FILE_MISSING, NESDA_FILE_MISSING_DTYPE, NESDA_FILE_MISSING_SUMMARY, \
    NESDA_FILE_MISSING_SUMMARY_DTYPE, NESDA_FILE_LABELS, get_data

USE_SUMMARY = False
summary_prefix = '_summary' if USE_SUMMARY else ''
N_JOBS = 1
N_JOBS_RF = 1
N_PERM = 1
SAVE_FOLDER = 'rfc{}'.format(summary_prefix)
DOMAINS = ('IA', 'IIA', 'IIIA', 'IVA', 'VA')


def get_classifier(n_subj, random_state, n_jobs_rf=1, multiclass=False):
    if multiclass:
        # multiplication with 0.9 required to make the subject number agree with training set AND because one of the
        # classes has only very few subject such that we can't reasonably sample more than 100 subjects
        subsample_size = round(n_subj * 0.9 * 0.5 / 4)
        estimator = BalancedRandomForestClassifier(n_estimators=1000, class_weight='balanced', oob_score=False,
                                                   sampling_strategy={0: subsample_size, 1: subsample_size,
                                                                      2: subsample_size, 3: subsample_size},
                                                   n_jobs=n_jobs_rf, random_state=random_state, bootstrap=False,
                                                   replacement=False)
    else:
        subsample_size = round(n_subj * 0.632 / 2)
        estimator = BalancedRandomForestClassifier(n_estimators=1000, class_weight='balanced', oob_score=False,
                                                   sampling_strategy={0: subsample_size, 1: subsample_size},
                                                   n_jobs=n_jobs_rf, random_state=random_state, bootstrap=False,
                                                   replacement=False)
    return estimator


def score(y_true, y_pred, y_score, multiclass=False):
    if multiclass:
        y_onh = (y_true[:, np.newaxis] == np.unique(y_true)[np.newaxis, :]).astype(np.int)
        labels_order = ['acc_comorbid', 'acc_anxiety', 'acc_other', 'acc_nothing']
        acc_dict = {}
        acc_all = 0
        for i in range(len(labels_order) - 1, -1, -1):
            id_class = y_true == i
            acc_dict[labels_order[i]] = np.mean(y_true[id_class] == y_pred[id_class])
            acc_all += acc_dict[labels_order[i]] / len(labels_order)

        acc_dict['acc_overall'] = acc_all

        auc_micro = roc_auc_score(y_onh, y_score, average='micro')
        auc_macro = roc_auc_score(y_onh, y_score, average='macro')
        auc_weighted = roc_auc_score(y_onh, y_score, average='weighted')
        auc_samples = roc_auc_score(y_onh, y_score, average='samples')

        scores = {'auc_one_vs_rest': auc_weighted,
                  'auc_micro': auc_micro,
                  'auc_macro': auc_macro,
                  'auc_weighted': auc_weighted,
                  'auc_samples': auc_samples}
        scores = dict(**acc_dict, **scores)
    else:
        scores = {'acc': balanced_accuracy_score(y_true=y_true, y_pred=y_pred),
                  'sens': recall_score(y_true=y_true, y_pred=y_pred, pos_label=y_true.max()),
                  'spec': recall_score(y_true=y_true, y_pred=y_pred, pos_label=y_true.min()),
                  'AUC': roc_auc_score(y_true=y_true, y_score=y_score),
                  'PPV': precision_score(y_true=y_true, y_pred=y_pred, pos_label=y_true.max()),
                  'NPV': precision_score(y_true=y_true, y_pred=y_pred, pos_label=y_true.min())}
    return scores


def run_ml(save_folder=SAVE_FOLDER, domains=DOMAINS, n_jobs=N_JOBS, use_summary=USE_SUMMARY,
           type_of_analysis='standard', combination_length=(1, 5), target_thresh=0.6, n_perm=N_PERM,
           target_metric='AUC', seed=None, n_jobs_rf=N_JOBS_RF, cat_encoding=None):

    if not osp.exists(save_folder):
        os.makedirs(save_folder)

    if seed is None:
        seed = int(time())

    # because of all the extra analysis we will always get the more descriptive 'pureanxiety' column as well
    target_col = ['persistance_anxiety', 'pureanxiety']
    print(type_of_analysis)

    for i_comb, comb_len in enumerate(combination_length):
        for i_dom, dom in enumerate(combinations(domains, comb_len)):
            dom = list(dom)
            random_state = np.random.RandomState(seed=seed)
            save_pattern = osp.join(save_folder, '_'.join(dom) + '_{}')
            print('Max domains: {} {}/{}; Domain combination: {}'.format(comb_len, i_dom + 1,
                                                                         int(binom(len(domains), comb_len)), dom))
            df, df_dtype, y = get_data(modality_name=dom,
                                       load_df=NESDA_FILE_MISSING,
                                       load_df_dtypes=NESDA_FILE_MISSING_DTYPE,
                                       load_df_summary=NESDA_FILE_MISSING_SUMMARY,
                                       load_df_dtypes_summary=NESDA_FILE_MISSING_SUMMARY_DTYPE,
                                       load_df_labels=NESDA_FILE_LABELS,
                                       use_summary=use_summary,
                                       target_col=target_col)
            print('Shape Data: {}'.format(df.shape))
            cat_vars = df_dtype.variable_name[(df_dtype.data_type == 'Nominal')].values
            other_vars = df_dtype.variable_name[(df_dtype.data_type != 'Nominal')].values

            y, multiclass = create_labels(y, type_of_analysis)

            res = run_cross_validation(df_X=df, y=y, cat_vars=cat_vars, other_vars=other_vars, n_jobs=n_jobs,
                                       random_state=random_state, n_jobs_rf=n_jobs_rf, cat_encoding=cat_encoding,
                                       multiclass=multiclass)
            df_perf_train, df_perf_test, df_pred_test, df_feat_import = get_results(res)

            df_perf_test.to_csv(save_pattern.format('performance_test.csv'), index=False)
            df_perf_train.to_csv(save_pattern.format('performance_train.csv'), index=False)
            df_feat_import.to_csv(save_pattern.format('var_importance.csv'), index=False)
            df_pred_test.to_csv(save_pattern.format('predictions.csv'), index=False)

            print()
            print("Training-Set:")
            print(df_perf_train.mean())
            print()
            print('Test-Set:')
            print(df_perf_test.mean())
            print()

            mean_target_cv = df_perf_test[target_metric].mean()

            if (mean_target_cv >= target_thresh) and (n_perm > 1):
                print('{}: {} >= {}'.format(target_metric, mean_target_cv, target_thresh))
                print('Running Permutations... (n={})'.format(n_perm))
                print()

                feat_imp_columns = df_feat_import.columns[df_feat_import.columns.str.startswith('cv_')]
                var_names = df_feat_import.var_name.values
                df_feat_import_all_perm, df_perf_test_all_perm = run_permutations(df_X=df, y=y, cat_vars=cat_vars,
                                                                                  other_vars=other_vars,
                                                                                  perf_columns=df_perf_test.columns,
                                                                                  var_names=var_names,
                                                                                  feat_imp_columns=feat_imp_columns,
                                                                                  n_jobs=n_jobs, n_perm=n_perm,
                                                                                  random_state=random_state,
                                                                                  n_jobs_rf=n_jobs_rf,
                                                                                  multiclass=multiclass)

                df_perf_test_all_perm.to_csv(save_pattern.format('perf_permutations.csv'), index=False)
                df_feat_import_all_perm.to_csv(save_pattern.format('var_imprt_permutations.csv'), index=False)

    np.savez(osp.join(save_folder, 'seed.npz'), np.array([seed]))


def create_labels(y, type_of_analysis):
    multiclass = False
    if type_of_analysis == 'any_disorder':
        id_other = (y.pureanxiety == 'comorbid->pure_other') | \
                   (y.pureanxiety == 'pure_anxiety->pure_other')
        y.persistance_anxiety.loc[id_other] = 1
        y['clf_labels'] = y.persistance_anxiety.values
    elif type_of_analysis == 'multiclass':
        clf_labels = np.empty(y.shape[0])
        class_comorbid = (y.pureanxiety == 'comorbid->comorbid') | (y.pureanxiety == 'pure_anxiety->comorbid')
        class_anxiety = ((y.pureanxiety == 'comorbid->pure_anxiety') |
                         (y.pureanxiety == 'pure_anxiety->pure_anxiety'))
        class_other = (y.pureanxiety == 'comorbid->pure_other') | (y.pureanxiety == 'pure_anxiety->pure_other')
        class_nothing = (y.pureanxiety == 'comorbid->nothing') | (y.pureanxiety == 'pure_anxiety->nothing')
        clf_labels[class_comorbid] = 3
        clf_labels[class_anxiety] = 2
        clf_labels[class_other] = 1
        clf_labels[class_nothing] = 0
        y['clf_labels'] = clf_labels
        multiclass = True
    elif type_of_analysis == 'any_anxiety':
        y['clf_labels'] = y.persistance_anxiety.values
    else:
        raise NotImplementedError('{} not recognized'.format(type_of_analysis))

    print(y.clf_labels.value_counts())
    print(y.groupby('clf_labels').pureanxiety.value_counts())

    y = y.clf_labels.values.squeeze()
    return y, multiclass


def run_permutations(df_X, y, cat_vars, other_vars, perf_columns, var_names, feat_imp_columns, n_jobs, n_perm,
                     random_state, n_jobs_rf, multiclass):
    y_perm = y.copy()
    time_diff = 0
    df_perf_test_all_perm = pd.DataFrame(columns=perf_columns)
    df_feat_import_all_perm = pd.DataFrame(data=var_names, columns=['var_name'])
    t1_total = time()
    for i_perm in range(n_perm):
        t1 = time()
        print('Perm. Iteration: {}/{}; time: {:0.2f}m'.format(i_perm + 1, n_perm, time_diff), end='\r')
        np.random.shuffle(y_perm)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UndefinedMetricWarning)
            res_perm = run_cross_validation(df_X=df_X, y=y_perm, cat_vars=cat_vars, other_vars=other_vars,
                                            n_jobs=n_jobs, random_state=random_state, verbose=0, n_jobs_rf=n_jobs_rf,
                                            multiclass=multiclass)
        _, df_perf_test_perm, _, df_feat_import_perm = get_results(res_perm)

        df_perf_test_all_perm = df_perf_test_all_perm.append(df_perf_test_perm.mean(), ignore_index=True)
        df_feat_import_all_perm['perm_{}'.format(i_perm + 1)] = \
            df_feat_import_perm[feat_imp_columns].astype(np.float).mean(axis='columns').values

        t2 = time()
        time_diff = (t2 - t1) / 60.
    t2_total = time()
    print()
    print('Total time: {:0.2f}h'.format((t2_total - t1_total) / 3600.))
    return df_feat_import_all_perm, df_perf_test_all_perm


def get_results(cv_res):
    scores_test_names = cv_res[0][0].keys()
    scores_train_names = cv_res[0][-1].keys()
    pred_test_names = cv_res[0][1].columns
    col_encoded = cv_res[0][-2]
    df_perf_test = pd.DataFrame(columns=scores_test_names)
    df_perf_train = pd.DataFrame(columns=scores_train_names)
    df_feat_import = pd.DataFrame(data=col_encoded, columns=['var_name'])
    df_pred_test = pd.DataFrame(columns=pred_test_names)

    for i_cv in range(len(cv_res)):
        # the 2nd to last argument are the (potentially) encoded variable names but we don't need them here
        scores_test, df_pred_cv, feature_importances, _, scores_train = cv_res[i_cv]

        df_perf_test = df_perf_test.append(scores_test, ignore_index=True)
        df_perf_train = df_perf_train.append(scores_train, ignore_index=True)
        df_pred_test = df_pred_test.append(df_pred_cv, ignore_index=True)

        assert np.all(df_feat_import.var_name == feature_importances.var_name), 'Mismatch in features'
        df_feat_import['cv_{}'.format(i_cv + 1)] = feature_importances.var_importance.values
    return df_perf_train, df_perf_test, df_pred_test, df_feat_import


def run_cross_validation(df_X, y, cat_vars, other_vars, n_jobs, random_state, n_jobs_rf=1, verbose=1,
                         cat_encoding=None, multiclass=False):
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=random_state)
    estimator = get_classifier(n_subj=df_X.shape[0], random_state=random_state, n_jobs_rf=n_jobs_rf,
                               multiclass=multiclass)
    res = Parallel(n_jobs=n_jobs,
                   verbose=verbose,
                   max_nbytes='50M')(delayed(one_cv_run_ref)(clone(estimator), df_X, y, train, test, i_cv + 1, cat_vars,
                                                             other_vars, cat_encoding, multiclass)
                                     for i_cv, (train, test) in enumerate(cv.split(df_X, y)))
    return res


def categorical_encoding(df_X, y, cat_vars, id_train, method=None):
    if method is None:
        return df_X.values, df_X.columns

    target_enc = TargetEncoder(cols=cat_vars, drop_invariant=False, return_df=True, impute_missing=False,
                               handle_unknown='error')
    target_enc.fit(df_X.iloc[id_train], pd.Series(y).iloc[id_train])
    df_X = target_enc.transform(df_X)

    return df_X.values, df_X.columns


def one_cv_run_ref(estimator, X, y, id_train, id_test, i_cv, cat_vars, other_vars, cat_encoding=None, multiclass=False):
    y_train, y_test = y[id_train], y[id_test]

    X_cat = X.loc[:, cat_vars]
    # replace categorical missing values with mode of training set
    X.loc[:, cat_vars] = X_cat.fillna(X_cat.iloc[id_train].mode().iloc[0])
    # replace other missing values with median of training set
    X_other = X.loc[:, other_vars]
    X.loc[:, other_vars] = X_other.fillna(X_other.iloc[id_train].median())

    X, col_encoded = categorical_encoding(X, y, cat_vars, id_train, method=cat_encoding)
    X_train, X_test = X[id_train], X[id_test]

    # estimator = get_classifier(n_subj=X_train.shape[0], n_features=X_train.shape[1], random_state=random_state,
    #                            n_jobs_rf=n_jobs_rf)
    estimator.fit(X_train, y_train)

    score_label = ['y_score']
    id_score = 1

    if multiclass:
        score_label = ['y_score_0', 'y_score_1', 'y_score_2', 'y_score_3']
        id_score = [0, 1, 2, 3]

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=UndefinedMetricWarning)
        y_pred = estimator.predict(X_test)
        y_score = estimator.predict_proba(X_test)[:, id_score]
        scores_test = score(y_true=y_test, y_pred=y_pred, y_score=y_score, multiclass=multiclass)

        y_score_train = estimator.predict_proba(X_train)[:, id_score]
        y_pred_train = estimator.predict(X_train)
        scores_train = score(y_true=y_train, y_pred=y_pred_train, y_score=y_score_train, multiclass=multiclass)

    df_pred = pd.DataFrame(data=np.column_stack((id_test, y_test, y_pred, y_score, [i_cv] * y_test.size)),
                           columns=['test_index', 'y_true', 'y_pred'] + score_label + ['cv_fold'])

    feature_importances = estimator.feature_importances_
    feature_names = col_encoded.values.astype(str)
    df_features = pd.DataFrame(data=np.column_stack((feature_names, feature_importances)), columns=['var_name',
                                                                                                    'var_importance'])
    return scores_test, df_pred, df_features, feature_names, scores_train


if __name__ == '__main__':
    run_ml(save_folder=SAVE_FOLDER, domains=DOMAINS, n_jobs=N_JOBS, n_jobs_rf=N_JOBS_RF, use_summary=USE_SUMMARY,
           n_perm=N_PERM)
