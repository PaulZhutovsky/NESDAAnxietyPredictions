import numpy as np
import pandas as pd
import os
import os.path as osp
from sklearn.metrics import recall_score, balanced_accuracy_score, roc_auc_score, precision_score, brier_score_loss
from sklearn.model_selection import RepeatedStratifiedKFold
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
from random_forest_subsample import RandomForestSubsampleClassifier, RandomForestAdjustedClassifier
from itertools import combinations
from data_handling import NESDA_FILE_MISSING, NESDA_FILE_MISSING_DTYPE, NESDA_FILE_MISSING_SUMMARY, \
    NESDA_FILE_MISSING_SUMMARY_DTYPE, NESDA_FILE_LABELS, get_data
from scipy.special import binom
from time import time
from joblib import Parallel, delayed
from random_search_oob import RandomSearchOOB, RFCSearchOOB
from category_encoders import TargetEncoder
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.base import clone
import warnings


USE_SUMMARY = False
summary_prefix = '_summary' if USE_SUMMARY else ''
N_JOBS = 15
SAVE_FOLDER = 'rfc4{}'.format(summary_prefix)
DOMAINS = ['IA', 'IIA', 'IIIA', 'IVA', 'VA']


# definition follows from: https://rdrr.io/cran/measures/man/multiclass.Brier.html
def brier_score_multiclass(y_true, y_prob):
    """

    :param y_true: has to be one-hot-encoded!
    :param y_prob:
    :return:
    """
    sqr_error = (y_true - y_prob) ** 2
    return sqr_error.sum(axis=1).mean()


def get_classifier(n_subj, n_features, random_state, n_iter=30, n_jobs_rf=1, multiclass=False):
    # rfc = RandomForestClassifier(n_estimators=301, oob_score=True, class_weight='balanced_subsample',
    #                              random_state=random_state, n_jobs=n_jobs_rf)
    #
    # rfc = RandomForestSubsampleClassifier(n_estimators=201, class_weight='balanced_subsample',
    #                                       random_state=random_state, n_jobs=n_jobs_rf)
    # scorer = brier_score_loss
    # use_min = True
    # if multiclass:
    #     scorer = brier_score_multiclass
    # # min_samples_leaf is adjusted based on https://arxiv.org/pdf/1804.03515.pdf (3.5 the tuneRanger package)
    # x = np.random.rand(n_iter)
    # min_samples_leaf = ((n_subj * 0.2) ** x).astype(np.int)
    # max_features = np.random.randint(1, int(0.75 * n_features) + 1, size=n_iter)
    # subsample_size = np.arange(int(0.2 * n_subj), int(0.9 * n_subj) + 1)
    # subsample_size = subsample_size[np.random.randint(0, subsample_size.size, size=n_iter)]
    # estimator = RandomSearchOOB(random_forest=rfc,
    #                             max_features=max_features,
    #                             min_samples_leaf=min_samples_leaf,
    #                             subsample_size=subsample_size,
    #                             n_iter=n_iter,
    #                             scoring=scorer,
    #                             use_min=use_min,
    #                             multiclass=multiclass)
    # estimator = RandomForestAdjustedClassifier(n_estimators=501, class_weight='balanced_subsample',
    #                                            random_state=random_state,
    #                                            n_jobs=n_jobs_rf)

    # rfc = RandomForestAdjustedClassifier(n_estimators=500, class_weight='balanced_subsample', random_state=random_state,
    #                                      n_jobs=n_jobs_rf)
    # scorer = brier_score_loss
    # use_min = True
    # min_samples_leaf = np.arange(0, 45, 5)
    # min_samples_leaf[0] = 1
    # params = {'min_samples_leaf': min_samples_leaf}
    # estimator = RFCSearchOOB(random_forest=rfc, params=params, scoring=scorer, use_min=use_min)
    if multiclass:
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


def one_cv_run(X, y, train, test, i_cv, cat_vars, other_vars, random_state, n_jobs_rf=1, multiclass=False,
               ignore_in_train=None):
    train = np.setdiff1d(train, ignore_in_train)
    y_train, y_test = y[train], y[test]

    X_cat = X.loc[:, cat_vars]
    # replace categorical missing values with mode of training set
    X.loc[:, cat_vars] = X_cat.fillna(X_cat.iloc[train].mode().iloc[0])
    # replace other missing values with median of training set
    X_other = X.loc[:, other_vars]
    X.loc[:, other_vars] = X_other.fillna(X_other.iloc[train].median())
    col_used = X.columns
    X = X.values
    X_train, X_test = X[train], X[test]

    estimator = get_classifier(n_subj=X_train.shape[0], n_features=X_train.shape[1], random_state=random_state,
                               n_jobs_rf=n_jobs_rf, multiclass=multiclass)
    estimator.fit(X_train, y_train)
    id_prob = 1
    y_score_labels = ['y_score']
    if multiclass:
        id_prob = [0, 1, 2]
        y_score_labels = ['y_score_0', 'y_score_1', 'y_score_2']
    y_pred = estimator.predict(X_test)
    y_score = estimator.predict_proba(X_test)[:, id_prob]
    scores_test = score(y_true=y_test, y_pred=y_pred, y_score=y_score, multiclass=multiclass)
    y_score_train = estimator.oob_decision_function_[:, id_prob]
    if multiclass:
        y_pred_train = y_score_train.argmax(axis=1)
    else:
        y_pred_train = (y_score_train > 0.5).astype(np.int)
    scores_train = score(y_true=y_train, y_pred=y_pred_train, y_score=y_score_train, multiclass=multiclass)

    y_predicted = pd.DataFrame(data=np.column_stack((test, y_test, y_pred, y_score, [i_cv] * y_test.size)),
                               columns=['test_index', 'y_true', 'y_pred'] + y_score_labels + ['cv_fold'])
    feature_importances = estimator.feature_importances_
    feature_names = col_used.values.astype(str)
    features = pd.DataFrame(data=np.column_stack((feature_names, feature_importances)), columns=['var_name',
                                                                                                 'var_importance'])
    # picked_hyperparam = {'best_min_samples_leaf': estimator.best_min_samples_leaf,
    #                      'best_max_features': estimator.best_max_features,
    #                      'best_oob_score': estimator.best_oob_score,
    #                      'best_subsample_size': estimator.best_subsample_size}
    # train_set_evaluations = {**picked_hyperparam, **scores_train}
    train_set_evaluations = scores_train

    return scores_test, y_predicted, features, train_set_evaluations


def run_ml(save_folder=SAVE_FOLDER, domains=DOMAINS, n_jobs=N_JOBS, use_summary=USE_SUMMARY,
           type_of_analysis='standard', combination_length=(1, 2, 3, 4, 5), target_thresh=0.6, n_perm=1000,
           target_metric='AUC', seed=None, n_jobs_rf=1, cat_encoding=None):

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
                df_feat_import_all_perm, df_perf_test_all_perm = run_permutations(df_X=df, y=y, cat_vars=cat_vars,
                                                                                  other_vars=other_vars,
                                                                                  perf_columns=df_perf_test.columns,
                                                                                  var_names=df_feat_import.var_name.values,
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
    print()
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
    estimator = get_classifier(df_X.shape[0], df_X.shape[1], random_state, n_jobs_rf=n_jobs_rf, multiclass=multiclass)
    res = Parallel(n_jobs=n_jobs,
                   verbose=verbose,
                   max_nbytes='50M')(delayed(one_cv_run_ref)(clone(estimator), df_X, y, train, test, i_cv + 1, cat_vars,
                                                             other_vars, cat_encoding, multiclass)
                                     for i_cv, (train, test) in enumerate(cv.split(df_X, y)))
    return res


def categorical_encoding(df_X, y, cat_vars, id_train, id_test, method=None):
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

    X, col_encoded = categorical_encoding(X, y, cat_vars, id_train, id_test, method=cat_encoding)
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


def run(save_folder=SAVE_FOLDER, domains=DOMAINS, n_jobs=N_JOBS, use_summary=USE_SUMMARY, type_of_analysis='standard',
        combination_length=(1, 2, 3, 4, 5)):

    if not osp.exists(save_folder):
        os.makedirs(save_folder)

    seed = int(time())
    multiclass = True if type_of_analysis.startswith('multiclass') else False
    # because of all the extra analysis we will always get the more descriptive 'pureanxiety' column as well
    target_col = ['persistance_anxiety', 'pureanxiety']
    # for the case where the pure_other cases are ignored during training
    ignore_in_train = None
    print(type_of_analysis)
    print('Multiclass: {}'.format(multiclass))

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
            cat_vars = df_dtype.variable_name[(df_dtype.data_type == 'Nominal')]
            ord_vars = df_dtype.variable_name[(df_dtype.data_type == 'Ordinal')].values.tolist()
            invl_vars = df_dtype.variable_name[(df_dtype.data_type == 'Scale')].values.tolist()
            other_vars = ord_vars + invl_vars
            col_encoded = df.columns

            # one-hot-encoder categorical variables
            # if cat_vars.size > 0:
            #     tmp = pd.get_dummies(df, prefix=cat_vars, columns=cat_vars, dummy_na=False)
            #     col_encoded = tmp.columns
            if type_of_analysis == 'pureanxiety':
                index = y.index.values
                id_pureanxiety = (y.pureanxiety == "pure_anxiety->pure_anxiety") | \
                                 (y.pureanxiety == "pure_anxiety->nothing")
                print(y.persistance_anxiety.value_counts())
                train = index[id_pureanxiety]
                test = index[~id_pureanxiety]
                res = one_cv_run(df, y.persistance_anxiety.values.squeeze(), train, test, 1, cat_vars, other_vars,
                                 random_state, n_jobs_rf=n_jobs)
            else:
                if type_of_analysis == 'any_disease':
                    id_other = (y.pureanxiety == 'comorbid->pure_other') | \
                               (y.pureanxiety == 'pure_anxiety->pure_other')
                    y.persistance_anxiety.loc[id_other] = 1

                elif type_of_analysis == 'remove_other_disease':
                    id_other = (y.pureanxiety == 'comorbid->pure_other') | \
                               (y.pureanxiety == 'pure_anxiety->pure_other')
                    id_keep = ~id_other
                    y = y.loc[id_keep]
                    y.reset_index(drop=True, inplace=True)
                    df = df.loc[id_keep]
                    df.reset_index(drop=True, inplace=True)

                elif type_of_analysis == 'ignore_other_disease':
                    id_other = (y.pureanxiety == 'comorbid->pure_other') | \
                               (y.pureanxiety == 'pure_anxiety->pure_other')
                    ignore_in_train = id_other.index[id_other].values

                elif type_of_analysis == 'multiclass':
                    id_other = (y.pureanxiety == 'comorbid->pure_other') | \
                               (y.pureanxiety == 'pure_anxiety->pure_other')
                    id_anxiety = y.persistance_anxiety == 1
                    y.loc[id_anxiety, 'persistance_anxiety'] = 2
                    y.loc[id_other, 'persistance_anxiety'] = 1

                elif type_of_analysis == 'multiclass_wicher':
                    id_other = (y.pureanxiety == 'comorbid->pure_other') | \
                               (y.pureanxiety == 'pure_anxiety->pure_other') | \
                               (y.pureanxiety == 'pure_anxiety->comorbid') | \
                               (y.pureanxiety == 'comorbid->comorbid')
                    id_anxiety = y.persistance_anxiety == 1
                    y.loc[id_anxiety, 'persistance_anxiety'] = 2
                    y.loc[id_other, 'persistance_anxiety'] = 1

                y = y.persistance_anxiety
                print(y.value_counts())
                cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=random_state)

                res = Parallel(n_jobs=n_jobs, verbose=1, max_nbytes='50M')(delayed(one_cv_run)(df,
                                                                                               y.values.squeeze(),
                                                                                               train,
                                                                                               test,
                                                                                               i_cv + 1,
                                                                                               cat_vars,
                                                                                               other_vars,
                                                                                               random_state,
                                                                                               10,
                                                                                               multiclass,
                                                                                               ignore_in_train)
                                                                           for i_cv, (train, test) in enumerate(cv.split(df, y)))

            scores_test = res[0][0].keys()
            scores_train = res[0][-1].keys()
            df_scores = pd.DataFrame(columns=scores_test)
            df_perf_train = pd.DataFrame(columns=scores_train)
            df_features = pd.DataFrame(data=col_encoded.values.astype(str), columns=['var_name'])

            if type_of_analysis == 'pureanxiety':
                scores_test, df_predictions, features, scores_train = res
                df_scores = df_scores.append(scores_test, ignore_index=True)
                df_perf_train = df_perf_train.append(scores_train, ignore_index=True)
                assert np.all(df_features.var_name == features.var_name), 'Mismatch in features'
                df_features['cv_1'] = features.var_importance.values
            else:
                for i_cv in range(cv.get_n_splits(df, y)):
                    scores_cv, predictions, features, scores_train = res[i_cv]
                    # scores_cv, predictions = res[i_cv]
                    df_scores = df_scores.append(scores_cv, ignore_index=True)
                    df_perf_train = df_perf_train.append(scores_train, ignore_index=True)
                    if i_cv == 0:
                        df_predictions = predictions
                    else:
                        df_predictions = df_predictions.append(predictions, ignore_index=True)
                    assert np.all(df_features.var_name == features.var_name), 'Mismatch in features'
                    df_features['cv_{}'.format(i_cv + 1)] = features.var_importance.values

            df_scores.to_csv(save_pattern.format('performance_test.csv'), index=False)
            df_perf_train.to_csv(save_pattern.format('performance_train.csv'), index=False)
            df_features.to_csv(save_pattern.format('var_importance.csv'), index=False)
            df_predictions.to_csv(save_pattern.format('predictions.csv'), index=False)
            print('Test-Set:')
            print(df_scores.mean())
            print("Training-Set:")
            print(df_perf_train.mean())
    np.savez('seed.npz', np.array([seed]))


if __name__ == '__main__':
    # folder_names = ['rfc_invd_scores_300trees_rsearch_30', 'rfc_sum_scores_300trees_rsearch_30']
    # folder_names = ['rfc_invd_scores_300trees_rsearch_30_simpleSearch',
    #                 'rfc_sum_scores_300trees_rsearch_30_simpleSearch']
    # folder_names = ['rfc_indv_scores_300trees_rsearch_30_pureanxiety']
    # folder_names = ['rfc_indv_scores_300trees_rsearch_30_any_disease',
    #                 'rfc_indv_scores_300trees_rsearch_30_remove_other_disease']
    # folder_names = ['rfc_indv_scores_300trees_rsearch_30_remove_other_disease']
    folder_names = ['rfsc_invd_scores_300trees_rsearch_30_simpleSearch']
    # folder_names = ['rfc_indv_scores_300trees_rsearch_30_any_disease']
    use_summary = False
    # type_of_analysis = 'Standard'
    # type_of_analysis = 'pureanxiety'
    # type_of_analysis = ['any_disease', 'remove_other_disease']
    # type_of_analysis = ['remove_other_disease']
    type_of_analysis = ['Standard']
    # type_of_analysis = ['any_disease']

    n_jobs = 5
    domains = ['IA', 'IIA', 'IIIA', 'IVA', 'VA']
    combination_length = (1, 2, 3, 4, 5)

    for i in range(len(folder_names)):
        print(folder_names[i])
        print()
        run(save_folder=folder_names[i], use_summary=use_summary, n_jobs=n_jobs, domains=domains,
            type_of_analysis=type_of_analysis[i], combination_length=combination_length)
