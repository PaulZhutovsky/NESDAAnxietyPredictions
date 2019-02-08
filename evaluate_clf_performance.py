import pandas as pd
import numpy as np
import os.path as osp
from itertools import combinations
from ml_classification import DOMAINS


def load_data(folder_results, domain_label):
    performance_file = osp.join(folder_results, '{}_performance_test.csv'.format(domain_label))
    prediction_file = osp.join(folder_results, '{}_predictions.csv'.format(domain_label))
    return pd.read_csv(performance_file), pd.read_csv(prediction_file)


def load_labels(folder_data, nesda_label='nesda_anxiety_labels_with_dropped_subjects.csv'):
    df_labels = pd.read_csv(osp.join(folder_data, nesda_label))
    return df_labels.pureanxiety


def create_tables(folder_results, folder_labels, metrics=('acc', 'sens', 'spec', 'AUC', 'PPV', 'NPV'), cv_folds=10,
                  target_metric='AUC', combination_length=(1, 5)):
    df_detailed_labels = load_labels(folder_data=folder_labels)

    for i_comb, comb_length in enumerate(combination_length):
        print('Max. domain: {}'.format(comb_length))
        all_domains = ['_'.join(dom) for dom in combinations(DOMAINS, comb_length)]
        df_perf_all = pd.DataFrame(index=pd.MultiIndex.from_product([all_domains, metrics]),
                                   columns=['Mean', 'SD', 'SEM', 'P'])
        df_detailed_labels_dist = df_detailed_labels.value_counts()

        pred_correct_clf = []
        target_all = []
        for i_dom, dom in enumerate(all_domains):
            print(dom)
            df_perf, df_pred = load_data(folder_results=folder_results, domain_label=dom)

            df_target = df_perf.loc[:, [target_metric]]
            df_target['domain_used'] = [dom] * df_target.shape[0]
            target_all.append(df_target)

            print('Compute performance')
            df_perf_mean = df_perf.mean()
            df_perf_std = df_perf.std()
            df_perf_sem = df_perf.sem()
            perm_file = osp.join(folder_results, '{}_perf_permutations.csv'.format(dom))
            if osp.exists(perm_file):
                df_perm = pd.read_csv(perm_file)
                df_perf_perm = ((df_perm >= df_perf_mean).sum() + 1) / (df_perm.shape[0] + 1)
            else:
                df_perf_perm = pd.Series(data=[np.nan] * df_perf_mean.size, index=df_perf_mean.index)

            df_perf = pd.concat((df_perf_mean, df_perf_std, df_perf_sem, df_perf_perm), axis=1,
                                keys=['Mean', 'SD', 'SEM', 'P'])
            index_order = df_perf_all.loc[dom].index
            column_order = df_perf_all.loc[dom].columns
            df_perf_all.loc[dom, :] = df_perf.loc[index_order, column_order].values

            print('Compute correctly classified labels')
            n_repeats = int(df_pred.cv_fold.max()/cv_folds)
            start_id, stop_id = 0, cv_folds
            correct_pred_all = []
            for i_cv in range(n_repeats):
                id_cv = (df_pred.cv_fold > start_id) & (df_pred.cv_fold <= stop_id)
                start_id = stop_id
                stop_id += cv_folds
                df_cv = df_pred.loc[id_cv]
                correct_prediction = df_cv.y_true == df_cv.y_pred
                index_correct_prediction = df_cv.test_index.loc[correct_prediction].values.astype(np.int)
                correct_pred_all.append(df_detailed_labels.iloc[index_correct_prediction].value_counts() /
                                        df_detailed_labels_dist * 100)
            df_tmp = pd.concat(correct_pred_all, axis=1, keys=['repeat_{}'.format(i_rep + 1)
                                                               for i_rep in range(n_repeats)])
            df_pred_correct_clf = pd.concat((df_tmp.mean(axis=1), df_tmp.std(axis=1)), keys=['Mean', 'SD'], axis=1)
            pred_correct_clf.append(df_pred_correct_clf)

        print('Compare domains')
        if comb_length != len(DOMAINS):
            df_target_all = pd.concat(target_all, axis=0, ignore_index=True)
            df_target_comp_p = compare_domains(df_target_all, max_domains=comb_length, target_metric=target_metric)
            df_target_comp_p.to_csv(osp.join(folder_results,
                                             'comparison_between_domains_{}_max_domain{}_p.csv'.format(target_metric,
                                                                                                       comb_length)))
            # diagonal of p-values matrix is all NaN: we don't test against ourselves
            alpha_corrected = 0.05 / (df_target_comp_p.size - df_target_comp_p.isnull().sum().sum())
            sign = np.zeros(df_target_comp_p.shape, dtype=str)
            sign.fill('-')
            sign[np.diag_indices(sign.shape[0])] = ''
            sign[df_target_comp_p < alpha_corrected] = '*'
            df_target_comp_significance = pd.DataFrame(data=sign, columns=df_target_comp_p.columns,
                                                       index=df_target_comp_p.index)
            df_target_comp_significance.to_csv(osp.join(
                folder_results, 'comparison_between_domains_{}_max_domain{}_significance.csv'.format(target_metric,
                                                                                                     comb_length)))

        print('Store files')

        df_perf_all.to_csv(osp.join(folder_results,
                                    'performance_classification_max_domain{}.csv'.format(comb_length)))
        df_pred_correct_clf_all = pd.concat(pred_correct_clf, axis=1, keys=all_domains, sort=True)
        df_pred_correct_clf_all.to_csv(
            osp.join(folder_results, 'prediction_correctly_classified_subj_max_domain{}.csv'.format(comb_length)))

        print('')


def compare_domains(df, max_domains=1, target_metric='AUC', n_perm=10000):
    domain_order = {1: ['IA', 'IIA', 'IIIA', 'IVA', 'VA'],
                    2: ['IA_IIA', 'IA_IIIA', 'IA_IVA', 'IA_VA', 'IIA_IIIA', 'IIA_IVA', 'IIA_VA'],
                    3: ['IA_IIA_IIIA', 'IA_IIA_IVA', 'IA_IIA_VA', 'IA_IIIA_IVA', 'IA_IIIA_VA', 'IA_IVA_VA',
                        'IIA_IIIA_IVA', 'IIA_IIIA_VA', 'IIA_IVA_VA', 'IIIA_IVA_VA'],
                    4: ['IA_IIA_IIIA_IVA', 'IA_IIA_IIIA_VA', 'IA_IIA_IVA_VA', 'IA_IIIA_IVA_VA', 'IIA_IIIA_IVA_VA']}
    domain_used = domain_order[max_domains]
    df_p_values = pd.DataFrame(columns=domain_used, index=domain_used)

    for domain in domain_used:
        metric_target = df.loc[df.domain_used == domain, target_metric].values
        domain_others = np.setdiff1d(domain_used, [domain])
        for domain_other in domain_others:
            metric_other = df.loc[df.domain_used == domain_other, target_metric].values
            _, p_value = sign_difference_test(metric_target, metric_other, n_perm=n_perm)
            df_p_values.loc[domain, domain_other] = p_value
    return df_p_values


def sign_difference_test(target_scores, other_scores, n_perm=10000):
    diff_target_other = target_scores - other_scores
    neutr_perm = np.mean(diff_target_other) / np.std(diff_target_other)

    random_signs = np.random.randint(0, 2, size=(target_scores.size, n_perm))
    random_signs[random_signs == 0] = -1
    permuted_diff_target_other = random_signs * diff_target_other[:, np.newaxis]
    permuted_difference = np.mean(permuted_diff_target_other, axis=0) / np.std(permuted_diff_target_other, axis=0)
    p_value = (np.sum(permuted_difference >= neutr_perm) + 1) / (n_perm + 1)
    return neutr_perm, p_value


def run(data_folder, result_folder, metrics, target_metric, combination_length, cv_folds=10):
    create_tables(folder_results=result_folder, folder_labels=data_folder, metrics=metrics, target_metric=target_metric,
                  combination_length=combination_length, cv_folds=cv_folds)


if __name__ == '__main__':
    folder_names = ['/data/pzhutovsky/NESDA_anxiety/rfc_indv_scores_1000trees_balanced_any_anxiety',
                    '/data/pzhutovsky/NESDA_anxiety/rfc_indv_scores_1000trees_balanced_any_disorder',
                    '/data/pzhutovsky/NESDA_anxiety/rfc_indv_scores_1000trees_balanced_multiclass']
    metrics = [('acc', 'sens', 'spec', 'AUC', 'PPV', 'NPV'),
               ('acc', 'sens', 'spec', 'AUC', 'PPV', 'NPV'),
               ('acc_overall', 'acc_comorbid', 'acc_anxiety', 'acc_other', 'acc_nothing', 'auc_one_vs_rest',
                'auc_micro', 'auc_macro', 'auc_weighted', 'auc_samples')]
    target_metric = ['AUC', 'AUC', 'auc_one_vs_rest']
    combination_length = [1, 5]
    data_folder = '/data/pzhutovsky/NESDA_anxiety'
    cv_folds = 10

    for i, folder_name in enumerate(folder_names):
        print(folder_name)
        run(data_folder=data_folder, result_folder=folder_name, metrics=metrics[i], target_metric=target_metric[i],
            combination_length=combination_length, cv_folds=cv_folds)
