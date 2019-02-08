import os.path as osp

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.stats.multitest import multipletests

from evaluate_clf_performance import sign_difference_test


def run(save_folders, domains, target_metric, type_of_analysis):
    fig1, ax1 = plt.subplots(1, 2, sharey=True, figsize=(20, 10))
    fig2, ax2 = plt.subplots(2, 1, sharex=True, figsize=(20, 10))
    fig3, ax3 = plt.subplots(2, 1, sharex=True, figsize=(6, 11))
    fig4, ax4 = plt.subplots(1, 2, sharex=True, figsize=(13.8, 12))
    for i, save_folder in enumerate(save_folders):
        df_metrics, p_vals = load_perf_data(domains, save_folder, target_metric)
        df_correct_pred = load_pred_data(save_folder, domains)
        df_feat_imp, plot_order = load_feat_imp(save_folder)
        df_significant = compare_domains(df_metrics, domains, target_metric)

        if i == 0:
            ylabel = 'Domains'
            xlabel = ''
            legend = False
        else:
            ylabel = ''
            xlabel = 'Patient trajectory'
            legend = True
        make_figure1(df_metrics, p_vals,  x_axis=target_metric, y_axis='Domains',
                     type_of_analysis=type_of_analysis[i], ax=ax1[i], ylabel=ylabel)
        make_figure2(df_correct_pred, x_axis='Patient trajectory', y_axis='Correctly classified [%]', hue='Domains',
                     xlabel=xlabel, ax=ax2[i], type_of_analysis=type_of_analysis[i], legend=legend)
        make_figure3(df_significant, type_of_analysis=type_of_analysis[i], ax=ax3[i],
                     domain_labels=('IA', 'IIA', 'IIIA', 'IVA', 'VA', 'IA+IIA+IIIA\nIVA+VA'))
        make_figure4(df_feat_imp, type_of_analysis=type_of_analysis[i], ax=ax4[i], x_axis='Feature importance [a.u.]',
                     y_axis='var_name', order=plot_order)

    fig1.tight_layout()
    fig2.tight_layout()
    fig3.tight_layout()
    fig4.tight_layout()
    fig1.savefig('figures/performance.png', dpi=600)
    fig2.savefig('figures/correct_classification.png', dpi=600)
    fig3.savefig('figures/comparison_domains.png', dpi=600)
    fig4.savefig('figures/feature_importances.png', dpi=600)


def load_feat_imp(save_folder):
    df = pd.read_csv(osp.join(save_folder, 'IA_IIA_IIIA_IVA_VA_var_importance.csv'), index_col=0)
    var_names = df.index.values
    df_perm = pd.read_csv(osp.join(save_folder, 'permuted_variable_importances_domains_all.csv'), index_col=0)
    n_cv = df.shape[1]
    n_perm = df_perm.shape[1] + 1

    p_values = np.zeros((var_names.size, n_cv))
    for i_var, var_name in enumerate(var_names):
        null_dist = df_perm.loc[var_name, :].values

        for i_cv in range(n_cv):
            true_val = df.loc[var_name, 'cv_{}'.format(i_cv + 1)]
            p_values[i_var, i_cv] = (np.sum(null_dist >= true_val) + 1) / n_perm

    # multiple testing correction
    reject_all = np.zeros_like(p_values, dtype=bool)
    for i_cv in range(n_cv):
        p_values_cv = p_values[:, i_cv]
        reject_all[:, i_cv], _, _, _ = multipletests(p_values_cv, alpha=0.05, method='fdr_tsbh')

    reject_H0 = reject_all.sum(axis=1) > (n_cv / 2)

    var_names = var_names[reject_H0]

    data = []
    labels = []
    for var in var_names:
        data.append(df.loc[var, :].values)
        labels.append([var] * data[-1].size)

    data = np.concatenate(data)
    labels = np.concatenate(labels)
    df_all = pd.DataFrame(data=np.column_stack((data, labels)), columns=['Feature importance [a.u.]', 'var_name'])
    df_all['Feature importance [a.u.]'] = df_all['Feature importance [a.u.]'].astype(np.float)
    plot_order = df_all.groupby(['var_name']).mean().sort_values('Feature importance [a.u.]', ascending=False).index
    plot_order = plot_order.values
    return df_all, plot_order


def load_pred_data(save_folder, domains):
    df_dim1 = pd.read_csv(osp.join(save_folder, 'prediction_correctly_classified_subj_max_domain1.csv'), header=[0, 1],
                          index_col=0)
    df_dim5 = pd.read_csv(osp.join(save_folder, 'prediction_correctly_classified_subj_max_domain5.csv'), header=[0, 1],
                          index_col=0)
    df_all = pd.concat((df_dim1, df_dim5), axis=1)
    idx = pd.IndexSlice
    df_mean = df_all.loc[:, idx[:, 'Mean']]
    categories = df_mean.index.values

    data = []
    labels = []
    doms = []
    for dom in domains:
        data.append(df_mean[dom].values)
        labels.append(categories)
        doms.append([dom] * len(data[-1]))
    data = np.concatenate(data)
    labels = np.concatenate(labels)
    doms = np.concatenate(doms)

    data = np.column_stack((data, labels, doms))
    df = pd.DataFrame(data=data, columns=['Correctly classified [%]', 'Patient trajectory', 'Domains'])
    return df


def compare_domains(df_perf, domains, target_metric, alpha=0.05):
    df_sign = pd.DataFrame(columns=domains, index=domains)
    alpha /= (len(domains) * len(domains)) - len(domains)
    for dom_compare in domains:
        id_dom = df_perf.Domains == dom_compare
        metric_to_compare = df_perf.loc[id_dom, target_metric].values
        dom_others = np.setdiff1d(domains, dom_compare)

        for dom_other in dom_others:
            id_other_dom = df_perf.Domains == dom_other
            metric_other = df_perf.loc[id_other_dom, target_metric].values
            _, p_value = sign_difference_test(metric_to_compare, metric_other, n_perm=10000)

            if p_value < alpha:
                df_sign.loc[dom_compare, dom_other] = 1
            else:
                df_sign.loc[dom_compare, dom_other] = 0
    return df_sign.astype(np.float)


def make_figure1(df_metrics, p_vals, ax, x_axis, y_axis, type_of_analysis, ylabel):
    to_replace = {'IA_IIA_IIIA_IVA_VA': 'IA+IIA+IIIA\nIVA+VA'}
    df_metrics[y_axis].replace(to_replace=to_replace, inplace=True)
    placement = {'IA': 0, 'IIA': 1, 'IIIA': 2, 'IVA': 3, 'VA': 4, 'IA_IIA_IIIA_IVA_VA': 5}
    sns.violinplot(x=x_axis, y=y_axis, data=df_metrics, color='white', inner='quartile', linewidth=2, ax=ax)
    sns.swarmplot(x=x_axis, y=y_axis, data=df_metrics, color='k', ax=ax)

    x_place = 0.89
    for key, val in p_vals.items():
        y_place = placement[key]
        ax.text(x_place, y_place, '*', fontsize=26, ha='center', va='center', fontweight='bold')

    ax.set_xticks([0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9])
    ax.tick_params(axis='both', labelsize=16)
    ax.axvline(0.5, 0, 1, lw=2, ls='--', c='k')
    ax.set_xlim((0.4, 0.9))
    ax.grid(True, axis='x')
    ax.set_xlabel(x_axis, fontsize=20)
    ax.set_ylabel(ylabel, fontsize=20)
    ax.set_title('Predicting *{}* at follow-up'.format(type_of_analysis), fontsize=20)


def make_figure2(df_correct_pred, x_axis, y_axis, hue, xlabel, ax, type_of_analysis, legend):
    sns.barplot(x=x_axis, y=y_axis, hue=hue, data=df_correct_pred, ax=ax, ci=None)
    ax.tick_params(axis='both', labelsize=16)
    ax.tick_params(axis='x', labelrotation=20)
    if legend:
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, fontsize=18, bbox_to_anchor=(1.02, 1), loc='upper left')
    else:
        ax.legend([])
    ax.grid(True, axis='y')
    ax.set_ylim([0, 88])
    ax.set_xlabel(xlabel, fontsize=20)
    ax.set_ylabel(y_axis, fontsize=20)
    ax.set_title('Predicting *{}* at follow-up'.format(type_of_analysis), fontsize=20)


def make_figure3(df_significant, type_of_analysis, ax, domain_labels):
    sns.heatmap(df_significant, annot=False, cbar=False, ax=ax, xticklabels=domain_labels, yticklabels=domain_labels,
                linewidths=0.5)
    ax.set_title('Comparison between domains\n(*{}* at follow-up)'.format(type_of_analysis), fontsize=20)
    ax.tick_params(axis='both', labelsize=16)
    ax.tick_params(axis='y', labelrotation=0)
    ax.tick_params(axis='x', labelrotation=30)


def make_figure4(df_feat_imp, type_of_analysis, ax, x_axis, y_axis, order):
    sns.barplot(x=x_axis, y=y_axis, data=df_feat_imp, ci='sd', ax=ax, color='c', order=order)
    ax.set_ylabel('')
    ax.tick_params(axis='both', labelsize=14)
    title_string = 'Predicting *{}* at follow-up'.format(type_of_analysis)
    ax.set_title(title_string, fontsize=20)
    ax.set_xlabel(x_axis, fontsize=20)


def load_perf_data(domains, save_folder, target_metric):
    data_dict = {}
    p_vals = {}
    for domain in domains:
        df_metric = pd.read_csv(osp.join(save_folder, '{}_performance_test.csv'.format(domain)))[target_metric]
        data_dict[domain] = df_metric

        if osp.exists(osp.join(save_folder, '{}_perf_permutations.csv'.format(domain))):
            metric_true = df_metric.mean()
            metric_perm = pd.read_csv(osp.join(save_folder,
                                               '{}_perf_permutations.csv'.format(domain)))[target_metric].values
            p_vals[domain] = permutation_test(metric_true, metric_perm)
    metric_all = np.concatenate([np.column_stack((val.values, [key] * val.size)) for key, val in data_dict.items()],
                                axis=0)
    df_metric_all = pd.DataFrame(data=metric_all, columns=[target_metric, 'Domains'])
    df_metric_all[target_metric] = df_metric_all[target_metric].astype(np.float)
    return df_metric_all, p_vals


def permutation_test(val_true, val_perm):
    return (np.sum(val_perm >= val_true) + 1) / (val_perm.size + 1)


if __name__ == '__main__':
    SAVE_FOLDERS = ['/data/pzhutovsky/NESDA_anxiety/rfc_indv_scores_1000trees_balanced_any_anxiety',
                    '/data/pzhutovsky/NESDA_anxiety/rfc_indv_scores_1000trees_balanced_any_disorder']
    analysis_type = ['anxiety', 'any disorder']
    TARGET_METRIC = 'AUC'
    DOMAINS = ['IA', 'IIA', 'IIIA', 'IVA', 'VA', 'IA_IIA_IIIA_IVA_VA']

    run(SAVE_FOLDERS, DOMAINS, target_metric=TARGET_METRIC, type_of_analysis=analysis_type)
