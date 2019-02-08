import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os.path as osp


NESDA_FILE = '/data/pzhutovsky/NESDA_anxiety/nesda_anxiety_file_numbers.csv'
NESDA_FILE_DTYPE = '/data/pzhutovsky/NESDA_anxiety/nesda_anxiety_data_types.csv'
NESDA_FILE_MISSING = '/data/pzhutovsky/NESDA_anxiety/nesda_anxiety_with_missing_NaN.csv'
NESDA_FILE_MISSING_DTYPE = '/data/pzhutovsky/NESDA_anxiety/nesda_anxiety_data_types_with_missing_removed.csv'
NESDA_FILE_MISSING_SUMMARY = '/data/pzhutovsky/NESDA_anxiety/nesda_anxiety_summary_with_missing_NaN.csv'
NESDA_FILE_MISSING_SUMMARY_DTYPE = '/data/pzhutovsky/NESDA_anxiety/' \
                                   'nesda_anxiety_data_types_summary_with_missing_removed.csv'
NESDA_FILE_LABELS = '/data/pzhutovsky/NESDA_anxiety/nesda_anxiety_labels_with_dropped_subjects.csv'
NESDA_FIGURES_FOLDER = '/data/pzhutovsky/NESDA_anxiety/figures'


def load_data(nesda_file=NESDA_FILE, nesda_file_dtype=NESDA_FILE_DTYPE, target_col='persistance_anxiety',
              col_to_remove=('Pident', 'persistance_anxiety', 'persistance_any', 'cbaiscal', 'pureanxiety', 'Class',
                             'cduration', 'ctot32mv'),
              col_to_keep=('pureanxiety', 'Pident'),
              print_diagnostic=True):

    df = pd.read_csv(nesda_file).astype(np.float)
    df_dtypes = pd.read_csv(nesda_file_dtype)
    y = df[[target_col] + list(col_to_keep)].copy()
    y.replace(to_replace={'pureanxiety': {-6: 'comorbid->pure_other',
                                           -5: 'comorbid->comorbid',
                                           -4: 'comorbid->nothing',
                                           -3: 'comorbid->pure_anxiety',
                                           -2: 'pure_anxiety->pure_other',
                                           -1: 'pure_anxiety->comorbid',
                                            0: 'pure_anxiety->nothing',
                                            1: 'pure_anxiety->pure_anxiety'
                                          }
                          }, value=None, inplace=True)
    df.drop(list(col_to_remove), axis='columns', inplace=True)
    df_dtypes = remove_var_names(df_dtypes, col_to_remove)

    if print_diagnostic:
        print(df.shape)
        print(y[target_col].value_counts())
        print(y['pureanxiety'].value_counts())

    return df, df_dtypes, y


def check_for_missing(df, df_dtypes, missing_value=-9, replace_missing=False, threshold_to_drop_perc=None):
    n_subj = df.shape[0]
    missing_per_feature = (df == missing_value).sum(axis=0).values
    missing_per_feature_perc = missing_per_feature / n_subj * 100

    if threshold_to_drop_perc:
        id_drop = missing_per_feature_perc > threshold_to_drop_perc
        col_drop = df.columns[id_drop]
        assert np.all(col_drop == df_dtypes.variable_name[id_drop]), 'dtypes df and data df do not align'
        print('Dropped variables (n={}):'.format(col_drop.size))
        print('{}'.format(col_drop))
        print('Percentage missing (value={}): {}'.format(missing_value, missing_per_feature_perc[id_drop]))
        df.drop(col_drop, axis='columns', inplace=True)
        df_dtypes = remove_var_names(df_dtypes, col_drop)

    if replace_missing:
        df.replace(to_replace=missing_value, value=np.nan, inplace=True)

    return df, df_dtypes


def remove_var_names(df_dtypes, var_names_remove):
    df_dtypes = df_dtypes.loc[~df_dtypes.variable_name.isin(var_names_remove)]
    df_dtypes.reset_index(drop=True, inplace=True)
    return df_dtypes


def replace_not_applicable(df, df_dtypes, var_type='Nominal', not_applicable_code=-8):
    col_var_type = df_dtypes.variable_name.loc[df_dtypes.data_type == var_type].values
    assert col_var_type.size > 0, 'Wrong variable type: {}'.format(var_type)

    if (var_type == 'Ordinal') or (var_type == 'Scale'):
        # replace by 0
        to_replace = dict()
        for i, column_name in enumerate(col_var_type):
            to_replace[column_name] = {not_applicable_code: 0}
    else:
        # replace by creating a new category
        to_replace = dict()
        for i, column_name in enumerate(col_var_type):
            to_replace[column_name] = {not_applicable_code: df[column_name].max() + 1}

    df.replace(to_replace=to_replace, value=None, inplace=True)
    return df, df_dtypes


def extract_single_modality(col_names, modality_name):
    modality_names_allowed = ['IA', 'IIA', 'IIIA', 'IVA', 'VA']

    if modality_name in modality_names_allowed:
        return col_names[np.char.find(col_names, modality_name) == 0]
    else:
        raise RuntimeError('Only {} allowed. You chose {}'.format(modality_names_allowed, modality_name))


def drop_subj_with_many_missing(df, y, target_var, missing_thresh=0.2, missing_val=-9):
    missing_proportion = (df == missing_val).sum(axis=1)/df.shape[1]
    id_keep = missing_proportion <= missing_thresh
    id_drop = ~id_keep
    df = df.loc[id_keep]
    df.reset_index(drop=True, inplace=True)
    y = y.loc[id_keep]
    y = y.reset_index(drop=False)
    print('Dropping subjects with more than {} missing features: n={}'.format(missing_thresh, id_drop.sum()))
    print('Percentage of missing variables: {}'.format(missing_proportion[id_drop].values * 100))
    print('New shape: {}'.format(df.shape))
    print(y[target_var].value_counts())
    return df, y


def missing_data_handling(target_col='persistance_anxiety', load_df='', load_df_dtypes='', load_df_labels='',
                          save_df=NESDA_FILE_MISSING, save_df_dtypes=NESDA_FILE_MISSING_DTYPE,
                          save_df_labels=NESDA_FILE_LABELS, make_plots=False, figures_folder=NESDA_FIGURES_FOLDER):
    """
    This repeats the missing data handling outlined in the 01-investigating-missing-data notebook and just reproduces it
    in a script. All the reasoning/comments can be found in the notebook.
    :return: df_missing_handled, df_dtype_missing_handled, y
    """

    if load_df and load_df_dtypes and load_df_labels:
        return pd.read_csv(load_df), pd.read_csv(load_df_dtypes), pd.read_csv(load_df_labels)[target_col]
    else:
        df, df_dtypes, y = load_data(target_col=target_col)
        print('"IA271_01" "not applicable" values will be considered for imputation according to discussion with '
              'Wicher')
        df.loc[df['IA271_01'] == -8, 'IA271_01'] = -9

        if make_plots:
            fig, ax = plt.subplots(2, 2, figsize=(15, 12))
            ax = ax.ravel()
            missing_vals = [-9, -8, -7]

            for i in range(len(missing_vals)):
                missing = (df == missing_vals[i]).sum(axis=1)/df.shape[1] * 100
                id_missing = missing > 0
                missing = missing[id_missing]
                n_missing = id_missing.sum()
                ax[i].hist(missing.values, bins=30)
                ax[i].set_xticks(np.arange(0, 110, 10))
                ax[i].tick_params(axis='both', which='major', labelsize=18)
                ax[i].set_xlim([0, 100])
                ax[i].set_xlabel('%Missing value: {}'.format(missing_vals[i]), fontsize=18)
                ax[i].set_ylabel('#Subjects', fontsize=18)
                ax[i].set_title('#Subjects with at least 1 missing ({}): {}/{}'.format(missing_vals[i], n_missing,
                                                                                       df.shape[0]), fontsize=20)
                ax[i].axvline(20, 0, 1, lw=2, c='k')

            missing_total = (df == -9) | (df == -8) | (df == -7)
            missing_total = missing_total.sum(axis=1)/df.shape[1] * 100
            id_missing_total = missing_total > 0
            missing_total = missing_total[id_missing_total]
            n_missing_total = id_missing_total.sum()
            ax[3].hist(missing_total.values, bins=30)
            ax[3].set_xlim([0, 100])
            ax[3].set_xticks(np.arange(0, 110, 10))
            ax[3].tick_params(axis='both', which='major', labelsize=18)
            ax[3].set_xlabel('%Any missing', fontsize=18)
            ax[3].set_ylabel('#Subjects', fontsize=18)
            ax[3].set_title('#Subjects with at least 1 missing (any): {}/{}'.format(n_missing_total, df.shape[0]),
                            fontsize=20)
            ax[3].axvline(20, 0, 1, lw=2, c='k')
            fig.tight_layout()
            fig.savefig(osp.join(figures_folder, 'missing_value_across_subjects.png'))

        df, y = drop_subj_with_many_missing(df, y, target_var=target_col, missing_thresh=0.2)

        var_to_recode = ['IVA351_01', 'IVA351_02', 'IVA351_04', 'IVA351_05']
        val_to_recode = len(var_to_recode) * [{-8: 0}]
        to_replace = dict(zip(var_to_recode, val_to_recode))
        print('According to communication with Wicher {} will be recoded to 0 (from -8)'.format(var_to_recode))
        df.replace(to_replace=to_replace, value=None, inplace=True)

        if make_plots:
            fig, ax = plt.subplots(2, 2, figsize=(15, 12))
            ax = ax.ravel()
            missing_vals = [-9, -8, -7]

            for i in range(len(missing_vals)):
                missing = (df == missing_vals[i]).sum(axis=0)/df.shape[0] * 100
                id_missing = missing > 0
                missing = missing[id_missing]
                n_missing = id_missing.sum()
                ax[i].hist(missing.values, bins=30)
                ax[i].set_xticks(np.arange(0, 110, 10))
                ax[i].tick_params(axis='both', which='major', labelsize=18)
                ax[i].set_xlim([0, 100])
                ax[i].set_xlabel('%Missing value: {}'.format(missing_vals[i]), fontsize=18)
                ax[i].set_ylabel('#Variables', fontsize=18)
                ax[i].set_title('#Variable with at least 1 missing ({}): {}/{}'.format(missing_vals[i], n_missing,
                                                                                       df.shape[1]), fontsize=20)
                ax[i].axvline(20, 0, 1, lw=2, c='k')

            missing_total = (df == -9) | (df == -8) | (df == -7)
            missing_total = missing_total.sum(axis=0)/df.shape[0] * 100
            id_missing_total = missing_total > 0
            missing_total = missing_total[id_missing_total]
            n_missing_total = id_missing_total.sum()
            ax[3].hist(missing_total.values, bins=30)
            ax[3].set_xlim([0, 100])
            ax[3].set_xticks(np.arange(0, 110, 10))
            ax[3].tick_params(axis='both', which='major', labelsize=18)
            ax[3].set_xlabel('%Any missing', fontsize=18)
            ax[3].set_ylabel('#Variables', fontsize=18)
            ax[3].set_title('#Variable with at least 1 missing (any): {}/{}'.format(n_missing_total, df.shape[1]),
                            fontsize=20)
            ax[3].axvline(20, 0, 1, lw=2, c='k')
            fig.tight_layout()
            fig.savefig(osp.join(figures_folder, 'missing_value_across_variables.png'))

        missing_values = [-9, -8, -7]
        for missing_value in missing_values:
            replace_missing = (missing_value == -9) or (missing_value == -7)
            df, df_dtypes = check_for_missing(df, df_dtypes, missing_value=missing_value,
                                              replace_missing=replace_missing, threshold_to_drop_perc=20)

        # replace remaining -8 (not applicable) values
        df, df_dtypes = replace_not_applicable(df, df_dtypes, var_type='Nominal')
        df, df_dtypes = replace_not_applicable(df, df_dtypes, var_type='Ordinal')
        df, df_dtypes = replace_not_applicable(df, df_dtypes, var_type='Scale')

        print('Shape after removing variables with to many missing values:')
        print(df.shape)

        # cleanup: remove constant variables
        df_std = df.std(axis=0)
        var_const = df_std[df_std == 0].index
        print('Variables with constant value: {}'.format(var_const))
        df.drop(var_const, axis='columns', inplace=True)
        df_dtypes = remove_var_names(df_dtypes, var_const)

        # According to discussion with Wicher we will NOT remove variables with low variance because there are too many
        # variables in there which should have predictive value
        # df, df_dtypes = remove_low_variance(df, df_dtypes, df_std)

        col_drop = ['IIIA204_13', 'IIIA204_14', 'IIIA204_15']
        print('Drop {} in accordance with discussion with Wicher'.format(col_drop))
        df.drop(col_drop, axis='columns', inplace=True)
        df_dtypes = remove_var_names(df_dtypes, col_drop)
        assert np.all(df.columns == df_dtypes.variable_name), 'Mismatch between dtypes and df DataFrames'

        if make_plots:
            fig, ax = plt.subplots(2, 1, figsize=(10, 15))
            df_missing = df.isnull()
            xlabels = {0: 'Variables',
                       1: 'Subjects'}

            for i in [0, 1]:
                missing_total = df_missing.sum(axis=i)/df.shape[i] * 100
                id_missing_total = missing_total > 0
                missing_total = missing_total[id_missing_total]
                n_missing_total = id_missing_total.sum()
                ax[i].hist(missing_total.values, bins=40)
                ax[i].set_xticks(np.arange(0, 105, 5))
                ax[i].set_xlim([0, 15])
                ax[i].tick_params(axis='both', which='major', labelsize=18)
                ax[i].set_xlabel('%Any missing', fontsize=18)
                ax[i].set_ylabel('#{}'.format(xlabels[i]), fontsize=18)
                ax[i].set_title('#{} with at least 1 missing: {}/{}'.format(xlabels[i], n_missing_total,
                                                                                  df.shape[i]),
                                fontsize=20)
            fig.tight_layout()
            fig.savefig(osp.join(figures_folder, 'missing_value_final.png'))

        print('Final shape: {}'.format(df.shape))
        print('Missing value stats:')
        print('Variables with missing values: n={}'.format((df.isnull().any(axis=0)).sum()))
        missing_perc = df.isnull().sum()/df.shape[0] * 100
        missing_perc = missing_perc[missing_perc != 0]
        print('Missing value stats: Mean={:0.2f}, Median={:0.2f}, SD={:0.2f}, min={:0.2f}, max={:0.2f}'.format(
            missing_perc.mean(), missing_perc.median(), missing_perc.std(), missing_perc.min(), missing_perc.max()))
        print('Missing ranges: 0-1: {}, 1-5: {}, 5-10: {}'.format((missing_perc <= 1).sum(),
                                                                  np.sum((missing_perc > 1) & (missing_perc <= 5)),
                                                                  np.sum((missing_perc > 5))))

        df.to_csv(save_df, index=False)
        y.to_csv(save_df_labels, index=False)
        df_dtypes.to_csv(save_df_dtypes, index=False)
        return df, df_dtypes, y[target_col]


def remove_low_variance(df, df_dtypes, df_std):
    print('Drop variables which do not vary a lot')
    # Interval scale variables:
    col_scale = df.columns[df_dtypes.data_type == 'Scale']
    scale_std = df_std[col_scale]
    var_to_drop_scale = scale_std[scale_std < 0.2].index
    # manually checked histograms and saw that the following variables have low std because of their low feature
    # scale and not because there is no variation in the data. So these variables will be kept for the analysis
    var_to_drop_scale = np.setdiff1d(var_to_drop_scale, ['IVA360_05', 'IVA409_01', 'IVA409_03', 'IVA409_04'])
    print('Interval scale variables (n={}):'.format(var_to_drop_scale.size))
    print(var_to_drop_scale)
    # Ordinal and nominal scale variables:
    col_ordinal = df.columns[df_dtypes.data_type == 'Ordinal']
    col_nominal = df.columns[df_dtypes.data_type == 'Nominal']
    col_nom_ord = np.concatenate((col_ordinal, col_nominal))
    # Counts of the different categories across all ordinal/nominal variables
    rel_freq_nom_ord = df[col_nom_ord].apply(pd.value_counts)
    rel_freq_nom_ord /= (~df[col_nom_ord].isnull()).sum()
    rel_freq_nom_ord *= 100
    # discard all variables where at least 90% of the data belongs to one level
    var_to_drop_nom_ord = col_nom_ord[(rel_freq_nom_ord >= 90).any()]
    print('Ordinal/Nominal scale variables (n={})'.format(var_to_drop_nom_ord.size))
    print(var_to_drop_nom_ord)
    var_to_drop = np.concatenate((var_to_drop_scale, var_to_drop_nom_ord))
    print('Dropped variables: {} (n = {})'.format(var_to_drop, var_to_drop.size))
    df.drop(var_to_drop, axis='columns', inplace=True)
    df_dtypes = remove_var_names(df_dtypes, var_to_drop)
    return df, df_dtypes


def get_summary_df(df, df_dtypes, load_df='', load_df_dtypes='', save_df=NESDA_FILE_MISSING_SUMMARY,
                   save_df_dtype=NESDA_FILE_MISSING_SUMMARY_DTYPE):
    if load_df and load_df_dtypes:
        return pd.read_csv(load_df), pd.read_csv(load_df_dtypes)
    else:
        vars_all = df.columns.values.astype(str)
        vars_to_take = []
        test_used = []
        for var in vars_all:
            test_name = var.partition('_')[0]
            # needed for filtering one test: IA354 and IA354andIA355 would lead to taking a test twice
            test_name += '_'
            if test_name in test_used:
                continue
            all_test_var = df.columns.str.startswith(test_name)
            test_var = vars_all[all_test_var]
            find_summary = np.char.endswith(test_var, 'summary')
            if np.any(find_summary):
                vars_to_take.extend(test_var[find_summary])
            else:
                vars_to_take.extend(test_var)
            test_used.append(test_name)
        df = df[vars_to_take]
        print('New summary df shape: {}'.format(df.shape))
        df_dtypes = df_dtypes.loc[df_dtypes.variable_name.isin(vars_to_take)]
        df_dtypes.reset_index(drop=True, inplace=True)
        assert np.all(df.columns == df_dtypes.variable_name), 'mismatch'

        df.to_csv(save_df, index=False)
        df_dtypes.to_csv(save_df_dtype, index=False)
        return df, df_dtypes


def get_data(load_df='', load_df_summary='', load_df_dtypes='', load_df_dtypes_summary='', load_df_labels='',
             modality_name='all', use_summary=False, target_col='persistance_anxiety', make_plots=False):
    df, df_dtypes, y = missing_data_handling(load_df=load_df, load_df_dtypes=load_df_dtypes, make_plots=make_plots,
                                             load_df_labels=load_df_labels, target_col=target_col)
    if use_summary:
        df, df_dtypes = get_summary_df(df, df_dtypes, load_df=load_df_summary, load_df_dtypes=load_df_dtypes_summary)
    else:
        var_names = df.columns.values
        summary_var = np.array([col for col in var_names if col.endswith('_summary')])
        df.drop(summary_var, axis='columns', inplace=True)
        df_dtypes = remove_var_names(df_dtypes, summary_var)
        print('Shape after dropping summary scores: {}'.format(df.shape))
    var_names = df.columns.values.astype(str)

    if isinstance(modality_name, str):
        if modality_name == 'all':
            col_names = df.columns.values.astype(str)
        else:
            col_names = extract_single_modality(var_names, modality_name)

    elif hasattr(modality_name, '__iter__') and not isinstance(modality_name, str):
        modality_name = sorted(modality_name)
        col_names = []
        for modality in modality_name:
            col_names.extend(extract_single_modality(var_names, modality))
    else:
        raise NotImplementedError

    df_dtypes = remove_var_names(df_dtypes, np.setdiff1d(var_names, col_names))
    df = df[col_names]
    assert np.all(df.columns == df_dtypes.variable_name).sum(), 'Mismatch in columns'
    return df, df_dtypes, y


if __name__ == '__main__':
    _, _, _ = get_data(modality_name='all', use_summary=False, make_plots=False)
