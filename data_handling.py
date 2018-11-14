import numpy as np
import pandas as pd


NESDA_FILE = '/data/pzhutovsky/NESDA_anxiety/nesda_anxiety_file_numbers.csv'
NESDA_FILE_DTYPE = '/data/pzhutovsky/NESDA_anxiety/nesda_anxiety_data_types.csv'
NESDA_FILE_MISSING = '/data/pzhutovsky/NESDA_anxiety/nesda_anxiety_with_missing_NaN.csv'
NESDA_FILE_MISSING_DTYPE = '/data/pzhutovsky/NESDA_anxiety/nesda_anxiety_data_types_with_missing_removed.csv'


def load_data(target_col='persistance_anxiety', col_to_remove=['Pident', 'persistance_anxiety', 'persistance_any',
                                                               'cbaiscal', 'pureanxiety', 'Class', 'cduration',
                                                               'ctot32mv']):
    df = pd.read_csv(NESDA_FILE).astype(np.float)
    df_dtypes = pd.read_csv(NESDA_FILE_DTYPE)
    y = df[target_col]
    df.drop(col_to_remove, axis='columns', inplace=True)
    df_dtypes = remove_var_names(df_dtypes, col_to_remove)
    print(df.shape)
    print(y.value_counts())
    return df, df_dtypes, y


def check_for_missing(df, df_dtypes, missing_value=-9, replace_missing=False, threshold_to_drop_perc=None):
    n_subj = df.shape[0]
    missing_per_feature = (df == missing_value).sum(axis=0).values
    missing_per_feature_perc = missing_per_feature / n_subj * 100

    if threshold_to_drop_perc:
        id_drop = missing_per_feature_perc > threshold_to_drop_perc
        col_drop = df.columns[id_drop]
        assert np.all(col_drop == df_dtypes.variable_name[id_drop]), 'dtypes df and data df do not align'
        print('Dropped variables (n={}):'.format(id_drop.sum()))
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
        to_replace = dict()
        for i, column_name in enumerate(col_var_type):
            to_replace[column_name] = {not_applicable_code: 0}
    else:
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


def missing_data_handling(load_df='', load_df_dtypes='', save_df=NESDA_FILE_MISSING,
                          save_df_dtypes=NESDA_FILE_MISSING_DTYPE):
    """
    This repeats the missing data handling outlined in the 01-investigating-missing-data notebook and just reproduces it
    in a script. All the reasoning/comments can be found in the notebook.
    :return: df_missing_handled, df_dtype_missing_handled, y
    """

    if load_df and load_df_dtypes:
        return pd.read_csv(load_df), pd.read_csv(load_df_dtypes)
    else:
        df, df_dtypes, y = load_data()
        print('"IA271_01" "not applicable" values will be considered for imputation according to discussion with '
              'Wicher')
        df.loc[df['IA271_01'] == -8, 'IA271_01'] = -9

        missing_values = [-9, -8, -7]
        for missing_value in missing_values:
            replace_missing = (missing_value == -9) or (missing_value == -7)
            df, df_dtypes = check_for_missing(df, df_dtypes, missing_value=missing_value,
                                              replace_missing=replace_missing, threshold_to_drop_perc=20)

        # replace remaining -8 (not applicable) values
        df, df_dtypes = replace_not_applicable(df, df_dtypes, var_type='Nominal')
        df, df_dtypes = replace_not_applicable(df, df_dtypes, var_type='Ordinal')
        df, df_dtypes = replace_not_applicable(df, df_dtypes, var_type='Scale')

        # cleanup: remove constant variables
        var_std = df.std(axis=0).values
        id_constant = var_std == 0
        var_const = df.columns[id_constant]
        print('Variables with constant value: {}'.format(var_const))
        df.drop(var_const, axis='columns', inplace=True)
        df_dtypes = remove_var_names(df_dtypes, var_const)

        print('Drop "IIIA204_13", "IIIA204_14", and "IIIA204_15" in accordance with discussion with Wicher')
        col_drop = ['IIIA204_13', 'IIIA204_14', 'IIIA204_15']
        df.drop(col_drop, axis='columns', inplace=True)
        df_dtypes = remove_var_names(df_dtypes, col_drop)

        assert np.all(df.columns == df_dtypes.variable_name), 'Mismatch between dtypes and df DataFrames'
        print('Final shape: {}'.format(df.shape))

        # add back the target variable
        df[y.name] = y.values
        df.to_csv(save_df, index=False)
        df_dtypes.to_csv(save_df_dtypes, index=False)

        return df, df_dtypes


def get_data(target_var='persistance_anxiety', load_df='', load_df_dtypes='', modality_name='all',
             summary_scores=False):
    df, df_dtypes = missing_data_handling(load_df=load_df, load_df_dtypes=load_df_dtypes)
    y = df[target_var]
    df.drop(target_var, axis='columns', inplace=True)
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

    if summary_scores:
        col_names = [col_name for col_name in col_names if col_name.find('_summary') != -1]
    else:
        col_names = [col_name for col_name in col_names if col_name.find('_summary') == -1]

    df_dtypes = remove_var_names(df_dtypes, np.setdiff1d(var_names, col_names))
    df = df[col_names]
    assert np.all(df.columns == df_dtypes.variable_name).sum(), 'Mismatch in columns'

    return df, df_dtypes, y


if __name__ == '__main__':
    df, df_dtypes, y = get_data(modality_name=['VA'], summary_scores=True)
