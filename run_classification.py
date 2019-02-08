import warnings

from sklearn.exceptions import UndefinedMetricWarning

from ml_classification import run_ml

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

SAVE_FOLDER = ['rfc_indv_scores_1000trees_balanced_any_anxiety',
               'rfc_indv_scores_1000trees_balanced_any_disorder',
               'rfc_indv_scores_1000trees_balanced_multiclass']
TYPE_OF_ANALYSIS = ['any_anxiety', 'any_disorder', 'multiclass']
N_JOBS = 15
N_JOBS_RF = 3
N_PERM = 1000
USE_SUMMARY = False
COMB_LENGTH = (1, 5)
DOMAINS = ['IA', 'IIA', 'IIIA', 'IVA', 'VA']
TARGET_METRIC = ['AUC', 'AUC', 'auc_one_vs_rest']
TARGET_METRIC_THRESH = 0.6
CATEGORICAL_ENCODING = None

for i in range(len(SAVE_FOLDER)):
    print('{}/{}: Analysis: {}; Folder: {}'.format(i + 1, len(SAVE_FOLDER), TYPE_OF_ANALYSIS[i], SAVE_FOLDER[i]))
    run_ml(save_folder=SAVE_FOLDER[i], domains=DOMAINS, n_jobs=N_JOBS, use_summary=USE_SUMMARY,
           type_of_analysis=TYPE_OF_ANALYSIS[i], combination_length=COMB_LENGTH, target_metric=TARGET_METRIC[i],
           target_thresh=TARGET_METRIC_THRESH, n_perm=N_PERM, n_jobs_rf=N_JOBS_RF, cat_encoding=CATEGORICAL_ENCODING)
