import os
import pandas as pd
import numpy as np
import re
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, confusion_matrix, roc_curve
from matplotlib import pyplot as plt
from miRNA_databases import get_HMDD_disease_associated_db
from scipy.stats import hypergeom
from sklearn.ensemble import AdaBoostClassifier

import pickle
from pathomir import print_results_dict
from prediction import make_ensemble, predict_disease_causality, load_train_test_split, AdaBoostEnsembleModel
from confidence_intervals import delong_roc_variance, delong_significance_between_two_aucs
import scipy.stats
from scipy import stats

#Miscellaneous functions used in the paper

def create_aggregate_cancer_dataset(path_to_miRNA_datas = os.getcwd()):
    run_identifiers = ["gastric", "breast", "hepato"]

    cancer_data = []
    for run_identifier in run_identifiers:
        miRNA_data = pd.read_csv(path_to_miRNA_datas  + "/" + run_identifier + "_miRNA_data.csv")

        cancer_data.append(miRNA_data)

    cancer_data = pd.concat(cancer_data)

    return cancer_data

def create_aggregate_cancer_model(path_to_miRNA_datas = os.getcwd()):
    cancer_data = create_aggregate_cancer_dataset(path_to_miRNA_datas)

    X = cancer_data[['Disease_Influence', 'Network_Influence', 'Conservation']]
    y = cancer_data[['Causality']]

    cancer_aggregate_model = AdaBoostClassifier(random_state=None, n_estimators=1500)
    cancer_aggregate_model.fit(np.asarray(X), np.asarray(y).ravel())

    filepath = os.getcwd() + "/cancer_aggregate_model.pickle"
    with open(filepath, 'wb') as handle:
            pickle.dump(cancer_aggregate_model, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return cancer_aggregate_model

def test_aggregate_cancer_model(path_to_model, path_to_miRNA_datas = os.getcwd(), logging = True):
    cancer_aggregate_model = pd.read_pickle(path_to_model)

    cancer_data = create_aggregate_cancer_dataset(path_to_miRNA_datas)

    X = cancer_data[['Disease_Influence', 'Network_Influence', 'Conservation']]
    y = cancer_data[['Causality']]

    y_pred = cancer_aggregate_model.predict_proba(X)[:,1]
    agg_rocauc = roc_auc_score(y, y_pred)

    alzheimers_miRNA_data = pd.read_csv(path_to_miRNA_datas + "/" + "alzheimers" + "_miRNA_data.csv")
    X_alz = alzheimers_miRNA_data[['Disease_Influence', 'Network_Influence', 'Conservation']]
    y_alz = alzheimers_miRNA_data[['Causality']]

    alz_y_pred = cancer_aggregate_model.predict_proba(X_alz)[:,1]
    alz_rocauc = roc_auc_score(y_alz, alz_y_pred)

    if logging == True:
        print("FEATURE IMPORTANCES: " + str(cancer_aggregate_model.feature_importances_))

        print("AUC ON AGGREGATE DATASET: " + str(agg_rocauc))
        
        print("AUC ON ALZHEIMER'S DATASET: " + str(alz_rocauc))
    
    return alz_y_pred, y_alz

def compare_alz_auc_to_agg_auc(path_to_model, path_to_miRNA_datas = os.getcwd(), path_to_alz_preds = os.getcwd()):
    agg_alz_y_pred, y_alz = test_aggregate_cancer_model(path_to_model, path_to_miRNA_datas, logging = False)
    print(np.array(y_alz))
    print(agg_alz_y_pred)
    alz_y_pred = pd.read_csv(path_to_alz_preds + "/" + "alzheimers" + "_predictions.csv")[['Average_Prob']]

    z, p = delong_significance_between_two_aucs(agg_alz_y_pred, np.array(alz_y_pred), np.array(y_alz))
    print("P-VALUE: " + str(p))
    print("Z-SCORE: " + str(z))

    return z, p

#graph aucs for multiple diseases on same plot
def plot_all_aucs(input_path, run_identifiers, HMDD_disease_names = None, ds = False, filename = "all_aucs", save_path = "disease_data", logging = True):
    """
    Plots all the AUC curves on the same figure

    If you don't want to save the AUC image, set save_path to None
    """

    folder_to_rgb_dict = {
    'breast': "#2b769c",
    'hepato':  "#61d9e6",
    'gastric': "#bda6df",
    'alzheimers': "#c41919"
    }

    folder_to_disease_name_dict = {
    'breast': 'Breast Cancer', 
    'alzheimers': 'Alzheimer\'s Disease',
    'hepato': 'Hepatocellular Cancer',
    'gastric': 'Gastric Cancer', 
    }

    if input_path == "None":
        input_path = os.getcwd()

    for i, run_identifier in enumerate(run_identifiers):
        predictions = pd.read_csv(input_path + "/" + run_identifier + "_predictions.csv")

        if ds == True:
            disease_associated_miRs = get_HMDD_disease_associated_db(HMDD_disease_names[i])["mir"]
            
            predictions['miRNAs'] = predictions["miRNAs"].apply(lambda x: re.sub('(-5p|-3p|.3p|.5p)$', '', x))

            predictions = predictions[predictions["miRNAs"].isin(disease_associated_miRs)]

        y_test = predictions["HMDD_Class"]
        y_pred = predictions["Average_Prob"]
        y_pred_class = predictions.iloc[:, 3:4]

        causal_count = predictions[predictions["HMDD_Class"] == True].shape[0]

        population_size = len(predictions.index)

        CM = confusion_matrix(y_test, y_pred_class)
        p_value = hypergeom.sf(CM[1][1]-1, population_size, causal_count, CM[1][1]+CM[0][1])

        fpr, tpr, thresholds = roc_curve(y_test, y_pred)

        rocauc = roc_auc_score(y_test, y_pred)

        auc, auc_cov = delong_roc_variance(
            y_test,
            y_pred)

        alpha = .95
        auc_std = np.sqrt(auc_cov)
        lower_upper_q = np.abs(np.array([0, 1]) - (1 - alpha) / 2)

        ci = stats.norm.ppf(
            lower_upper_q,
            loc=auc,
            scale=auc_std)

        if logging:
            print(run_identifier)

            print('AUC:', auc)
            print('AUC COV:', auc_cov)
            print('95% AUC CI:', ci)

            print('CM:', CM)
            print('P-Value:', p_value)

        rgb = folder_to_rgb_dict[run_identifier]

        plt.plot(fpr, tpr, c = rgb, label = folder_to_disease_name_dict[run_identifier] + ' AUC = %0.3f' % rocauc)

    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')

    if save_path is not None:
        plt.savefig(save_path + "/" + filename+'.png', bbox_inches='tight')

    plt.show()
 
if __name__ == "__main__":
    # plot_all_aucs("None", ["gastric", "breast", "hepato", "alzheimers"], filename = "avg_pred_aucs", save_path = os.getcwd())

    # plot_all_aucs("None", ["gastric", "breast", "hepato", "alzheimers"], 
    #             HMDD_disease_names = ["Gastric Neoplasms", "Breast Neoplasms", "Carcinoma, Hepatocellular", "Alzheimer Disease"],
    #             ds = True, filename = "ds_aucs", save_path = os.getcwd())

    # create_aggregate_cancer_model(path_to_miRNA_datas = "results/adaboost_class_final")
    # test_aggregate_cancer_model(path_to_model = os.getcwd() + "/cancer_aggregate_model.pickle", path_to_miRNA_datas = "results/adaboost_class_final")
    
    compare_alz_auc_to_agg_auc(path_to_model = os.getcwd() + "/cancer_aggregate_model.pickle",  path_to_miRNA_datas = "results/adaboost_class_final", path_to_alz_preds="results/adaboost_class_final")

    # run_identifiers = ["gastric", "breast", "hepato", "alzheimers"]
    # for run_identifier in run_identifiers:
    #     results_dict = pd.read_pickle(run_identifier + "_results_dict.pickle")
    #     print_results_dict(results_dict)
  
    