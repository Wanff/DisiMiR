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
from disimir import print_results_dict
from prediction import make_ensemble, predict_disease_causality, load_train_test_split, AdaBoostEnsembleModel
from confidence_intervals import delong_roc_variance, delong_significance_between_two_aucs
import scipy.stats
from scipy import stats
from find_false_pos import print_false_pos_summary_statistics
#Miscellaneous functions used in the paper

def create_aggregate_cancer_dataset(path_to_miRNA_datas = os.getcwd()):
    run_identifiers = ["gastric", "breast_cancer", "hepato"]

    cancer_data = []
    for run_identifier in run_identifiers:
        miRNA_data = pd.read_csv(path_to_miRNA_datas  + "/" + run_identifier + "/" + run_identifier + "_miRNA_data.csv")

        cancer_data.append(miRNA_data)

    cancer_data = pd.concat(cancer_data)

    return cancer_data

def create_aggregate_cancer_model(path_to_miRNA_datas = os.getcwd()):
    cancer_data = create_aggregate_cancer_dataset(path_to_miRNA_datas)

    X = cancer_data[['Disease_Influence', 'Network_Influence', 'Conservation', "Num_Targets"]]
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

    X = cancer_data[['Disease_Influence', 'Network_Influence', 'Conservation', "Num_Targets"]]
    y = cancer_data[['Causality']]

    y_pred = cancer_aggregate_model.predict_proba(X)[:,1]
    agg_rocauc = roc_auc_score(y, y_pred)

    alzheimers_miRNA_data = pd.read_csv(path_to_miRNA_datas + "/alzheimers/" + "alzheimers" + "_miRNA_data.csv").sort_values(by = 'Unnamed: 0')
    X_alz = alzheimers_miRNA_data[['Disease_Influence', 'Network_Influence', 'Conservation', "Num_Targets"]]
    y_alz = alzheimers_miRNA_data[['Causality']]

    alz_y_pred = cancer_aggregate_model.predict_proba(X_alz)[:,1]
    alz_rocauc = roc_auc_score(y_alz, alz_y_pred)

    if logging == True:
        print("FEATURE IMPORTANCES: " + str(cancer_aggregate_model.feature_importances_))

        print("AUC ON AGGREGATE DATASET: " + str(agg_rocauc))
        
        print("AUC ON ALZHEIMER'S DATASET: " + str(alz_rocauc))
    
    return alz_y_pred, y_alz

def compare_alz_auc_to_agg_auc(path_to_model, path_to_miRNA_datas = os.getcwd(), path_to_alz_preds = os.getcwd()):
    agg_alz_y_pred, y_alz_miRNA_data = test_aggregate_cancer_model(path_to_model, path_to_miRNA_datas, logging = False)
    # print(np.array(y_alz))
    # print(agg_alz_y_pred)

    alz_data = pd.read_csv(path_to_alz_preds + "/alzheimers/" + "alzheimers" + "_predictions.csv").sort_values(by = 'Unnamed: 0')
    alz_y_pred = alz_data[['Average_Prob']]
    y_alz_from_preds = alz_data[['HMDD_Class']]

    print("AGGEGATE AUC ON ALZHEIMER'S DATASET: " + str(roc_auc_score(y_alz_miRNA_data, agg_alz_y_pred)))
    print("OG AUC ON ALZHEIMER'S DATASET: " + str(roc_auc_score(y_alz_from_preds, alz_y_pred)))

    z, p = delong_significance_between_two_aucs(agg_alz_y_pred, np.array(alz_y_pred), np.array(y_alz_miRNA_data))
    print("P-VALUE: " + str(p))
    print("Z-SCORE: " + str(z))

    return z, p

def compare_aucs(predictions1_path, predictions2_path):
    #reorder the predictions so that they are the same
    predictions1 = pd.read_csv(predictions1_path).sort_values(by = 'Unnamed: 0')
    predictions2 = pd.read_csv(predictions2_path).sort_values(by = 'Unnamed: 0')

    print(predictions1)
    print(predictions2)

    z, p = delong_significance_between_two_aucs(np.array(predictions1['Average_Prob']), np.array(predictions2['Average_Prob']), np.array(predictions1['HMDD_Class']))
    print("P-VALUE: " + str(p))
    print("Z-SCORE: " + str(z))

    return z, p

import pandas as pd
import re
from itertools import combinations
import Levenshtein as lev
from tqdm import tqdm

def preprocess_targetscan_db(path = "miRNA_databases/Summary_Counts.all_predictions.txt"):
    db = pd.read_csv(path, sep = "\t", header = None)
    db = db.iloc[1:,:]
    db = db[[1, 2, 13]]

    db.columns = ["Gene Symbol", "miRNA Family", "miRNA"]

    db = db[db.miRNA.str.contains("hsa", na = False)]
    print(db)

    db.to_csv("miRNA_databases/TargetScan_predicted_hsa_targets.txt")

    #228049 MTIs in conserved targetscan
    #321-ish miRs in all target scan
 
    #7765065 MTIs in all target scan
    #2479-ish miRs in all target scan

def calculate_seq_sim(path = "miRNA_databases/"):
    mirnas = pd.read_csv(path + "mirna.txt", sep="\t", header=None)
    mirnas.columns = ['mir_num_id', 'mir_acc-id', 'mir', 'alt_mir', 'loop', 'seq', 'paper', 'irrelevant', 'irrelevant']

    mirnas = mirnas[mirnas.mir.str.contains("hsa", na = False)]
    print(mirnas)

    mir_pairs = combinations(list(mirnas['mir']), 2)

    mir_seq_sim_adj = pd.DataFrame(index = list(mirnas['mir']), columns = list(mirnas['mir']))
    #adj matric where [i,j] equals len(i) - lev(i, j) / len(i)

    print(mirnas[mirnas['mir'] == "hsa-let-7a-1"]['seq'])

    for mir_pair in tqdm(mir_pairs):
        seq1 = str(mirnas[mirnas['mir'] == mir_pair[0]]['seq'])
        seq2 = str(mirnas[mirnas['mir'] == mir_pair[1]]['seq'])

        lev_dist = lev.distance(seq1, seq2)
        mir_seq_sim_adj.loc[mir_pair[0], mir_pair[1]] = (len(mir_pair[0]) - lev_dist) / len(mir_pair[0])
        mir_seq_sim_adj.loc[mir_pair[1], mir_pair[0]] = (len(mir_pair[1]) - lev_dist) / len(mir_pair[1])

    return mir_seq_sim_adj

from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, confusion_matrix, roc_curve

def cm_for_thresholds(y_test, y_pred, false_pos_weight = 1, false_neg_weight = 1):
    __, __, thresholds = roc_curve(y_test, y_pred)
    print(len(thresholds))

    threshold_to_errors_dict = {}
    threshold_to_cm = {}
    for threshold in thresholds:
        y_pred_class = []
        for pred in y_pred:
            if pred > threshold:
                y_pred_class.append(1)
            else:
                y_pred_class.append(0)
        
        CM = confusion_matrix(y_test, y_pred_class)
        
        threshold_to_cm[threshold] = CM
        threshold_to_errors_dict[threshold] = CM[0][1] * false_pos_weight + CM[1][0] * false_neg_weight 
        # sens =  CM[1][1] / (CM[1][0] + CM[1][1]) 
        # spec =  CM[1][1] / (CM[0][1] + CM[1][1]) 

        # threshold_to_errors_dict[threshold] = spec * false_pos_weight + sens * false_neg_weight 

    sorted_thresholds = list(sorted(threshold_to_errors_dict.items(), key = lambda x: x[1], reverse = False))
    for thresh in sorted_thresholds[:5]:
        print(threshold_to_errors_dict[thresh[0]])
        print(threshold_to_cm[thresh[0]])

#graph aucs for multiple diseases on same plot
def plot_all_aucs(input_path, run_identifiers, HMDD_disease_names = None, ds = False, filename = "all_aucs", save_path = "disease_data", logging = True):
    """
    Plots all the AUC curves on the same figure

    If you don't want to save the AUC image, set save_path to None
    """

    folder_to_rgb_dict = {
    'breast_cancer': "#2b769c",
    'hepato':  "#61d9e6",
    'gastric': "#bda6df",
    'alzheimers': "#c41919"
    }

    folder_to_disease_name_dict = {
    'breast_cancer': 'Breast Cancer', 
    'alzheimers': 'Alzheimer\'s Disease',
    'hepato': 'Hepatocellular Cancer',
    'gastric': 'Gastric Cancer', 
    }

    if input_path == "None":
        input_path = os.getcwd()

    for i, run_identifier in enumerate(run_identifiers):
        predictions = pd.read_csv(input_path + "/" + run_identifier + "/" + run_identifier + "_predictions.csv")

        if ds == True:
            disease_associated_miRs = get_HMDD_disease_associated_db(HMDD_disease_names[i])["mir"]
            predictions['miRNAs'] = predictions["miRNAs"].apply(lambda x: re.sub('(-5p|-3p|.3p|.5p)$', '', x))

            predictions = predictions[predictions["miRNAs"].isin(disease_associated_miRs)]

        y_test = predictions["HMDD_Class"]
        y_pred = predictions["Average_Prob"]
        y_pred_class = predictions["Class_Prediction"]

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
 
def get_num_irrelevant_mirs(input_path, run_identifiers, HMDD_disease_names = None):
    for i, run_identifier in enumerate(run_identifiers):
        predictions = pd.read_csv(input_path + "/" + run_identifier + "/" + run_identifier + "_predictions.csv")

        disease_associated_miRs = get_HMDD_disease_associated_db(HMDD_disease_names[i])["mir"]
        predictions['miRNAs'] = predictions["miRNAs"].apply(lambda x: re.sub('(-5p|-3p|.3p|.5p)$', '', x))

        predictions = predictions[(~predictions["miRNAs"].isin(disease_associated_miRs)) & (predictions["HMDD_Class"] != False)]

        print(len(predictions.index))


if __name__ == "__main__":
    # plot_all_aucs("results/adaboost_seq_sim_and_targets", ["gastric", "breast_cancer", "hepato", "alzheimers"], filename = "avg_pred_aucs", save_path = "results/adaboost_seq_sim_and_targets")

    # plot_all_aucs("results/adaboost_seq_sim_and_targets", ["gastric", "breast_cancer", "hepato", "alzheimers"], 
    #             HMDD_disease_names = ["Gastric Neoplasms", "Breast Neoplasms", "Carcinoma, Hepatocellular", "Alzheimer Disease"],
    #             ds = True, filename = "ds_aucs", save_path = "results/adaboost_seq_sim_and_targets")
    
    # get_num_irrelevant_mirs("results/adaboost_seq_sim_and_targets", ["gastric", "breast_cancer", "hepato", "alzheimers"], 
    #             HMDD_disease_names = ["Gastric Neoplasms", "Breast Neoplasms", "Carcinoma, Hepatocellular", "Alzheimer Disease"])

    # create_aggregate_cancer_model(path_to_miRNA_datas = "results/adaboost_seq_sim_and_targets")
    # test_aggregate_cancer_model(path_to_model = os.getcwd() + "/cancer_aggregate_model.pickle", path_to_miRNA_datas = "results/adaboost_seq_sim_and_targets")
    
    compare_alz_auc_to_agg_auc(path_to_model = os.getcwd() + "/cancer_aggregate_model.pickle",  path_to_miRNA_datas = "results/adaboost_seq_sim_and_targets", path_to_alz_preds="results/adaboost_seq_sim_and_targets")
    # compare_alz_auc_to_agg_auc(path_to_model = "results/adaboost_class_final/cancer_aggregate_model.pickle",  path_to_miRNA_datas = "results/adaboost_class_final", path_to_alz_preds="results/adaboost_class_final")

    # run_identifiers = ["alzheimers", 'breast_cancer','hepato','gastric']
    # for run_identifier in run_identifiers:
    #     false_pos_df = pd.read_csv("results/adaboost_seq_sim_and_targets/" + run_identifier + "/" + run_identifier +"_false_pos.csv")
    #     print(run_identifier)
    #     print_false_pos_summary_statistics(false_pos_df, print_papers = False)
    #     print("\n")
    
    # for dis in ["breast_cancer", "gastric", "hepato","alzheimers"]:
    #     filepath1 = "results/adaboost_seq_sim_and_targets" + "/" + dis + "/" + dis + "_predictions.csv"
    #     filepath2 = "results/adaboost_class_final" + "/" + dis + "_predictions.csv"

    #     print(dis)
    #     compare_aucs(filepath1,filepath2)
    #     print("\n")


    # for dis in ["breast_cancer", "gastric", "hepato","alzheimers"]:
    #     preds = pd.read_csv("results/adaboost_seq_sim_and_targets/" + dis + "/" + dis + "_predictions.csv")
    #     print(dis)
    #     cm_for_thresholds(preds["HMDD_Class"].tolist(), preds["Average_Prob"].tolist(), false_neg_weight = 2)

    