import pandas as pd
import numpy as np
from tqdm import tqdm
import re
from datetime import datetime
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from scipy.stats import hypergeom
from scipy import stats
from confidence_intervals import delong_roc_variance
import networkx as nx
import pickle 
import argparse
import os

from consensus_network_inference import infer_consensus_based_network, process_networks, create_source_to_target_network
from influence_inference import create_miRNA_graph
from miRNA_databases import get_HMDD_causal_db, get_miRNA_conservation, get_HMDD_disease_associated_db, get_miRNA_targets
from prediction import load_train_test_split, stratified_random_split, predict_disease_causality, plot_auc
  
def create_miRNA_data(HMDD_disease_name, miRNA_graph, disspec_miRNA_graph, key_to_miRNA_name_dict):

    #simulate for disease-associated miRNAs in disease-associated miRNAs
    HMDD_causal_db = get_HMDD_causal_db()

    print("creating mirna information dataframe...")
    
    print("getting conservation information...")
    conservation_score = get_miRNA_conservation(list(key_to_miRNA_name_dict.values()))

    print("getting target information...")
    miR_to_targets = get_miRNA_targets(list(key_to_miRNA_name_dict.values()))

    miRNAs = {
        'miRNA':[],
        'Disease_Influence':[],
        'Network_Influence':[],
        'Conservation':[],
        'Num_Targets': [],
        'Causality':[],
    }

    whole_shortest_path_dict = dict(nx.all_pairs_shortest_path_length(nx.DiGraph(miRNA_graph.graph)))
    dis_shortest_path_dict = dict(nx.all_pairs_shortest_path_length(nx.DiGraph(disspec_miRNA_graph.graph)))

    for mir in tqdm(list(key_to_miRNA_name_dict.values())):
        network_coverage = 0
        disease_coverage = 0

        key = list(key_to_miRNA_name_dict.keys())[list(key_to_miRNA_name_dict.values()).index(mir)]

        network_coverage += miRNA_graph.getInfluence(key, whole_shortest_path_dict)
        disease_coverage += disspec_miRNA_graph.getInfluence(key, dis_shortest_path_dict)

        miRNAs['miRNA'].append(mir)
        miRNAs['Network_Influence'].append(network_coverage)
        miRNAs['Disease_Influence'].append(disease_coverage)

        no_p_mir = re.sub('(-5p|-3p|.3p|.5p)$', '', mir)
        miRNAs['Conservation'].append(conservation_score[no_p_mir])

        miRNAs['Num_Targets'].append(miR_to_targets[mir])

        is_mir_causal = 'yes' in HMDD_causal_db[HMDD_causal_db['disease'].isin(HMDD_disease_name)].loc[(HMDD_causal_db['mir'] == no_p_mir),:].causality.unique()
        # is_mir_causal = 'yes' in HMDD_causal_db.loc[(HMDD_causal_db['mir'] == no_p_mir) & (HMDD_causal_db['disease'] in HMDD_disease_name), :].causality.unique()
        miRNAs['Causality'].append(is_mir_causal)
        
    return miRNAs

def print_results_dict(results_dict):
    for metric in results_dict.keys():
        if "p_val" in metric:
            print(metric + ": " + str(results_dict[metric]))
        elif "mir_false_pos_ratio" in metric:
            print("Percentage of Times a miRNA was a false positive out of 100 random splits")
            print(results_dict[metric])
        else:
            print(metric + ": " + str(np.round_(results_dict[metric], 3)))

def pipeline(args, random_state = None, split = None, return_value = None):
    # folder, return_value = "hc", random_state = None, 
    if args.run_type == "miRNA_data":
        print("creating consensus based network")

        if args.input_path != "None":
            networks, columns = process_networks(path = args.input_path)
        else:
            networks, columns = process_networks(path = os.getcwd())    

        consensus_based_network = infer_consensus_based_network(networks)
            
        consensus_based_network = pd.DataFrame(data=consensus_based_network, index = columns, columns = columns)
        consensus_based_network = pd.DataFrame(data = np.where(consensus_based_network < .7, 0, consensus_based_network), index = columns, columns = columns)
        
        miRNA_graph, disspec_miRNA_graph, key_to_miRNA_name_dict = create_miRNA_graph(args.HMDD_disease_name, network = consensus_based_network)
        miRNA_data = create_miRNA_data(args.HMDD_disease_name, miRNA_graph, disspec_miRNA_graph, key_to_miRNA_name_dict)
        miRNA_data = pd.DataFrame(miRNA_data)
    else:
        if args.input_path == "None":
            miRNA_data = pd.read_csv(os.getcwd()  + "/" + args.run_identifier + "_miRNA_data.csv")
        else:
            miRNA_data = pd.read_csv(args.input_path  + "/" + args.run_identifier + "_miRNA_data.csv")
        
        key_to_miRNA_name_dict = {}
        for index, mir in enumerate(miRNA_data["miRNA"]):
            key_to_miRNA_name_dict[index] = mir
        
        if args.run_type == "false_positives":
            return key_to_miRNA_name_dict

    # if logging == True:
    #     print("Number of Null Relationships:" + str((consensus_based_network.values == 0).sum()))
    #     print("Number of Valid Relationships: " + str((consensus_based_network.values > 0).sum()))
        
    #     print(miRNA_data)
    
    if return_value == "miRNA_data":
        if args.output_path != "None":
            miRNA_data.to_csv(args.output_path + "/" + args.run_identifier + "_miRNA_data.csv")
        else:
            miRNA_data.to_csv(os.getcwd() + "/" + args.run_identifier + "_miRNA_data.csv")
        return miRNA_data
    if return_value == "key_to_miRNA_name_dict":
        return key_to_miRNA_name_dict
    # elif return_value == "train_test_split":
    #     X_train, X_test, y_train, y_test = load_train_test_split(miRNA_data, random_state = random_state)

    #     return X_train, X_test, y_train, y_test
    # elif return_value=="auc":
    #     X_train, X_test, y_train, y_test = load_train_test_split(miRNA_data, random_state = random_state)

    #     rocauc = predict_disease_causality(X_train, X_test, y_train, y_test, miRNA_data = miRNA_data, use_pretrained_model = args.use_pretrained_model, return_value = "auc")
    #     return rocauc
    elif return_value == "source_to_target_network":
        source_to_target_network = create_source_to_target_network(consensus_based_network)
        if args.output_path != "None":
            miRNA_data.to_csv(args.output_path + "/" + args.run_identifier + "_source_to_target_network.csv")
        else:
            miRNA_data.to_csv(os.getcwd() + "/" + args.run_identifier + "_source_to_target_network.csv")
    elif return_value == "metrics_and_predictions":
        # train_indices, test_indices, X, y = stratified_random_split(miRNA_data, random_state = random_state)

        # X_ = np.array(X)
        # y_ = np.array(y)
        # X_train, X_test, y_train, y_test = X_[train_indices[split]], X_[test_indices[split]], y_[train_indices[split]], y_[test_indices[split]]
        # y_test = pd.DataFrame(y_test)
        X_train, X_test, y_train, y_test = load_train_test_split(miRNA_data, random_state = random_state)

        rocauc, CM, p_value, feature_importances, false_positives, predictions, optimal_cutoff = predict_disease_causality(X_train, X_test, y_train, y_test, miRNA_data = miRNA_data, use_pretrained_model = args.use_pretrained_model,return_value = "metrics_and_predictions")
        return rocauc, CM, p_value, feature_importances, false_positives, predictions, key_to_miRNA_name_dict, optimal_cutoff

def main(args):
    if args.run_type == "metrics_and_predictions":
        results_dict = {}

        random_states = np.random.randint(1000, size = (100))

        rocaucs = []
        CMs = []
        mir_num_false_pos_dict = {}
        mir_num_test_dict = {}
        feature_importances_list = []
        p_values = []

        predictions_dict = {}

        for random_state in tqdm(random_states):
            # for split in range(3):
                rocauc, CM, p_value, feature_importances, false_positives, predictions, key_to_miRNA_name_dict, optimal_cutoff = pipeline(args, 
                                return_value = "metrics_and_predictions", random_state = random_state)
            
                rocaucs.append(rocauc)
                CMs.append(CM)
                p_values.append(p_value)
                feature_importances_list.append(feature_importances)
                
                for mir in false_positives:
                    if mir in mir_num_false_pos_dict.keys():
                        mir_num_false_pos_dict[mir] += 1
                    else:
                        mir_num_false_pos_dict[mir] = 1

                for pred in predictions:
                    if pred[1] not in predictions_dict.keys():
                        predictions_dict[pred[1]] = {
                            "summed_preds": pred[0],
                            "times_in_test": 1,
                            "class": pred[2]
                        }
                    else:
                        predictions_dict[pred[1]]["summed_preds"] += pred[0]
                        predictions_dict[pred[1]]["times_in_test"] += 1            
        
        predictions_csv = {
            "miRNAs": [],
            "Average_Prob": [],
            "Class_Prediction": [],
            "HMDD_Class": []
        }
        for mir in predictions_dict.keys():
            predictions_csv["miRNAs"].append(key_to_miRNA_name_dict[mir])
            average_prob = predictions_dict[mir]["summed_preds"] / predictions_dict[mir]["times_in_test"]
            predictions_csv["Average_Prob"].append(average_prob)
            predictions_csv["HMDD_Class"].append(predictions_dict[mir]["class"])
            predictions_csv["Class_Prediction"].append(True if average_prob > optimal_cutoff else False)

        predictions_csv = pd.DataFrame(predictions_csv)
        
        #AVERAGE PREDICTION METRICS
        causal_count = predictions_csv[predictions_csv["HMDD_Class"] == True].shape[0]
        population_size = len(predictions_csv.index)

        avg_pred_CM = confusion_matrix(predictions_csv.loc[:,"HMDD_Class"], predictions_csv.loc[:,"Class_Prediction"])
        avg_pred_p_val =  hypergeom.sf(CM[1][1]-1, population_size, causal_count, CM[1][1]+CM[0][1])

        avg_pred_auc, avg_pred_auc_cov = delong_roc_variance(
            predictions_csv.loc[:,"HMDD_Class"], 
            predictions_csv.loc[:,"Average_Prob"]
        )

        alpha = .95
        auc_std = np.sqrt(avg_pred_auc_cov)
        lower_upper_q = np.abs(np.array([0, 1]) - (1 - alpha) / 2)
        avg_pred_CI = stats.norm.ppf(
            lower_upper_q,
            loc=avg_pred_auc,
            scale=auc_std)

        #AVERAGE SPLIT METRICS
        CMs = np.array(CMs)
        feature_importances_list = np.array(feature_importances_list)

        average_auc = sum(rocaucs)/len(rocaucs)
        average_CM = CMs.sum(axis = 0)/len(CMs)
        average_feature_importances = feature_importances_list.sum(axis = 0)/len(feature_importances_list)
        average_p_val = sum(p_values)/len(p_values)

        auc_std = np.std(rocaucs)
        CM_std = np.std(CMs, axis = 0)
        feat_imps_std = np.std(feature_importances_list, axis = 0)
        p_val_std = np.std(p_values)

        # median_auc = np.median(rocaucs)
        # random_state_for_median_auc = random_states[rocaucs.index(median_auc)]

        mir_false_pos_ratio = {}
        for mir in mir_num_false_pos_dict.keys():
            mir_false_pos_ratio[mir] = mir_num_false_pos_dict[mir] / predictions_dict[mir]["times_in_test"]

        results_dict = {
            'avg_split_auc': average_auc,
            'avg_split_auc_std': auc_std,
            'avg_split_CM': average_CM,
            'avg_split_CM_STD': CM_std,
            # 'median_auc': median_auc,
            # 'random_state_for_median_auc': random_state_for_median_auc,
            'avg_split_p_val' :average_p_val,
            'avg_split_p_val_std': p_val_std,
            'avg_split_max_p_val': max(p_values),
            'avg_split_min_p_val': min(p_values),
            'avg_split_feat_imps': average_feature_importances,
            'avg_split_feat_imps_std':feat_imps_std,
            'avg_pred_auc': avg_pred_auc,
            'avg_pred_CM': avg_pred_CM,
            'avg_pred_p_val': avg_pred_p_val,
            'avg_pred_auc_cov': avg_pred_auc_cov,
            'avg_pred_CI': avg_pred_CI,
            'threshold': optimal_cutoff, 
            'mir_false_pos_ratio': mir_false_pos_ratio,
            'num_mirs': population_size,
            'num_causal_mirs': causal_count
        }

        print_results_dict(results_dict)

        results_dict_filename =  args.run_identifier + "_results_dict.pickle"
        filepath = args.output_path + "/" + results_dict_filename if args.output_path != "None" \
                    else os.getcwd() + "/" + results_dict_filename
        with open(filepath, 'wb') as handle:
            pickle.dump(results_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        print(predictions_csv)

        if args.output_path != "None":
            predictions_csv.to_csv(args.output_path + "/" + args.run_identifier + "_predictions.csv")
        else:
            predictions_csv.to_csv(os.getcwd() + "/" + args.run_identifier + "_predictions.csv")
    elif "graph_auc" in args.run_type:
        if args.input_path == "None":
            predictions = pd.read_csv(os.getcwd()  + "/" + args.run_identifier + "_predictions.csv")
        else:
            predictions = pd.read_csv(args.input_path  + "/" + args.run_identifier + "_predictions.csv")

        save_path = os.getcwd()  + "/" if args.output_path == "None" else args.output_path

        if args.run_type == "graph_auc_ds":
            disease_associated_miRs = get_HMDD_disease_associated_db(args.HMDD_disease_name)["mir"]
            
            predictions['miRNAs'] = predictions["miRNAs"].apply(lambda x: re.sub('(-5p|-3p|.3p|.5p)$', '', x))

            predictions = predictions[predictions["miRNAs"].isin(disease_associated_miRs)]
            plot_auc(predictions, auc_label = args.run_identifier, filename = args.run_identifier + "_ds_auc", save_path = save_path)
        elif args.run_type == "graph_auc":
            plot_auc(predictions, auc_label = args.run_identifier, filename = args.run_identifier + "_auc", save_path = save_path)
    elif args.run_type == "miRNA_data":
        pipeline(args, return_value = "miRNA_data")
    elif args.run_type == "source_to_target_network":
        pipeline(args, return_value = "source_to_target_network")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--output_path', type=str, default = "None",
                        help='path to save all outputs. if None, current directory will be used')
    parser.add_argument('--input_path', type=str, default="None",
                        help='path to genie3, mrnet, mrnetb, aracne and clr networks OR \
                            path to miRNA_data (the influence and conservation features). If None,\
                            current directory will be used.')
    parser.add_argument('--HMDD_disease_name',  nargs="+", 
                        help='the name of the disease in the HMDD disease dataset. Unfortunately, \
                        you will have to search this up yourself')
    parser.add_argument('--run_identifier', type=str,
                        help='all files will be saved and read as run_identifier_filename. e.g. \
                            gastric_results_dict.pickle, gastric_miRNA_data.csv, etc. ' )
    parser.add_argument("--use_pretrained_model", default = False, action='store_true',
                        help = "use pretrained_model or not")      
    parser.add_argument("--run_type", choices=["metrics_and_predictions", "miRNA_data",
                        "source_to_target_network", "graph_auc", "graph_auc_ds"], help = "the metrics_and_predictions option saves p-value, auc, \
                            confusion matrix, feature importances, and \
                            mir_false_pos_ratio into results_dict.pickle and the predictions into predictions.csv. input: miRNA_data\
                            miRNA_data saves the miRNA_data into a .csv file. input: inferred_networks \
                            source_to_target_network saves the network (for cytoscape visualization)\
                            in a .csv file. input: inferred_networks \
                            graph_auc_ds graphs the auc for disease-associated miRs in HMDD on a png. input: predictions.csv \
                            graph_auc graphs the auc for all miRs on a png. input: predictions.csv")
                
    args = parser.parse_args()

    main(args)
"""
python3 disimir.py --output_path results/adaboost_seq_sim_and_targets \
                    --input_path disease_data/breast_cancer/inferred_networks \
                    --HMDD_disease_name "Breast Neoplasms" \
                    --run_identifier "breast_cancer" \
                    --run_type miRNA_data 

python3 disimir.py --output_path results/adaboost_seq_sim_and_targets/breast_cancer \
                    --input_path results/adaboost_seq_sim_and_targets/breast_cancer \
                    --HMDD_disease_name "Breast Neoplasms" \
                    --run_identifier "breast_cancer" \
                    --run_type metrics_and_predictions 


python3 disimir.py --output_path results/adaboost_seq_sim_and_targets/hepato \
                    --input_path results/adaboost_seq_sim_and_targets/hepato \
                    --HMDD_disease_name "Carcinoma, Hepatocellular" \
                    --run_identifier "hepato" \
                    --run_type metrics_and_predictions 

python3 disimir.py --output_path results/adaboost_seq_sim_and_targets/gastric \
                    --input_path results/adaboost_seq_sim_and_targets/gastric \
                    --HMDD_disease_name "Gastric Neoplasms" \
                    --run_identifier "gastric" \
                    --run_type metrics_and_predictions 

python3 disimir.py --output_path results/adaboost_seq_sim_and_targets/alzheimers \
                    --input_path results/adaboost_seq_sim_and_targets/alzheimers \
                    --HMDD_disease_name "Alzheimer Disease" \
                    --run_identifier "alzheimers" \
                    --run_type metrics_and_predictions                 
"""
