import pandas as pd
import numpy as np
from tqdm import tqdm
import re
from datetime import datetime
from sklearn.model_selection import KFold
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
import networkx as nx
import pickle 
import argparse
import os

from consensus_network_inference import infer_consensus_based_network, process_networks, create_source_to_target_network
from influence_inference import create_miRNA_graph
from miRNA_databases import get_HMDD_causal_db, get_miRNA_conservation, get_HMDD_disease_associated_db
from prediction import load_train_test_split, predict_disease_causality, plot_auc
  
def create_miRNA_data(HMDD_disease_name, miRNA_graph, disspec_miRNA_graph, key_to_miRNA_name_dict):

    #simulate for disease-associated miRNAs in disease-associated miRNAs
    HMDD_causal_db = get_HMDD_causal_db()

    print("creating mirna information dataframe...")

    conservation_score = get_miRNA_conservation(list(key_to_miRNA_name_dict.values()))
    
    miRNAs = {
        'miRNA':[],
        'Disease_Influence':[],
        'Network_Influence':[],
        'Conservation':[],
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

        is_mir_causal = 'yes' in HMDD_causal_db.loc[(HMDD_causal_db['mir'] == no_p_mir) & (HMDD_causal_db['disease'] == HMDD_disease_name), :].causality.unique()
        miRNAs['Causality'].append(is_mir_causal)
        
    return miRNAs

def print_results_dict(results_dict):
    print(" AUC: " + str(np.round_(results_dict["auc"], 3)))
    print(" AUC STD: " + str(np.round_(results_dict["auc_std"],3)))
    print(" AUC MEDIAN: " + str(np.round_(results_dict["median_auc"],3)))
    print("\n")
    print(" CM: " + str(np.round_(results_dict["CM"],3)))
    print(" CM STD: " + str(np.round_(results_dict["CM STD"],3)))
    print("\n")
    print(" P_VALUE: " + str(results_dict["average_p_val"]))
    print(" P_VALUE STD: " + str(results_dict["p_val_std"]))
    print(" WORST P_VALUE (MAX): " + str(results_dict["max p_val"]))
    print(" BEST P_VALUE (MIN): " + str(results_dict["min p_val"]))
    print("\n")
    print(" Feature Importances: " + str(np.round_(results_dict["feat_imps"],3)))
    print(" Feature Importances STD: " + str(np.round_(results_dict["feat_imps_std"],3)))
    print("\n")
    print(results_dict["mir_false_pos_ratio"])

def pipeline(args, random_state = None, return_value = None):
    # folder, return_value = "hc", random_state = None, 
    if args.run_type == "miRNA_data" or args.run_type == "false_positives":
        print("creating consensus based network")

        if args.input_path != "None":
            networks, columns = process_networks(path = args.input_path)
        else:
            networks, columns = process_networks(path = os.getcwd())    

        consensus_based_network = infer_consensus_based_network(networks)
            
        consensus_based_network = pd.DataFrame(data=consensus_based_network, index = columns, columns = columns)
        consensus_based_network = pd.DataFrame(data = np.where(consensus_based_network < .7, 0, consensus_based_network), index = columns, columns = columns)
        
        miRNA_graph, disspec_miRNA_graph, key_to_miRNA_name_dict = create_miRNA_graph(args.HMDD_disease_name, network = consensus_based_network)

        if args.run_type == "false_positives":
            return key_to_miRNA_name_dict
        
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

    X_train, X_test, y_train, y_test = load_train_test_split(miRNA_data, random_state = random_state)

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
    elif return_value == "train_test_split":
        return X_train, X_test, y_train, y_test
    elif return_value=="auc":
        rocauc = predict_disease_causality(X_train, X_test, y_train, y_test, miRNA_data = miRNA_data, use_pretrained_model = args.use_pretrained_model, return_value = "auc")
        return rocauc
    elif return_value == "source_to_target_network":
        source_to_target_network = create_source_to_target_network(consensus_based_network)
        if args.output_path != "None":
            miRNA_data.to_csv(args.output_path + "/" + args.run_identifier + "_source_to_target_network.csv")
        else:
            miRNA_data.to_csv(os.getcwd() + "/" + args.run_identifier + "_source_to_target_network.csv")
    elif return_value == "all":
        rocauc, CM, p_value, feature_importances, false_positives, positives_in_test, predictions, optimal_cutoff = predict_disease_causality(X_train, X_test, y_train, y_test, miRNA_data = miRNA_data, use_pretrained_model = args.use_pretrained_model,return_value = "all")
        return rocauc, CM, p_value, feature_importances, false_positives, positives_in_test, predictions, key_to_miRNA_name_dict, optimal_cutoff

def main(args):
    if args.run_type == "all":
        results_dict = {}

        random_states = np.random.randint(1000, size = (101))

        rocaucs = []
        CMs = []
        mir_num_false_pos_dict = {}
        mir_num_test_dict = {}
        feature_importances_list = []
        p_values = []

        predictions_dict = {}

        for random_state in tqdm(random_states):
            rocauc, CM, p_value, feature_importances, false_positives, positives_in_test, predictions, key_to_miRNA_name_dict, optimal_cutoff = pipeline(args, 
                            return_value = "all", random_state = random_state)
         
            rocaucs.append(rocauc)
            CMs.append(CM)
            p_values.append(p_value)
            feature_importances_list.append(feature_importances)

            for mir in positives_in_test:
                if mir in mir_num_test_dict.keys():
                    mir_num_test_dict[mir] += 1
                else:
                    mir_num_test_dict[mir] = 1
            
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
            "Prediction with " + str(optimal_cutoff) + " Threshold": [],
            "HMDD_Class": []
        }
        for mir in predictions_dict.keys():
            predictions_csv["miRNAs"].append(key_to_miRNA_name_dict[mir])
            average_prob = predictions_dict[mir]["summed_preds"] / predictions_dict[mir]["times_in_test"]
            predictions_csv["Average_Prob"].append(average_prob)
            predictions_csv["HMDD_Class"].append(predictions_dict[mir]["class"])
            predictions_csv["Prediction with " + str(optimal_cutoff) + " Threshold"].append(True if average_prob > optimal_cutoff else False)

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

        median_auc = np.median(rocaucs)
        random_state_for_median_auc = random_states[rocaucs.index(median_auc)]

        mir_false_pos_ratio = {}
        for mir in mir_num_false_pos_dict.keys():
            mir_false_pos_ratio[mir] = mir_num_false_pos_dict[mir] / mir_num_test_dict[mir]

        results_dict = {
            'auc': average_auc,
            'auc_std': auc_std,
            'CM': average_CM,
            'CM STD':CM_std,
            'mir_false_pos_ratio': mir_false_pos_ratio,
            'median_auc': median_auc,
            'random_state_for_median_auc': random_state_for_median_auc,
            'average_p_val' :average_p_val,
            'p_val_std': p_val_std,
            'max p_val': max(p_values),
            'min p_val': min(p_values),
            'feat_imps': average_feature_importances,
            'feat_imps_std':feat_imps_std
        }

        print_results_dict(results_dict)

        results_dict_filename =  args.run_identifier + "_results_dict.pickle"
        filepath = args.output_path + "/" + results_dict_filename if args.output_path != "None" \
                    else os.getcwd() + "/" + results_dict_filename
        with open(filepath, 'wb') as handle:
            pickle.dump(results_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        predictions_csv = pd.DataFrame(predictions_csv)
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
            plot_auc(predictions, auc_label = args.HMDD_disease_name, filename = args.run_identifier + "_ds_auc", save_path = save_path)
        elif args.run_type == "graph_auc_all":
            plot_auc(predictions, auc_label = args.HMDD_disease_name, filename = args.run_identifier + "_auc", save_path = save_path)
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
    parser.add_argument('--HMDD_disease_name', type=str,
                        help='the name of the disease in the HMDD disease dataset. Unfortunately, \
                        you will have to search this up yourself')
    parser.add_argument('--run_identifier', type=str,
                        help='all files will be saved and read as run_identifier_filename. e.g. \
                            gastric_results_dict.pickle, gastric_miRNA_data.csv, etc. ' )
    parser.add_argument("--use_pretrained_model", default = False, action='store_true',
                        help = "use pretrained_model or not")      
    parser.add_argument("--run_type", choices=["all", "miRNA_data",
                        "source_to_target_network", "graph_auc_all", "graph_auc_ds"], help = "the all option saves p-value, auc, \
                            confusion matrix, median_auc, feature importances, and \
                            mir_false_pos_ratio into results_dict.pickle and the predictions into predictions.csv. input: miRNA_data\
                            miRNA_data saves the miRNA_data into a .csv file. input: inferred_networks \
                            source_to_target_network saves the network (for cytoscape visualization)\
                            in a .csv file. input: inferred_networks \
                            graph_auc_ds graphs the auc for disease-associated miRs in HMDD on a png. input: predictions.csv \
                            graph_auc_all graphs the auc for all miRs on a png. input: predictions.csv")
                
    args = parser.parse_args()

    main(args)

# python3 pathomir.py --output_path None \
#                     --input_path disease_data/gastric/inferred_networks \
#                     --HMDD_disease_name "Gastric Neoplasms" \
#                     --run_identifier "gastric" \
#                       --run_type miRNA_data 

# python3 pathomir.py --output_path None \
#                     --input_path None \
#                     --HMDD_disease_name "Carcinoma, Hepatocellular" \
#                     --run_identifier "hepato" \
#                     --run_type all 

# python3 pathomir.py --output_path None \
#                     --input_path None \
#                     --HMDD_disease_name "Alzheimer Disease" \
#                     --run_identifier "alzheimers" \
#                     --run_type all 

# python3 pathomir.py --output_path None \
#                     --input_path None \
#                     --HMDD_disease_name "Breast Neoplasms" \
#                     --run_identifier "breast" \
#                     --run_type all 

# python3 pathomir.py --output_path None \
#                     --input_path None \
#                     --HMDD_disease_name "Carcinoma, Hepatocellular" \
#                     --run_identifier "hepato" \
#                     --run_type graph_auc_ds


# python3 pathomir.py --output_path None \
#                     --input_path None \
#                     --HMDD_disease_name "Gastric Neoplasms" \
#                     --run_identifier "gastric" \
#                     --use_pretrained_model True \
#                     --run_type "all"

                    

# python3 pathomir.py --output_path None --input_path disease_data/gastric/inferred_networks --HMDD_disease_name "Gastric Neoplasms" --run_identifier "gastric" --run_type miRNA_data
# python3 pathomir.py --output_path None --input_path disease_data/breast_cancer/inferred_networks --HMDD_disease_name "Breast Neoplasms" --run_identifier "breast_cancer" --run_type miRNA_data

# python3 pathomir.py --output_path None --input_path None --HMDD_disease_name "Breast Neoplasms" --run_identifier "breast_cancer" --run_type all --use_pretrained_model True
