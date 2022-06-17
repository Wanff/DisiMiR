#%%
import pandas as pd
import re 

from matplotlib_venn import venn2, venn2_circles, venn2_unweighted
from matplotlib_venn import venn3, venn3_circles
from matplotlib import pyplot as plt
from venn import venn

from tqdm import tqdm
#%%
#plotting seq_sim against old conservation

# seq_sim_data = pd.read_csv("results/adaboost_seq_sim_and_targets/breast_cancer/breast_cancer_miRNA_data.csv")
# miRNA_data = pd.read_csv("results/adaboost_class_final/breast_miRNA_data.csv")

# lev = seq_sim_data["Conservation"] - miRNA_data["Conservation"]

# # plt.scatter(miRNA_data["Conservation"], lev)

# # plt.show()
# lev_miRNA = miRNA_data.loc[lev == miRNA_data["Conservation"]]

# mirs = lev_miRNA[lev_miRNA["Conservation"] > 0]["miRNA"].tolist()

# print([re.sub('(-5p|-3p|.3p|.5p)$', '', mir) for mir in mirs])
# # %%
# var = pd.Series([], )
# print(var.empty)
# # %%
# seq_sim_data = pd.read_csv("results/adaboost_seq_sim_and_targets/breast_cancer_miRNA_data.csv")
# miRNA_data = pd.read_csv("results/adaboost_class_final/breast_miRNA_data.csv")

# lev = seq_sim_data["Conservation"] - miRNA_data["Conservation"]

# plt.scatter(miRNA_data["Conservation"], lev)

# %%
#getting miR frequencies
# dis_to_causal_mirs = {}

# for dis in ["breast_cancer", "gastric", "hepato","alzheimers"]:
#     preds = pd.read_csv("results/adaboost_seq_sim_and_targets/" + dis + "/" + dis + "_predictions.csv")
#     print(dis)
#     causal_mirs  = list(preds[preds["Class_Prediction"] == True]["miRNAs"])
#     print(len(causal_mirs))
#     dis_to_causal_mirs[dis] = set(causal_mirs)
    
# # %%
# for dis1 in ["breast_cancer", "gastric", "hepato","alzheimers"]:
#     for dis2 in ["breast_cancer", "gastric", "hepato","alzheimers"]:
#         print(dis1 + " / " + dis2)
#         percent_intersect = len(dis_to_causal_mirs[dis1].intersection(dis_to_causal_mirs[dis2])) / len(dis_to_causal_mirs[dis1])
#         print(percent_intersect)
#     print("\n")
# # %%

# all_causal_mirs = []
# for dis in dis_to_causal_mirs:
#     all_causal_mirs += list(dis_to_causal_mirs[dis])

# mir_to_freq = {}
# for mir in all_causal_mirs:
#     if mir not in mir_to_freq:
#         mir_to_freq[mir] = 0
#     mir_to_freq[mir] += 1

# print(sorted(mir_to_freq.items(), key = lambda x: x[1]))

# for i in range(5):
#     print(len([value for value in mir_to_freq.values() if value == i]))

# #%%
# mir_to_dis = {}
# for dis in dis_to_causal_mirs:
#     for mir in list(dis_to_causal_mirs[dis]):
#         if mir not in mir_to_dis:
#             mir_to_dis[mir] = []
#         mir_to_dis[mir].append(dis)

# dis_to_unique_mirs = {
#     'breast_cancer': [],
#     'gastric': [],
#     'hepato': [],
#     'alzheimers': [],
# }
# for mir in mir_to_dis:
#     if len(mir_to_dis[mir]) ==1:
#         dis_to_unique_mirs[mir_to_dis[mir][0]].append(mir)

# for dis in dis_to_unique_mirs:
#     print(dis)
#     print(len(dis_to_unique_mirs[dis]))
# %%
#*miR FREQUENCIES

#get only the miRs that are present in all 4/3 datasets
dis_to_mirs = {}
for dis in ["breast_cancer", "gastric", "hepato","alzheimers"]:
    preds = pd.read_csv("results/adaboost_seq_sim_and_targets/" + dis + "/" + dis + "_predictions.csv")
    
    miRNAs = list(preds["miRNAs"])  
    dis_to_mirs[dis] = set(miRNAs)
    
b_g = dis_to_mirs["breast_cancer"].intersection(dis_to_mirs["gastric"])
a_b_g = b_g.intersection(dis_to_mirs["alzheimers"])
h_a_b_g = a_b_g.intersection(dis_to_mirs["hepato"])

print(len(h_a_b_g))

#%%
cluster_type = "HMDD_Class"
# cluster_type = "Class_Prediction"

print(cluster_type)

mir_to_freq = {}

dis_identifiers = ["breast_cancer", "hepato", "gastric","alzheimers"]


dis_to_mirs = {
    'Breast Cancer': set(),
    'Hepatocellular Cancer': set(),
    'Gastric Cancer': set(),
    'Alzheimer\'s Disease': set(),
}

id_to_rgb_dict = {
    'breast_cancer': "#2b769c",
    'hepato':  "#61d9e6",
    'gastric': "#bda6df",
    'alzheimers': "#c41919"
    }

id_to_disease_name_dict = {
    'breast_cancer': 'Breast Cancer', 
    'alzheimers': 'Alzheimer\'s Disease',
    'hepato': 'Hepatocellular Cancer',
    'gastric': 'Gastric Cancer', 
    }
    
for mir in tqdm(h_a_b_g):
    if mir not in mir_to_freq:
            mir_to_freq[mir] = []
            
    for dis in dis_identifiers:
        preds = pd.read_csv("results/adaboost_seq_sim_and_targets/" + dis + "/" + dis + "_predictions.csv")
            
        if mir in list(preds[preds[cluster_type] == True]["miRNAs"]):
            mir_to_freq[mir].append(dis)
            
            dis_to_mirs[id_to_disease_name_dict[dis]].add(mir)
# %%
venn(dis_to_mirs, cmap = [id_to_rgb_dict[dis] for dis in dis_identifiers])
# %%
dis_to_mirs.pop("Alzheimer\'s Disease", None)
venn(dis_to_mirs, cmap = [id_to_rgb_dict[dis] for dis in dis_identifiers[:3]])

# %%
