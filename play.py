#%%
import pandas as pd
import matplotlib.pyplot as plt
import re 
seq_sim_data = pd.read_csv("results/adaboost_seq_sim_and_targets/breast_cancer/breast_cancer_miRNA_data.csv")
miRNA_data = pd.read_csv("results/adaboost_class_final/breast_miRNA_data.csv")

lev = seq_sim_data["Conservation"] - miRNA_data["Conservation"]

# plt.scatter(miRNA_data["Conservation"], lev)

# plt.show()
lev_miRNA = miRNA_data.loc[lev == miRNA_data["Conservation"]]

mirs = lev_miRNA[lev_miRNA["Conservation"] > 0]["miRNA"].tolist()

print([re.sub('(-5p|-3p|.3p|.5p)$', '', mir) for mir in mirs])
# %%
var = pd.Series([], )
print(var.empty)
# %%
seq_sim_data = pd.read_csv("results/adaboost_seq_sim_and_targets/breast_cancer_miRNA_data.csv")
miRNA_data = pd.read_csv("results/adaboost_class_final/breast_miRNA_data.csv")

lev = seq_sim_data["Conservation"] - miRNA_data["Conservation"]

plt.scatter(miRNA_data["Conservation"], lev)


# %%
import Levenshtein as lev

print(lev.distance("", "abc"))
# %%
dis_to_causal_mirs = {}

for dis in ["breast_cancer", "gastric", "hepato","alzheimers"]:
    preds = pd.read_csv("results/adaboost_seq_sim_and_targets/" + dis + "/" + dis + "_predictions.csv")
    print(dis)
    causal_mirs  = list(preds[preds["Class_Prediction"] == True]["miRNAs"])
    print(len(causal_mirs))
    dis_to_causal_mirs[dis] = set(causal_mirs)
    
# %%
for dis1 in ["breast_cancer", "gastric", "hepato","alzheimers"]:
    for dis2 in ["breast_cancer", "gastric", "hepato","alzheimers"]:
        print(dis1 + " / " + dis2)
        percent_intersect = len(dis_to_causal_mirs[dis1].intersection(dis_to_causal_mirs[dis2])) / len(dis_to_causal_mirs[dis1])
        print(percent_intersect)
    print("\n")
# %%

all_causal_mirs = []
for dis in dis_to_causal_mirs:
    all_causal_mirs += list(dis_to_causal_mirs[dis])

mir_to_freq = {}
for mir in all_causal_mirs:
    if mir not in mir_to_freq:
        mir_to_freq[mir] = 0
    mir_to_freq[mir] += 1

print(sorted(mir_to_freq.items(), key = lambda x: x[1]))

for i in range(5):
    print(len([value for value in mir_to_freq.values() if value == i]))

#%%
mir_to_dis = {}
for dis in dis_to_causal_mirs:
    for mir in list(dis_to_causal_mirs[dis]):
        if mir not in mir_to_dis:
            mir_to_dis[mir] = []
        mir_to_dis[mir].append(dis)

dis_to_unique_mirs = {
    'breast_cancer': [],
    'gastric': [],
    'hepato': [],
    'alzheimers': [],
}
for mir in mir_to_dis:
    if len(mir_to_dis[mir]) ==1:
        dis_to_unique_mirs[mir_to_dis[mir][0]].append(mir)

for dis in dis_to_unique_mirs:
    print(dis)
    print(len(dis_to_unique_mirs[dis]))
# %%
keywords = ['by', 'function', 'suppress', 'intervention', 'gain-of-function', 'causal', 'induc', 'promot', 'inhibit', 'regulat', 'target', 'modulat', 'mediat', 'activat', 'sponge', 'increas', 'enhanc', 'proliferat', 'facilitat', 'progress']
if any(substring in "Identification of Key Circulating Exosomal microRNAs in Gastric Cancer." for substring in keywords):
    print("hi")
# %%
