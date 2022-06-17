import pandas as pd
import re
from tqdm import tqdm
import Levenshtein as lev

def get_HMDD_causal_db(path = 'miRNA_databases/HMDD3_causal_info.csv'):
    """
    Returns the entire HMDD miRNA causality database
    """
 
    causal = pd.read_csv(path, header = None, encoding= 'unicode_escape')

    causal = causal[[1, 2, 5]].iloc[1:]
    causal.columns = ['mir', 'disease', 'causality']
    
    return causal

def get_HMDD_disease_associated_db(disease, path = 'miRNA_databases/HMDD3_association_info.txt'):
    """
    Returns the set of miRNAs associated with a given disease in the HMDD miRNA-association database
    """

    if isinstance(disease, str):
        disease = [disease]

    associated = pd.read_csv(path, sep='\t', encoding= 'unicode_escape')

    associated_miRNA = associated[["mir", "disease"]]
    disease_associated_ = []
    for dis in disease:
        disease_associated_.append(associated_miRNA[associated_miRNA['disease'] == dis])
    
    return pd.concat(disease_associated_).drop_duplicates()

def get_miRNA_conservation(miRNA_names, path = "miRNA_databases/"):
    mirna_fam_map = pd.read_csv(path + "mirna_2_prefam.txt", sep="\t", header=None)
    mirnas = pd.read_csv(path + "mirna.txt", sep="\t", header=None)

    mirna_fam_map.columns = ['mir_num_id', 'fam_num_id']
    mirnas.columns = ['mir_num_id', 'mir_acc-id', 'mir', 'alt_mir', 'loop', 'seq', 'paper', 'irrelevant', 'irrelevant']
    
    fam_dict = {}
    for mir in tqdm(miRNA_names):
        mir = re.sub('(-5p|-3p|.3p|.5p)$', '', mir)

        #Gets the miRbase ID of the miRNA
        mir_num_id = mirnas[mirnas['mir'] == mir].mir_num_id 
        if len(mir_num_id.index) == 0: #check if alt_mir contains match
            mir_num_id = mirnas[mirnas['alt_mir'] == mir].mir_num_id

        if len(mir_num_id.index) == 0: #now check if str.contains works
            regex = r'^' + re.escape(mir) + '([a-z]|-\d|[a-z]-\d)$' 
            mir_num_id_idx = mirnas['mir'].loc[lambda x: x.str.match(regex, case=False, na=False)].index
            mir_num_id_idx.append(mirnas['alt_mir'].loc[lambda x: x.str.match(regex, case=False, na=False)].index)
            if len(mir_num_id.index) == 0:
                mir_num_id = 0.0
            else:
                mir_num_id = mirnas.iloc[mir_num_id_idx[0]].mir_num_id
        mir_num_id = int(mir_num_id)

        #Gets the family ID of the miRNA
        fam_num_id = mirna_fam_map[mirna_fam_map['mir_num_id']==mir_num_id].fam_num_id
        if len(fam_num_id.index) == 0:
            fam_num_id = 0
        fam_num_id = int(fam_num_id)

        num_fam_membs = len(mirna_fam_map[mirna_fam_map['fam_num_id']==fam_num_id].index) #rowcount for how many members of that family

        seq_sim_score = 0
        mir_seq = mirnas[mirnas['mir'] == mir]['seq'].tolist()

        for fam_mem in mirna_fam_map[mirna_fam_map['fam_num_id']==fam_num_id].mir_num_id:
            fam_mem_seq = mirnas[mirnas['mir_num_id'] == fam_mem]['seq'].tolist()
            
            if fam_mem_seq and mir_seq:
                lev_dist = lev.distance(mir_seq[0], fam_mem_seq[0])

                seq_sim_score += (len(mir_seq[0]) - lev_dist) / len(mir_seq[0])

        fam_dict[mir] = num_fam_membs + seq_sim_score

    return fam_dict

def get_miRNA_targets(miRNA_names, path = "miRNA_databases/TargetScan_predicted_hsa_targets.txt"):
    targets = pd.read_csv(path)
    targets = targets.iloc[:, 1:]

    targets["miRNA"] = targets["miRNA"].str.lower()

    target_counts = targets["miRNA"].value_counts().to_dict()

    #removes all miRs that end in .1 or .2 and adds their targets to mir w/o .1 or .2
    for mir in list(target_counts.keys()):
        if mir.endswith(".1") or mir.endswith(".2"):
            edited_mir = mir[:-2]

            if edited_mir in target_counts.keys(): #if edited_mir already exists, just add the .2 one to it
                target_counts[edited_mir] = target_counts[edited_mir] + target_counts[mir]
            else:
                target_counts[edited_mir] = target_counts[mir]

            target_counts.pop(mir ,None)

    no_matches = []
    mir_to_num_targets = {}

    for mir in miRNA_names:
        no_p_mir = re.sub('(-5p|-3p|.3p|.5p)$', '', mir)

        if mir in target_counts.keys(): #if normal mir with p 
            mir_to_num_targets[mir] = target_counts[mir]
        elif no_p_mir in target_counts.keys(): #mir without p
            mir_to_num_targets[mir] = target_counts[no_p_mir]
        else:
            mir_to_num_targets[mir] = 0
            no_matches.append(mir)
    
    print(str(len(no_matches)) + " NO TARGETSCAN MATCHES")
    # print(no_matches)

    return mir_to_num_targets    
