import pandas as pd
import re

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

    associated = pd.read_csv(path, sep='\t', encoding= 'unicode_escape')

    associated_miRNA = associated[["mir", "disease"]]
    disease_associated = associated_miRNA[associated_miRNA['disease'] == disease].drop_duplicates()
    
    return disease_associated

def get_miRNA_conservation(miRNA_names, path = "miRNA_databases/"):
    mirna_fam_map = pd.read_csv(path + "mirna_2_prefam.txt", sep="\t", header=None)
    mirnas = pd.read_csv(path + "mirna.txt", sep="\t", header=None)

    mirna_fam_map.columns = ['mir_num_id', 'fam_num_id']
    mirnas.columns = ['mir_num_id', 'mir_acc-id', 'mir', 'alt_mir', 'loop', 'seq', 'paper', 'irrelevant', 'irrelevant']

    fam_dict = {}
    for mir in miRNA_names:
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

        fam_num_id = mirna_fam_map[mirna_fam_map['mir_num_id']==mir_num_id].fam_num_id
        if len(fam_num_id.index) == 0:
            fam_num_id = 0
        fam_num_id = int(fam_num_id)

        num_fam_membs = len(mirna_fam_map[mirna_fam_map['fam_num_id']==fam_num_id].index) #rowcount for how many members of that family
        fam_dict[mir] = num_fam_membs

    return fam_dict