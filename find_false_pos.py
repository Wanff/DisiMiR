from io import StringIO
from Bio import Entrez, Medline
import pickle
import pandas as pd
import argparse
import os
from tqdm import tqdm
import re

from disimir import pipeline

#PUBMED
def search_medline(query, email,**kwargs):
    Entrez.email = email
    search = Entrez.esearch(db='pubmed', term=query, **kwargs)
    try:
        handle = Entrez.read(search)
        return handle
    except RuntimeError as e:
        return None
    except Exception as e:
        print(e)
        # raise IOError(str(e))
        return None
    finally:
        search.close()

def fetch_rec(rec_id, entrez_handle):
    fetch_handle = Entrez.efetch(db='pubmed', id=rec_id,
                                 rettype='Medline', retmode='text',
                                 webenv=entrez_handle['WebEnv'],
                                 query_key=entrez_handle['QueryKey'])
    rec = fetch_handle.read()
    return rec

def tryreturn(record, col):
    try: 
        if (col == 'FAU') or (col == 'AD'):
            return '; '.join(record[col])
        elif col=='COIS':
            return ' '.join(record[col])
        else:
            return record[col]
    except:
        return ''

def get_pubmed_papers(query, email):
    initiate = search_medline(query, email, retstart=0, retmax=20, usehistory='y')
    # print(initiate['Count'])
    rec_handler = search_medline(
        query, 
        email, 
        retstart=0, 
        retmax=20, 
        usehistory='y',
        webenv=initiate['WebEnv'],
        query_key=initiate['QueryKey'],
    )

    if rec_handler == None:
        return "no_papers_about_disease"
        
    pubmed_papers = []
    for rec_id in rec_handler['IdList']:
        rec = fetch_rec(rec_id, rec_handler)
        rec_file = StringIO(rec)
        medline_rec = Medline.read(rec_file)

        paper = {}
        if 'AB' in medline_rec:
            paper['abstract'] = medline_rec['AB']
            paper['title'] = medline_rec['TI']
            paper['PMID'] = medline_rec['PMID']

            pubmed_papers.append(paper)
    
    return pubmed_papers

def find_false_positives(args):
    false_pos_data = []
    keywords = ['function', 'suppress', 'intervention', 'gain-of-function', 'causal', 'induc', 'promot', 'inhibit', 'regulat', 'target', 'modulat', 'mediat', 'activat', 'sponge', 'increas', 'enhanc', 'proliferat', 'facilitat', 'progress']

    if args.path_to_predictions != "None":
        predictions = pd.read_csv(args.path_to_predictions + "/" + args.run_identifier + "_predictions.csv")
    else:
        predictions = pd.read_csv(os.getcwd() + "/" + args.run_identifier + "_predictions.csv")

    false_positives = predictions[(predictions["HMDD_Class"] == False) & (predictions["Class_Prediction"] == True)]['miRNAs'].tolist()
    print("NUM OF FALSE_POSITIVES: " + str(len(false_positives)))
    print(false_positives)

    for false_pos_mir in tqdm(false_positives):
        mir_query =  re.sub('(-5p|-3p|.3p|.5p)$', '', false_pos_mir)
        mir_query = re.sub('hsa-', '', mir_query)

        query = mir_query + " " + args.disease_pubmed_query
        pubmed_papers = get_pubmed_papers(query, args.email)

        causal_papers = []
        non_causal_papers = []
        if pubmed_papers == "no_papers_about_disease":
            pubmed_papers = []

            false_pos_data.append((mir_query, 
                            True if len(causal_papers) > 0 else False,
                            len(pubmed_papers), 
                            causal_papers,
                            non_causal_papers,
                            ))
            continue

        for paper in pubmed_papers:
            if any(substring in paper['title'] for substring in keywords):
                causal_papers.append((paper['PMID'], paper['title']))
            else:
                non_causal_papers.append((paper['PMID'], paper['title']))
        
        false_pos_data.append((mir_query, 
                         True if len(causal_papers) > 0 else False,
                        len(pubmed_papers), 
                        causal_papers,
                        non_causal_papers,
                        ))

    for mir in false_pos_data:
        print(mir[0])
        for paper in mir[3]:
            print(str(paper[0]) + ": " + paper[1])
        print("\n")
    
    false_pos_df = pd.DataFrame(false_pos_data, columns = ['miR', "Pathogenicity", 'num_papers_returned', 'Causal_Papers', "Non_Causal_Papers"])

    print_false_pos_summary_statistics(false_pos_df)

    false_pos_filename =  args.run_identifier + "_false_pos.csv"
    filepath = args.output_path + "/" + false_pos_filename if args.output_path != "None" \
                    else os.getcwd() + "/" + false_pos_filename
    false_pos_df.to_csv(filepath)
    # with open(filepath, 'wb') as handle:
    #     pickle.dump(false_pos, handle, protocol=pickle.HIGHEST_PROTOCOL)

def print_false_pos_summary_statistics(false_pos_df, print_papers = False):
    false_pos_stats = {
        'causal':0, 
        'has_papers_not_causal':0, 
        'no_papers_about_disease':0
        }
    
    for index, mir in false_pos_df.iterrows():
        if mir['num_papers_returned'] == 0:
            false_pos_stats['no_papers_about_disease'] += 1
        elif mir['Pathogenicity'] is False:
            false_pos_stats['has_papers_not_causal'] += 1
        elif mir['Pathogenicity'] is True:
            false_pos_stats['causal'] += 1
        
        if print_papers:
            print(mir['miR'])
            print(mir['Pathogenicity'])
            for paper in eval(mir['Causal_Papers']):
                print(paper)
            for paper in eval(mir['Non_Causal_Papers']):
                print(paper)          
            print("\n")

    print("NUM OF FALSE POS CAUSAL MIRS: " + str(false_pos_stats['causal']))
    print("NUM OF NON-CAUSAL MIRS WITH PAPERS: " + str(false_pos_stats['has_papers_not_causal']))
    print("NUM OF MIRS WITH NO PAPERS ABOUT DISEASE: " + str(false_pos_stats['no_papers_about_disease']))

    print("Percent Causal miRNAs: " + str(false_pos_stats['causal'] / (false_pos_stats['causal'] + false_pos_stats['has_papers_not_causal']))) 

def print_mirs_with_papers(args, false_pos):
    mirs_with_papers = []
    false_pos_stats = {
        'causal':0, 
        'has_papers_not_causal':0, 
        'no_papers_about_disease':0
        }

    for mir in false_pos:
        if len(mir[2]) > 0:
            false_pos_stats['causal'] +=1
            mirs_with_papers.append(mir[0])
        elif mir[1] == 0:
            false_pos_stats['no_papers_about_disease'] +=1
        elif mir[1] > 0:
            false_pos_stats['has_papers_not_causal'] +=1
            mirs_with_papers.append(mir[0])

    print("NUM OF FALSE POS CAUSAL MIRS: " + str(false_pos_stats['causal']))
    print("NUM OF NON-CAUSAL MIRS WITH PAPERS: " + str(false_pos_stats['has_papers_not_causal']))
    print("NUM OF MIRS WITH NO PAPERS ABOUT DISEASE: " + str(false_pos_stats['no_papers_about_disease']))
    print("NUM OF MIRS WITH NO PAPERS AT ALL: " + str(false_pos_stats['no_papers_at_all']))
     
    for mir in mirs_with_papers:
        query = mir + " " + args.disease_pubmed_query
        pubmed_papers = get_pubmed_papers(query, args.email)

        print(mir)
        for paper in pubmed_papers:
            print(str(paper['PMID']) + " " + paper["title"])
        
        print("\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--output_path', type=str, default = None,
                        help='path to save false_pos.pickle. If None, current directory is used')
    parser.add_argument('--path_to_predictions', type=str,
                        help='path to predictions (which has the predicted false positive miRNAs). If None,\
                            current directory will be used.')
    parser.add_argument("--run_type", choices=["false_positives"], help = "")
    parser.add_argument('--run_identifier', type=str,
                        help='all files will be saved and read as run_identifier_filename. e.g. \
                            gastric_results_dict.pickle, gastric_miRNA_data.csv, etc. ' )
    parser.add_argument('--HMDD_disease_name', nargs="+",
                        help='the name of the disease in the HMDD disease dataset. Unfortunately, \
                        you will have to search this up yourself')
    parser.add_argument('--email', type=str,default = None,
                        help='email to query pubmed api with')
    parser.add_argument('--disease_pubmed_query', type=str,default = None,
                        help='disease name to search up on pubmed with the miRNAs')

    args = parser.parse_args()

    find_false_positives(args)

"""
python3 find_false_pos.py --output_path results/adaboost_seq_sim_and_targets/gastric  \
                    --path_to_predictions results/adaboost_seq_sim_and_targets/gastric \
                    --HMDD_disease_name "Gastric Neoplasms" \
                    --run_identifier "gastric" \
                    --email kevn.wanf@gmail.com \
                    --disease_pubmed_query "Gastric Cancer" \
                    --run_type false_positives 
    
python3 find_false_pos.py --output_path results/adaboost_seq_sim_and_targets/hepato \
                    --path_to_predictions results/adaboost_seq_sim_and_targets/hepato \
                    --HMDD_disease_name "Carcinoma, Hepatocellular" \
                    --run_identifier "hepato" \
                    --email kevn.wanf@gmail.com \
                    --disease_pubmed_query "Hepatocellular Cancer" \
                    --run_type false_positives 

python3 find_false_pos.py --output_path results/adaboost_seq_sim_and_targets/breast_cancer \
                    --path_to_predictions results/adaboost_seq_sim_and_targets/breast_cancer \
                    --HMDD_disease_name "Breast Neoplasms" \
                    --run_identifier "breast_cancer" \
                    --email kevn.wanf@gmail.com \
                    --disease_pubmed_query "Breast Cancer" \
                    --run_type false_positives 

python3 find_false_pos.py --output_path results/adaboost_seq_sim_and_targets/alzheimers \
                    --path_to_predictions results/adaboost_seq_sim_and_targets/alzheimers \
                    --HMDD_disease_name "Alzheimer Disease" \
                    --run_identifier "alzheimers" \
                    --email kevn.wanf@gmail.com \
                    --disease_pubmed_query "Alzheimers" \
                    --run_type false_positives 
"""