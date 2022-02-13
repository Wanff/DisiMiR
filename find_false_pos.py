from io import StringIO
from Bio import Entrez, Medline
import pickle
import pandas as pd
import argparse
import os
from tqdm import tqdm

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
    elif int(initiate['Count']) > 100:
        return "no_papers_at_all"
        
    pubmed_papers = []
    for rec_id in rec_handler['IdList']:
        rec = fetch_rec(rec_id, rec_handler)
        rec_file = StringIO(rec)
        medline_rec = Medline.read(rec_file)

        paper = {}
        if 'AB' in medline_rec:
            paper['abstract'] = medline_rec['AB']
            paper['title'] = medline_rec['TI']
            paper['authors'] = medline_rec['FAU']
            paper['PMID'] = medline_rec['PMID']

            pubmed_papers.append(paper)
    
    return pubmed_papers

def find_false_positives(args):
    false_pos = []
    keywords = ['function', 'intervention', 'gain-of-function', 'causal', 'induce']

    if args.path_to_results_dict != "None":
        results_dict = pd.read_pickle(args.path_to_results_dict + "/" + args.run_identifier + "_results_dict.pickle")
    else:
        results_dict = pd.read_pickle(os.getcwd() + "/" + args.run_identifier + "_results_dict.pickle")
    
    key_to_miRNA_name_dict = pipeline(args)

    false_positives = [k for k,v in results_dict['mir_false_pos_ratio'].items() if v >= 0.5]
    
    print("NUM OF FALSE_POSITIVES: " + str(len(false_positives)))

    for false_pos_key in tqdm(false_positives):
        query = key_to_miRNA_name_dict[false_pos_key] + " " + args.disease_pubmed_query
        pubmed_papers = get_pubmed_papers(query, args.email)

        causal_papers = []
        if pubmed_papers == "no_papers_about_disease":
            pubmed_papers = [1]*1001

            false_pos.append((key_to_miRNA_name_dict[false_pos_key], len(pubmed_papers), causal_papers))
            continue
        elif pubmed_papers == "no_papers_at_all":
            pubmed_papers = []

            false_pos.append((key_to_miRNA_name_dict[false_pos_key], len(pubmed_papers), causal_papers))
            continue

        for paper in pubmed_papers:
            if any(substring in paper['abstract'] for substring in keywords):
                causal_papers.append(paper['title'])
            
        false_pos.append((key_to_miRNA_name_dict[false_pos_key], len(pubmed_papers), causal_papers))

    print_mirs_with_papers(args, false_pos)

    false_pos_filename =  args.run_identifier + "_false_pos.pickle"
    filepath = args.output_path + "/" + false_pos_filename if args.output_path != "None" \
                    else os.getcwd() + "/" + false_pos_filename
    with open(filepath, 'wb') as handle:
        pickle.dump(false_pos, handle, protocol=pickle.HIGHEST_PROTOCOL)

def print_mirs_with_papers(args, false_pos):
    mirs_with_papers = []
    false_pos_stats = {
        'causal':0, 
        'has_papers_not_causal':0, 
        'no_papers_at_all':0, 
        'no_papers_about_disease':0
        }

    for mir in false_pos:
        if len(mir[2]) > 0:
            false_pos_stats['causal'] +=1
            mirs_with_papers.append(mir[0])
        elif mir[1] == 1001:
            false_pos_stats['no_papers_about_disease'] +=1
        elif mir[1] > 0:
            false_pos_stats['has_papers_not_causal'] +=1
            mirs_with_papers.append(mir[0])
        elif mir[1] == 0:
            false_pos_stats['no_papers_at_all'] +=1

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
    parser.add_argument('--input_path', type=str, default="None",
                        help='path to miRNA_data (the influence and conservation features) in order to \
                            get key_to_miRNA_dict (maps the key in the graph object to miRNA name) If None,\
                            current directory will be used.')
    parser.add_argument('--path_to_results_dict', type=str,
                        help='path to results_dict (which has the mir_false_pos_ratios). If None,\
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