import numpy as np
import pandas as pd
from tqdm import tqdm

def process_networks(path = None, network_names = ['aracne', 'genie3', 'clr', 'mrnet', 'mrnetb']):
    """  
    Input: path to csv files that contain the networks from each respective network inference algorithm
    Output: list of processed networks
    """

    assert path != None
    
    networks = []
    final = []
    for network in network_names:
        networks.append(pd.read_csv(path + '/' + network + '.csv', header = None, low_memory=False))

    miRNA_names = []
    for network, network_name in tqdm(zip(networks, network_names)):
        network.replace(np.nan, 0, inplace=True)

        #re-formats the adjacency matrix from the R script into Pandas dataframe
        #basically making the first row and first column the row and column names
        del network[0]
        network.columns = network.iloc[0]
        network = network.iloc[1:]
        network.index = network.columns

        if network_name == 'genie3':
            #reordering genie3's rows & columns bc it's not in order. making sure that networks can overlap
            network = network[miRNA_names]
            network = network.loc[miRNA_names,:]
        else:
            try:
                network = network.drop([0.0], axis = 1)
                network = network.drop([0.0], axis = 0)

                miRNA_names = network.columns
            except:
                miRNA_names = network.columns

           

        network = network.astype(float).round(10)
        final.append(network)

    return final, miRNA_names

def infer_consensus_based_network(networks):
    normalized_networks = []

    for network in tqdm(networks):
        #normalize the networks to 0 to 1 with min-max normalization
        normalized_network = (network-network.min())/(network.max()-network.min())
        normalized_networks.append(normalized_network)

    return np.average(normalized_networks, axis=0, weights = [1]*5).round(2)

def create_source_to_target_network(network):
    #create three column dataframe that contains all the relationships. 
    influence_network = pd.DataFrame(columns=['Source', 'Target', 'influence'])

    print("building source to target network...")
    rel_position = []
    #https://stackoverflow.com/questions/42386629/pandas-find-index-of-value-anywhere-in-dataframe

    for row in tqdm(range(network.shape[0])): 
        for col in range(network.shape[1]):
            if network.iat[row,col] > 0:
                rel_position.append((row, col, network.iat[row,col]))

    for i, position in tqdm(enumerate(rel_position)):
        #https://stackoverflow.com/questions/10715965/add-one-row-to-pandas-dataframe
        influence_network.loc[i] = [network.columns[position[0]]] + [network.columns[position[1]]] + [position[2]]
       
    return influence_network