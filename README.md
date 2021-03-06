# DisiMiR 
DisiMiR is a computational tool that predicts pathogenic miRNAs in disease-context miRNA expression datasets. DisiMiR works by inferring characteristics of disease-causing miRNAs (miRNA conservation, network influence, disease-specific influence) and then training an AdaBoost Classifier on those features. You can read more about DisiMiR in our paper: "DisiMiR: Predicting Pathogenic miRNAs using Network Influence and miRNA Conservation"

You might want to use DisiMiR for a wide variety of purposes and in different ways, including:
1. **Generating hypotheses for further research.** In our paper, 41.2% of the false positives miRNAs predicted by DisiMiR have been validated as disease-causal in recent literature. 
2. **Differentiating disease-causal vs disease-associated miRNAs in the expression dataset.** Disease-association doesn't guarantee disease-causality, and in our paper, we show that DisiMiR can differentiate between disease-causal miRNAs and merely disease-associated miRNAs. 
3. **Using our aggregate cancer model (an AdaBoost classifier trained on three cancer datasets)** for purposes 1 or 2 if the expression dataset you're working with has too few positive classes to form a good training dataset. Using the aggregate cancer model on an Alzheimer's dataset increased its AUC by 0.058.

## Requirements
You will need Python and R to run DisiMiR. DisiMiR was tested with Python 3.9.6, R 4.0.3 and RStudio 1.2.5033.

First, clone DisiMiR 
```
git clone https://github.com/Wanff/DisiMiR
cd DisiMiR
pip install requirements.txt
```

## Work Flow
<p align="center">
  <img src="https://user-images.githubusercontent.com/50050060/148706718-474b4860-0a7b-443c-82d4-250e521328d8.png" height="540">
</p>

DisiMiR's workflow is a bit convoluted, mainly due to the network inference part. The consensus-based network inference algorithm infers the regulatory network from the miRNA expression data. This network is used to calculate whole-network and disease-specific coverage. These two characteristics are aggregated with miRNA conservation data from miRbase, and miRNA causality information from HMDD, to create a training dataset for an AdaBoost model, which predicts disease-causality.

## Network Inference
Since each expression dataset usually needs its own individual preprocessing, the network inference part is very manual. 

Open RStudio and copy network_inference_util.R into a .R file. Load your dataset into R (whether through GEO or manaully (ie drag-dropping a .csv file into Rstudio). 

Follow the workflow outlined in network_inference_util.R. If a function or a line is not prefaced by "For SomeDisease" then you should run that line for all datasets. The general flow is to load the dataset in, do any preprocessing, then infer and save the 5 different networks. 

## DisiMiR
Now that you have your networks, most of the manual work has been done. The only other thing you'll need to do is to go to the <a href= "https://www.cuilab.cn/hmdd" target = "_blank">HMDD Dataset</a> and find the name(s) of your disease in the dataset. Generally, the more names you find that fit your disease's description, the better, because you will have more positive classes in your training dataset. 

### DisiMiR Parameters
**output_path**: path to save all outputs of the run

**input_path**: path to inputs of the run. Either the inferred networks or miRNA_data

**HMDD_disease_name**: list of disease names in HMDD

**run_identifier**: this String will be prefixed in front of all outputs of the run i.e. disease_miRNA_data.csv

**run_type**: the type of run that DisiMiR will do:
* metrics_and_predictions
  * saves metrics (p-value, confusion matrix, AUC, feature importances) to results_dict.pickle and average predictions across 100 random splits to predictions.csv
* miRNA_data
  * creates miRNA_data.csv (all the features used to train the AdaBoost classifier)
* source_to_target_network
  * reformats the consensus-based network into source to target format (for Cytoscape visualization) and saves source_to_target_network.csv 
* graph_auc
  * graphs AUC for all miRNAs
* graph_auc_ds
  * graphs AUC for only the disease-associated miRNAs 

For all paths, if you input None, DisiMiR will use the current directory. 

### Generate the training data
```
python3 DisiMiR.py --output_path path/to/miRNA_data.csv \
                    --input_path path/to/inferred_networks \
                    --HMDD_disease_name "Disease Name 1" "Disease Name 2" \
                    --run_identifier "disease_name" \
                      --run_type miRNA_data 
```

### Get the predictions
```
python3 disimiR.py --output_path path/to/results_dict.pickle + predictions.csv \
                    --input_path path/to/miRNA_data.csv \
                    --HMDD_disease_name "Disease Name 1" "Disease Name 2" \
                    --run_identifier "disease_name" \
                    --run_type all 
```

If your training data fails to produce a good classifier you can use the aggregate cancer model like this:
```
python3 disimiR.py --output_path path/to/results_dict.pickle + predictions.csv \
                    --input_path path/to/miRNA_data.csv \
                    --HMDD_disease_name "Disease Name 1" "Disease Name 2" \
                    --run_identifier "disease_name" \
                    --use_pretrained_model \
                    --run_type metrics_and_predictions 
```

## Hypothesis Generation
Most parameters for hypothesis generation are the same for the normal DisiMir run. Hypothesis generation will need access to the miRNA_data for the key_to_miRNA_dict (which maps the key in the graph object to miRNA name) and also the results_dict, which stores the false_positives from the 100 random splits. 

The **email** parameter is the email that will be used to query the PubMED API and the **disease_pubmed_query** will be the query that will be searched on PubMED along with the false positive miRNAs. 
```
python3 find_false_pos.py --output_path path/to/false_pos.pickle \
                         --input_path path/to/miRNA_data.csv \
                         --path_to_results_dict path/to/results_dict.pickle \
                         --run_identifier "disease_name" \
                         --run_type "false_positives" \
                         --HMDD_disease_name "Disease Name 1" "Disease Name 2" \
                         --email "yourname@gmail.com" \
                         --disease_pubmed_query "Disease Name" 
 ```
 
 
