# PathomiR 
PathomiR is a computational tool that predicts pathogenic miRNAs in disease-context miRNA expression datasets. PathomiR works by inferring characteristics of disease-causing miRNAs (miRNA conservation, network influence, disease-specific influence) and then training an AdaBoost Classifier on those features. You can read more about PathomiR in our paper: "PathomiR: Predicting Pathogenic miRNAs using Network Influence and miRNA Conservation"

You might want to use PathomiR for a wide variety of purposes and in different ways, including:
1. **Generating hypotheses for further research.** In our paper, 41.2% of the false positives miRNAs predicted by PathomiR have been validated as disease-causal in recent literature. 
2. **Differentiating disease-causal vs disease-associated miRNAs in the expression dataset.** Disease-association doesn't guarantee disease-causality, and in our paper, we show that PathomiR can differentiate between disease-causal miRNAs and merely disease-associated miRNAs. 
3. **Using our aggregate cancer model (an AdaBoost classifier trained on three cancer datasets)** for purposes 1 or 2 if the expression dataset you're working with has too few positive classes to form a good training dataset. Using the aggregate cancer model on an Alzheimer's dataset increased its AUC by 0.058.

## Requirements
You will need Python and R to run PathomiR. PathomiR was tested with Python 3.9.6, R 4.0.3 and RStudio 1.2.5033.

First, clone PathomiR 
```
git clone https://github.com/Wanff/PathomiR
cd PathomiR
pip install requirements.txt
```

## Work Flow
<p align="center">
  <img src="https://user-images.githubusercontent.com/50050060/148706718-474b4860-0a7b-443c-82d4-250e521328d8.png" height="540">
</p>

PathomiR's workflow is a bit convoluted, mainly due to the network inference part.

### Network Inference
Since each expression dataset usually needs its own individual preprocessing, the network inference part is very manual. 

Open RStudio and copy network_inference_util.R into a .R file. Load your dataset into R (whether through GEO or manaully (ie drag-dropping a .csv file into Rstudio)). 

Follow the workflow outlined in network_inference_util.R. If a function or a line is not prefaced by "For SomeDisease" then you should run that line for all datasets. The general flow is to load the dataset in, do any preprocessing, then infer and save the 5 different networks. 

### PathomiR
Now that you have your networks, most of the manual work has been done. The only other thing you'll need to do is to go to the <a href= "https://www.cuilab.cn/hmdd" target = "_blank">HMDD Dataset</a> and find the name(s) of your disease in the dataset. Generally, the more names you find that fit your disease's description, the better, because you will have more positive classes in your training dataset. 

#### Generate the training data
```
python3 pathomir.py --output_path path/to/miRNA_data.csv \
                    --input_path path/to/inferred_networks \
                    --HMDD_disease_name "Disease Name 1" "Disease Name 2" \
                    --run_identifier "disease_name" \
                      --run_type miRNA_data 
```

#### Get the predictions
```
python3 pathomir.py --output_path path/to/results_dict.pickle + predictions.csv \
                    --input_path path/to/miRNA_data.csv \
                    --HMDD_disease_name "Disease Name 1" "Disease Name 2" \
                    --run_identifier "disease_name" \
                    --run_type all 
```

If your training data is bad, you can use the aggregate cancer model like this:
```
python3 pathomir.py --output_path path/to/results_dict.pickle + predictions.csv \
                    --input_path path/to/miRNA_data.csv \
                    --HMDD_disease_name "Disease Name 1" "Disease Name 2" \
                    --run_identifier "disease_name" \
                    --use_pretrained_model \
                    --run_type all 
```

### Hypothesis Generation

```
python3 find_false_pos.py --output_path path/to/false_pos.pickle \
                         --input_path path/to/inferred_networks \
                         --path_to_results_dict path/to/results_dict.pickle \
                         --run_identifier "disease_name" \
                         --run_type "false_positives" \
                         --HMDD_disease_name "Disease Name 1" "Disease Name 2" \
                         --email "yourname@gmail.com" \
                         --disease_pubmed_query "Disease Name" 
 ```
 
 
