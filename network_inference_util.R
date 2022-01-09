#install libraries
if (!requireNamespace("BiocManager", quietly = TRUE))
 install.packages("BiocManager")
 
BiocManager::install("GEOquery")
BiocManager::install("minet")
BiocManager::install("GENIE3")

#load gene expression data from GEOquery
#example datasets
#GSE155362  breast cancer
#GSE108724  hepatocellular carcinoma 
#GSE164174  gastric cancer 
#GSE150693  alzheimers
#GSE29190   tuberculosis
#GSE179105	heart failure
library(GEOquery)
gds <- getGEO("GSEXXXXX")
disease <- exprs(gds[[1]])

#preprocess dataset 

#MANUAL
#remove the samples that aren't in disease-context by indexing out the columns
#For Tuberculosis
disease <- disease[,1:12]
#For Heart Failure
disease <- disease[,1:36]

#if the disease-context samples aren't one after another, you can index them out like:
#For Hepato
disease <- disease[,c('GSM2912552', 'GSM2912554',	'GSM2912556', 'GSM2912558', 'GSM2912560','GSM2912562','GSM2912564')]

#Convert miRNA names
#For Breast Cancer
gene_exprs_id <- getGEO("GPL28943")
gene_expres_id = (gene_exprs_id@dataTable@table)
row.names(breast_cancer) <- gene_expres_id[, "Probe Name"]

#For Alzheimer's
library(miRBaseConverter)
data(alzheimers)
Accessions = row.names(alzheimers)
accession_to_mirs = miRNA_AccessionToName(Accessions)
row.names(alzheimers) <- accession_to_mirs[, "TargetName"]
alzheimers <- alzheimers[complete.cases(alzheimers), ] 

remove(Accessions)
remove(accession_to_mirs)

remove_miRs_no_var <- function(disease_expres) {
  #preprocess
  t_disease_expres <- t(disease_expres)
  
  #remove columns with 0 variance
  colvar0<-apply(t_disease_expres,2,function(x) var(x,na.rm=T)==0)
  t_disease_expres <- t_disease_expres[,!(colnames(t_disease_expres) %in% c(colnames(t_disease_expres)[which(colvar0)]))]
  
  disease_expres <<- t(t_disease_expres)
}

save_inferred_networks <- function(disease_expres, filepath) {
  library(GENIE3)
  weightMat <<- GENIE3(disease_expres)
  
  t_disease_expres <- t(disease_expres)
  
  library(minet)
  clr_network <<- minet(t_disease_expres, method="clr")
  mrnet_network <<- minet(t_disease_expres, method="mrnet")
  aracne_network <<- minet(t_disease_expres, method="aracne")
  mrnetb_network <<- minet(t_disease_expres, method="mrnetb")
  
  #export matrix to csv
  write.csv(weightMat, file = paste(filepath, "genie3.csv", sep=''), row.names = TRUE)
  write.csv(clr_network, file = paste(filepath, "clr.csv", sep=''), row.names = TRUE)
  write.csv(mrnet_network, file = paste(filepath, "mrnet.csv", sep=''), row.names = TRUE)
  write.csv(aracne_network, file =paste(filepath, "aracne.csv", sep=''), row.names = TRUE)
  write.csv(mrnetb_network, file = paste(filepath, "mrnetb.csv", sep=''), row.names = TRUE)
}

#filepath <- "path/to/inferred_networks"

#Miscellaneous functions
#finding duplicates
n_occur <- data.frame(table(tube$miRNA_ID))
n_occur[n_occur$Freq > 1,]

#change column values into row names
rownames(t_disease_expres) <- t_disease_expres[,1]
t_disease_expres[,1] <- NULL

#convert dataframe to matrix (numeric)
as.matrix(sapply(t_disease_expres, as.numeric))

remove(mrnet_network)
remove(mrnetb_network)
remove(aracne_network)
remove(clr_network)
remove(weightMat)