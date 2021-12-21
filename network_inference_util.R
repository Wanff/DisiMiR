#get gene expression dataset

#install the GEOquery library if you haven't already:
# if (!requireNamespace("BiocManager", quietly = TRUE))
# install.packages("BiocManager")
# BiocManager::install("GEOquery")
library(GEOquery)

#get the gds object with the gene expression data. This example case is the breast cancer dataset from the paper
gds <- getGEO("GSE155362") 

#parse through the gds object to get the expression data
breast_cancer = exprs(gds[[1]])

#You will probably have to do some preprocessing. See below for some methods that might be useful

#https://bioconductor.org/packages/release/bioc/vignettes/GENIE3/inst/doc/GENIE3.html
#install GENIE3 if you haven't already:
# if (!requireNamespace("BiocManager", quietly = TRUE))
#     install.packages("BiocManager")

# BiocManager::install("GENIE3")

#GENIE3 takes input as gene by sample
library(GENIE3)
weightMat <- GENIE3(breast_cancer)

#https://www.bioconductor.org/packages/release/bioc/manuals/minet/man/minet.pdf
#install minet if you haven't already:
# if (!requireNamespace("BiocManager", quietly = TRUE))
#     install.packages("BiocManager")

# BiocManager::install("minet")

#minet takes input as sample by gene
t_gene_expres <- t(breast_cancer)
library(minet)
clr <- minet(t_gene_expres, method="clr")
mrnet <- minet(t_gene_expres, method="mrnet")
aracne <- minet(t_gene_expres, method="aracne")
mrnetb <- minet(t_gene_expres, method="mrnetb")

filepath <- "path/to/inferred_networks"
#export matrix to csv
write.csv(weightMat, file = paste(filepath, "genie3.csv", sep=''), row.names = TRUE)
write.csv(clr, file = paste(filepath, "clr.csv", sep=''), row.names = TRUE)
write.csv(mrnet, file = paste(filepath, "mrnet.csv", sep=''), row.names = TRUE)
write.csv(aracne, file =paste(filepath, "aracne.csv", sep=''), row.names = TRUE)
write.csv(mrnetb, file = paste(filepath, "mrnetb.csv", sep=''), row.names = TRUE)

remove(weightMat)
remove(clr)
remove(mrnet)
remove(aracne)
remove(mrnetb)
remove(t_gene_expres)

#Helpful Methods. 
#Change the ID names in the GDS object to miRNA names 
gene_exprs_id <- getGEO("GPLXXXXX") 
gene_expres_id = (gene_exprs_id@dataTable@table)
row.names(disease) <- gene_expres_id[, "Probe Name"]

#convert MIMATO IDs to miRNA names (alzheimers)
library(miRBaseConverter)
data(disease)
Accessions = row.names(disease)
accession_to_mirs = miRNA_AccessionToName(Accessions)
row.names(disease) <- accession_to_mirs[, "TargetName"]
disease <- disease[complete.cases(disease), ] 

remove(Accessions)
remove(accession_to_mirs)

#get rid of rows where there is nothing in one column
t_gene_expres <- t_gene_expres[!(is.na(t_gene_expres$mir) | t_gene_expres$mir==""), ]

#get columns that have 0 variance and remove them. You need to do this for aracne and clr to work
colvar0<-apply(t_gene_expres,2,function(x) var(x,na.rm=T)==0)
which(colvar0) #sees which ones have 0 variance (evaluate to True)

t_gene_expres <- t_gene_expres[,!(colnames(t_gene_expres) %in% c(["miRNAs that have 0 variance"]))]

