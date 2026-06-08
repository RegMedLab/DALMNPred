
## Dependencies for Downloading data and for Annotation in Gene Expression Omnibus Database. Here
## microarray expression data has been obtained for datasets GSE13284, GSE71935 and GSE7874
# For training dataset GSE13284 has been used and for testing GSE7874 has been used ##



BiocManager::install(c("GEOquery", "AnnotationDbi"))

BiocManager::install("affy")
BiocManager::install(c("hgu133a.db", "hgu133acdf"))


require(GEOquery)

##Creating a directory in the system in accordance to the interest of the user ###   

setwd("E:/R/GSE/GSE13284") 

### Extract files in the designated folder #####

getGEOSuppFiles("GSE13284")

setwd("E:/R/GSE/GSE13284/GSE13284")

### untar/unzip the files  ###

untar("GSE13284_RAW.tar", exdir = "data")

cels = list.files("data", pattern = "CEL")
# sometiles, it is 'CEL',  it needs to be checked
sapply(paste("data", cels, sep = "/"), gunzip)

cels = list.files("data", pattern = "CEL")


library(affy)
library(hgu133a.db)
library(hgu133acdf)

# Set working directory for normalization

setwd("E:/R/GSE/GSE13284/GSE13284/data")
raw.data = ReadAffy(verbose = FALSE, filenames = cels)

r2 = exprs(raw.data)


### Here RMA has been used for data Normalization ####
rm4 = rma(raw.data)
r4 = exprs(rm4)


tt = cbind(row.names(r4), r4)
colnames(tt) = c("ProbID", sub(".cel", "", colnames(r4), ignore.case = TRUE))
rownames(tt) = NULL

## Extracting Gene Id corresponding to probe id

BiocManager::install("biomaRt")
require("biomaRt")


ensembl = useMart(biomart="ENSEMBL_MART_ENSEMBL",
                  dataset="hsapiens_gene_ensembl", 
                  host="uswest.ensembl.org",
                  ensemblRedirect = FALSE)


annotLookup <- getBM(
  mart=ensembl,
  attributes=c(
    "affy_hg_u133_plus_2",
    "gene_biotype",
    "refseq_ncrna",
    "external_gene_name"),
  filter = "affy_hg_u133_plus_2",
  values = rownames(r4), uniqueRows=TRUE)

library(dplyr)
library(readr)

### Extract only those genes with gene_biotype ="lncRNA" ###

r5 <- filter(annotLookup, gene_biotype == "lncRNA")
r6 <- r5 %>% mutate_all(na_if,"")
r7 <- filter(r6, refseq_ncrna != "NA")
r7 <- data.frame(r7)
tt <- data.frame(tt)

df2 <- inner_join(tt,r7, by = c("ProbID" = "affy_hg_u133_plus_2"))
df2 <- data.frame(df2)

### Obtain unique ProbIDs for data ###
a <-unique(df2["ProbID"])

### Feed the processed data To SAM(significance Analysis of Microarrays)

install.packages("devtools") 
library(devtools)
install_github("cran/samr")
install.packages("samr")

library(shiny)
library(impute)
runGitHub("SAM", "MikeJSeo")

