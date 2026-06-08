DALMNPred: Disease-Associated lncRNA-miRNA Network Prediction
Overview
DALMNPred (Disease-Associated lncRNA-miRNA Network Prediction) is an integrated bioinformatics and machine learning framework designed to identify biologically relevant lncRNA-miRNA-mRNA regulatory networks from transcriptomic datasets.
This repository contains R and Python scripts used for:
•	Differentially expressed lncRNA (DElncRNA) identification from microarray datasets.
•	lncRNA feature selection using machine learning approaches.
•	Classification of biological samples based on lncRNA expression signatures.
•	Weighted Gene Co-expression Network Analysis (WGCNA) for identification of co-expressed gene modules.
•	miRNA target prediction and validation.
•	Construction of lncRNA-miRNA-mRNA regulatory networks.
Although originally developed to investigate fetal hemoglobin (HbF) regulation and β-thalassemia, the workflow can be adapted to other diseases and biological systems.
________________________________________
Repository Contents
1. Differential Expression Analysis
File: File 1.DEGs.R
This script performs:
•	Download of GEO microarray datasets.
•	RMA normalization of Affymetrix microarray data.
•	Probe annotation and lncRNA extraction.
•	Identification of differentially expressed lncRNAs using Significance Analysis of Microarrays (SAM).
•	Preparation of data for hierarchical clustering and downstream analyses.

2. Machine Learning Classification
File: File 2. ML.py
This Python script performs:
•	Data preprocessing and train-test splitting.
•	Sequential Forward Feature Selection (SFS).
•	Random Forest classification.
•	Support Vector Machine (Linear and RBF kernel) classification.
•	Performance evaluation using:
o	Accuracy
o	Precision
o	Recall
o	F1-score
o	Confusion Matrix
o	ROC Curve Analysis
The workflow identifies optimal lncRNA signatures capable of distinguishing biological conditions.

3. Weighted Gene Co-expression Network Analysis (WGCNA)
File: R_code_wgcna.R
This script performs:
•	Construction of weighted gene co-expression networks using the WGCNA framework.
•	Selection of optimal soft-thresholding power based on scale-free topology criteria.
•	Generation of adjacency and topological overlap matrices (TOM).
•	Hierarchical clustering of genes based on network topology.
•	Detection of co-expression modules using dynamic tree cutting.
•	Merging of highly similar modules based on eigengene correlations.
•	Identification and export of module-specific gene sets.
•	Calculation of module eigengenes and module relationships.
•	Correlation analysis between candidate lncRNAs and mRNA expression profiles.
•	Identification of co-expressed mRNAs associated with machine learning-selected lncRNAs.
•	Output files include:
•	Module assignment files (e.g., blue, brown, turquoise, yellow modules).
•	Gene-gene topological overlap matrices.
•	lncRNA-mRNA correlation matrices.
•	Saved WGCNA workspace objects (.RData) for downstream analyses.

4. miRNA Target Prediction
File: mirna_target_prediction.R
This script utilizes the MultiMiR database to:
•	Predict miRNA-target interactions.
•	Retrieve experimentally validated targets.
•	Export predicted and validated interaction datasets.
•	Support downstream ceRNA network construction.
5. Supporting Data Files
•	.RData: Saved R workspace containing intermediate analysis objects and processed datasets.
________________________________________
Workflow
1. Obtain and preprocess GEO microarray datasets.
2. Identify differentially expressed lncRNAs.
3. Select optimal lncRNA sets using machine learning.
4. Construct weighted gene co-expression networks (WGCNA).
5. Identify co-expression modules associated with candidate lncRNAs.
6. Predict lncRNA-associated miRNAs.
7. Retrieve predicted and validated miRNA targets.
8. Integrate co-expression and miRNA-target information.
9. Construct lncRNA-miRNA-mRNA regulatory networks.
10. Interpret biological significance and therapeutic potential.
Software Requirements
R
Recommended version: R ≥ 4.0
Required packages include:
•	GEOquery
•	 affy
•	 hgu133a.db
•	 hgu133acdf
•	 biomaRt
•	 dplyr
•	 readr
•	 samr
•	 WGCNA
•	 multiMiR
•	 shiny
•	 impute
Python
Recommended version: Python ≥ 3.7
Required libraries:
•	numpy
•	pandas
•	matplotlib
•	scikit-learn
•	mlxtend
Install Python dependencies using:
pip install numpy pandas matplotlib scikit-learn mlxtend
________________________________________
Applications
This workflow can be applied to:
•	Fetal hemoglobin regulation studies
•	Hemoglobinopathies
•	Cancer transcriptomics
•	Biomarker discovery
•	ceRNA network analysis
•	Non-coding RNA biology
•	Systems biology investigations
________________________________________
Citation
If you use this repository in your research, please cite the corresponding publication describing the DALMNPred framework.
________________________________________
Contact
## Authors
- Motiur Rahaman (First Author)
- Nishant Chakravorty (Corresponding Author)

## Contact
For scientific inquiries regarding the study:
Dr. Nishant Chakravorty
School of Medical Science and Technology (SMST)
Indian Institute of Technology Kharagpur
Email: nishant@smst.iitkgp.ac.in. 
For code-related issues:
Dr. Motiur Rahaman
School of Medical Science and Technology (SMST)
Indian Institute of Technology Kharagpur
Email: motiurrahaman24@iitkgp.ac.in.

For scientific questions, suggestions, or collaborations, please feel free to contact.
