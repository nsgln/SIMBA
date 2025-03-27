# Real Graphs Results

This directory contains the results of the analysis of the real-world gene expression data using the NetREX algorithm. The data was obtained from the GTEx project and consists of gene expression profiles across three different human tissues: kidney, brain, and heart. 
The analysis aimed to identify gene modules that are specific to each tissue and to explore the biological processes and pathways associated with these modules.

## Content
### Figure 1. EnrichmentMap of Selected Gene Modules Across Organs 
The network diagram illustrates enriched pathways identified in selected modules (two modules per organ) across kidney, brain, and heart tissues. Each node represents a specific biological process or pathway, annotated using Gene Ontology (GO). Node sizes correspond to the number of genes associated with each pathway, and node colors indicate the organ where the module was identified: Green for kidney, Blue for brain, and Yellow for heart. Gray edges indicate shared genes or functional relationships between pathways, emphasizing interconnected processes across tissues.

### Supplementary Table S10. Gene Set Enrichment Analysis for Top Modules in Kidney. 
This table provides Gene Ontology (GO) and KEGG annotations for the top six modules identified in the kidney. The analysis shows key biological processes and pathways enriched within each module, with an adjusted p-value threshold of > 0.05 and at least two associated genes.

### Supplementary Table S11. Gene Set Enrichment Analysis for Top Modules in Brain. 
This table provides Gene Ontology (GO) and KEGG annotations for the top six modules identified in the brain. The analysis shows key biological processes and pathways enriched within each module, with an adjusted p-value threshold of > 0.05 and at least two associated genes.

### Supplementary Table S12. Gene Set Enrichment Analysis for Top Modules in Heart. 
This table provides Gene Ontology (GO) and KEGG annotations for the top six modules identified in the heart. The analysis shows key biological processes and pathways enriched within each module, with an adjusted p-value threshold of > 0.05 and at least two associated genes.

### Supplementary Table S8. Modules and Component Genes
This table lists the selected modules along with their respective component genes, highlighting the Fold Change (FC) values assigned to each gene within each module. The data provides insights into the contribution of individual genes to the identified modules.

### Graph_Tammaro_2024_Brain
This file contains the output of the SIMBA algorithm for the brain tissue.

### Graph_Tammaro_2024_Heart
This file contains the output of the SIMBA algorithm for the heart tissue.

### Graph_Tammaro_2024_Kidney
This file contains the output of the SIMBA algorithm for the kidney tissue.