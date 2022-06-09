# ChromGene: Gene-Based Modelling of Epigenomic Data

## Design
ChromGene is a method for annotating genomic intervals by modeling them as arising from a mixture of Hidden Markov Models (HMMs). As with a Gaussian Mixture Model, where each point is assumed to have been generated by exactly one Gaussian distribution, each gene is assumed to be generated by exactly one HMM. Each HMM is fully-connected, and is defined by emission parameters of each of S states, the transition parameters between them, and the prior probabilities of each state. The preprint is available at https://www.biorxiv.org/content/10.1101/2022.05.24.493345v1.

## Usage
ChromGene can be applied to any genomic intervals where histone marks and DNA accessibility tracks are expected to vary across the genome by modifying the provided code.

We applied ChromGene to a set of 19,919 protein-coding genes across 127 imputed epigenomes (Frankish et al. 2018, Roadmap Epigenomics Consortium et al. 2015). The full set of annotations is reported in this repository, and their descriptions and various metrics are reported in Supp. Table 1. Along with the annotations, we have included code to generate input files to use on top of ChromHMM (Ernst and Kellis, 2012), and to generate a ChromGene assignment matrix.

Usage requires only basic Python packages, which can be easily installed with Anaconda or pip.

First, create the input binary files to use on top of ChromHMM. This will require either a GTF or BED file to demarcate the positions of genes:
```
python generate_chromgene_input_files.py \
annotation [path to bed or gtf file] \
mark_files [ChromHMM chromatin mark binary call file paths] \
--num_states [number of states per mixture component; default: 3] \
--num_mixtures [number of mixtures components; default: 12] \
--binary [whether to output binary files, can be set to False for testing; default: True] \
--model_param [whether to output model param files, can be set to False for testing; default: True] \
--resolution [resolution of data to output; default: 200] \
--subsample [fraction to subsample positions to initialize ChromGene emission parameters; default: 1] \
--window [bases to go up/downstream of TSS/TES for flanking regions; default: 2000] \
--output_tss [only output TSS binary; default: False] \
--out_dir [output directory; default="."] \
--verbose [verbose; default: False]
```

Second, run ChromHMM on the binary files, passing the model file and binaries deposited into `out_dir`. The argument `total_num_states` should be (num_mixture components * num_states) + 1. The latest version of ChromHMM can be downloaded at https://ernstlab.biolchem.ucla.edu/ChromHMM/. The following command and flags should be used:
```
java -jar -mx24000M  path/to/ChromHMM.jar LearnModel \
-b 200 \
-d -1 \
-gzip \
-scalebeta \
-holdcolumnorder \
-holdroworder \
-init load \
-m path_to_model_file/model_total_num_states.txt \
-n 100 \
-e 0 \
-t 0 \
-p 16 \
-lowmem \
-printstatebyline \
-nobrowser \
-noenrich \
./binaries/chromgene_binaries \
./out_dir \
total_num_states \
assembly
```

Finally, create the ChromGene mixture assignments. This will create .npy files and .txt files
```
python chromgene_posteriors_to_mixtures.py
--segmentation_dir [directory with segmentation files; default: '.']
--states_per_mixture [number of states per mixture; default: 4]
--out_dir [directory in which to save data; default: '.']
```

## Annotations
We have generated ChromGene annotations for 127 cell types across 19,919 protein-coding genes using a single, unified model trained on 11 imputed histone marks and DNase. We have included these files in the main repository in the files `chromgene_assignments_eids.tsv.gz` for columns corresponding to EIDs, and the longer cell type name in `chromgene_assignments_cell_type_names.tsv.gz`. The correspondence between these two is available through Roadmap at https://docs.google.com/spreadsheets/d/1yikGx4MsO9Ei36b64yOy9Vb6oPC5IBGlFbYEt-N6gOM/edit#gid=15.


## Supplementary Data: ChromGene annotations
This is a gzip of a tab delimited file containing the ChromGene annotations for the 127 cell types. Each row after the header row corresponds to one gene. The first five columns from left to right are the chromosome of the gene, the left-most coordinate of the gene, the right-most coordinate of the gene, the gene symbol, and strand of the gene. This is based on hg19 and ENSEMBL v65/GENCODE v10. The remaining columns correspond to different cell types for which ChromGene annotations are reported as indicated by the names in the header.
