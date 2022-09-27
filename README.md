# ChromGene: Gene-Based Modelling of Epigenomic Data

## Design
ChromGene is a method for annotating genomic intervals by modeling them as arising from a mixture of Hidden Markov Models (HMMs). As with a Gaussian Mixture Model, where each point is assumed to have been generated by exactly one Gaussian distribution, each gene is assumed to be generated by exactly one HMM. Each HMM is fully-connected, and is defined by emission parameters of each of S states, the transition parameters between them, and the prior probabilities of each state. The preprint is available at https://www.biorxiv.org/content/10.1101/2022.05.24.493345v2.

## Usage
ChromGene can be applied to any genomic intervals where histone marks and DNA accessibility tracks are expected to vary across the genome by modifying the provided code.

We applied ChromGene to a set of 19,919 protein-coding genes across 127 imputed epigenomes (Frankish et al. 2018, Roadmap Epigenomics Consortium et al. 2015). The exact dataset used in our analysis can be accessed at https://egg2.wustl.edu/roadmap/data/byFileType/chromhmmSegmentations/binaryChmmInput/imputed12marks/binaryData/. The full set of annotations is reported in this repository, and their descriptions and various metrics are reported in Supp. Table 1. Along with the annotations, we have included code to generate input files to use on top of ChromHMM (Ernst and Kellis, 2012), and to generate a ChromGene assignment matrix.

Usage requires only basic Python packages, which can be easily installed with Anaconda (recommended) or pip. For Anaconda installation:
1. Install Conda (https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)
2. Create a conda environment with Python 3.8: `conda create -n chromgene python=3.8`
3. Activate the environment: `conda activate chromgene`
4. Install packages: `conda install -c conda-forge tqdm numpy pandas smart_open glob2 joblib`

Next, create the input binary files to use on top of ChromHMM. This will require either a GTF or BED (recommended) file to demarcate the positions of genes. Note: We do not recommend using a GTF file, as they tend to be very different in structure and are difficult to parse. We have retained the function used for ChromGene for posterity, but have disabled the function, and leave it to the user to create an appropriate BED file. Usage:
```
python generate_chromgene_input_files.py \
annotation [path to bed or gtf file] \
mark_files [ChromHMM chromatin mark binary call file paths] \
--num-states [number of states per mixture component; default: 3] \
--num-mixtures [number of mixtures components; default: 12] \
--binary [whether to output binary files, can be set to False for testing; default: True] \
--model-param [whether to output model param files, can be set to False for testing; default: True] \
--resolution [resolution of data to output; default: 200] \
--chroms [list of chromosomes to use; default: 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 X]
--subsample [fraction to subsample positions to initialize ChromGene emission parameters; default: 1] \
--window [bases to go up/downstream of TSS/TES for flanking regions; default: 2000] \
--output-tss [output TSS binaries for baseline comparison; default: False] \
--out-dir [output directory; default="."] \
--verbose [verbose; default: False]
```

Second, run ChromHMM on the binary files, passing the model file and binaries deposited into `out_dir`. The argument `total_num_states` should be `(num_mixture components * num_states) + 1`. The latest version of ChromHMM can be downloaded at https://ernstlab.biolchem.ucla.edu/ChromHMM/. The following command and flags should be used (modifiable arguments are followed by a # comment):
```
java -jar -mx24000M  path/to/ChromHMM.jar LearnModel \
-b 200 \  # The resolution of the data. 200 was used here in concordance with (Roadmap Epigenomics Consortium, 2015).
-d -1 \  # This should be set to a positive value if not sampling chromosome data files (-n). Default: 0.001.
-gzip \
-scalebeta \
-holdcolumnorder \
-holdroworder \
-init load \
-m path_to_model_file/model_total_num_states.txt \
-n 100 \  # The number of chromosome data files to sample for each training iteration. If using small datasets (fewer than 100 total chromosome input files), this argument can be omitted.
-e 0 \
-t 0 \
-p 16 \  # The number of processors to use. Can change this depending on the machine the model is trained on.
-lowmem \
-printstatebyline \
-printposterior \  # Optional, if interested in the posterior probabilities
-nobrowser \
-noenrich \
./binaries/chromgene_binaries \
./out_dir \
total_num_states \
assembly
```

Finally, create the ChromGene component assignments.
```
python chromgene_posteriors_to_mixtures.py
--segmentation-dir [directory with segmentation files; default: '.']
--states-per-mixture [number of states per mixture component; default: 3]
--out-dir [directory in which to save data; default: '.']
--ids-dir [directory that has ID files generated by the first step; default: '.']
--chroms [list of chromosomes to use; default: 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 X]
```

This will create .npy files and .txt files, which are matrices of shape (num celltypes, num genes), where cell types are in sorted lexical order and genes are sorted by chromosome, then by order in the original BED file (which is also identical to the `{chrom}_ID.bed.gz` files created in the first step. Additionall, a tsv.gz file is created that contains all the gene regions used with per-celltype assignments.

## Annotations
We have generated ChromGene annotations for 127 cell types across 19,919 protein-coding genes using a single, unified model trained on 11 imputed histone marks and DNase. We have included these files in the main repository in the files `chromgene_assignments_eids.tsv.gz` for columns corresponding to EIDs, and the longer cell type name in `chromgene_assignments_cell_type_names.tsv.gz`. The correspondence between these two is available through Roadmap at https://docs.google.com/spreadsheets/d/1yikGx4MsO9Ei36b64yOy9Vb6oPC5IBGlFbYEt-N6gOM/edit#gid=15.


## Supplementary Data: ChromGene annotations
This is a gzip of a tab delimited file containing the ChromGene annotations for the 127 cell types. Each row after the header row corresponds to one gene. The first five columns from left to right are the chromosome of the gene, the left-most coordinate of the gene, the right-most coordinate of the gene, the gene symbol, and strand of the gene. This is based on hg19 and ENSEMBL v65/GENCODE v10. The remaining columns correspond to different cell types for which ChromGene annotations are reported as indicated by the names in the header.
