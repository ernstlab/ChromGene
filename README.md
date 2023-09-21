# ChromGene: Gene-Based Modelling of Epigenomic Data

## Design
ChromGene is a method for annotating genomic intervals by modeling them as arising from a mixture of Hidden Markov Models (HMMs). As with a Gaussian Mixture Model, where each point is assumed to have been generated by exactly one Gaussian distribution, each gene is assumed to be generated by exactly one HMM. Each HMM is fully-connected, and is defined by emission parameters of each of S states, the transition parameters between them, and the prior probabilities of each state. The manuscript is available at https://genomebiology.biomedcentral.com/articles/10.1186/s13059-023-03041-5.

## Usage
ChromGene can be applied to any genomic intervals where histone marks and DNA accessibility tracks are expected to vary across the genome by modifying the provided code.

We applied ChromGene to a set of 19,919 protein-coding genes across 127 imputed epigenomes (Frankish et al. 2018, Roadmap Epigenomics Consortium et al. 2015). The exact dataset used in our analysis can be accessed at https://egg2.wustl.edu/roadmap/data/byFileType/chromhmmSegmentations/binaryChmmInput/imputed12marks/binaryData/. The full set of annotations is reported in the directory `chromgene_assignments`, and their descriptions and various metrics are reported in Supp. Table 1. Along with the annotations, we have included code to generate input files to use on top of ChromHMM (Ernst and Kellis, 2012), and to generate a ChromGene assignment matrix.

### 0. Setup
#### 0a. Preparing your environment
Usage requires only basic Python packages, which can be easily installed with Conda / Mamba (recommended) or pip. For Conda / Mamba installation:
1. Install Conda (https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html), or better yet, Mamba (https://mamba.readthedocs.io/en/latest/installation.html)
2. Create a conda environment with Python 3.8: `conda create -n chromgene python=3.8`
3. Activate the environment: `conda activate chromgene`
4. Install packages: `conda install -c conda-forge tqdm numpy pandas smart_open glob2 joblib`. If using Mamba, replace `conda` with `mamba`.

#### 0b. Downloading the ChromGene code
Download the ChromGene code to your directory of choice (e.g., `git clone https://github.com/ernstlab/ChromGene.git`). The scripts should be usable out of the box as long as you have the above environment activated.

### 1. Create ChromGene binary files
#### 1a. Download or generate ChromHMM binarized mark files
This follows the process described at (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5945550/). This is a tab-delimited text file of the format:
```
E003    chrX
DNase   H3K27ac H3K9ac  ...       H2A.Z
0       0       0       ...       1
0       1       1       ...       0
0       0       1       ...       0
```

We used imputed ChromHMM mark files for 129 epigenomes, found at https://egg2.wustl.edu/roadmap/data/byFileType/chromhmmSegmentations/binaryChmmInput/imputed12marks/binaryData/.

If you would like to use your own data for ChromGene, you will have to create your own ChromHMM binarized mark files, following the ChromHMM command BinarizeBed, BinarizeBam, or BinarizeSignal (http://compbio.mit.edu/ChromHMM/). 

#### 1b. Create ChromGene binary files using ChromHMM binary files
Next, create the ChromGene input binary files to use on top of ChromHMM. This will require either a GTF or BED (recommended) file to demarcate the positions of genes. Additionally, you will the need tab-delimited, binarized mark files taken from step 1a.

***Note:*** *We do not recommend using a GTF file, as they tend to be very different in structure and are difficult to parse. We have retained the function used for ChromGene for posterity, but have disabled the function, and leave it to the user to create an appropriate BED file. A simple command to create such a BED file could look something like:* `zcat gencode.v29.annotation.gtf.gz | awk 'BEGIN {OFS="\t"} ($3 == "gene") && ($0 ~ /protein_coding/) {split($10, gid, "\""); split($14, name, "\""); print $1, $4, $5, gid[2], name[2], $7}' > gencode.v29.annotation.bed`

***Note on hyperparameters:*** *We determined the number of components (`--num-components`) and states per component (`--num-states`) using the process descibed in our manuscript. This is likely a good starting place for you, but you may want to increase or decrease the number of components or states for your specific use case. If you have more marks than 12, you may want to use more states and/or components, and if you have fewer than 6 marks, you may want to use fewer states and/or components.*

Usage (note that argument names in brackets such as `[--no-binary]` are optional, and arguments themselves in brackets such as `--num-states [3]` denote default arguments:
```
python generate_chromgene_input_files.py \
annotation [path to bed or gtf file] \
mark_files [ChromHMM chromatin mark binary call file paths] \
--num-states [3] \  # number of states per mixture component
--num-components [12] \  # number of mixtures components
[--no-binary] \  # skip printing output binary files, can be used for testing
[--no-model-param] \  # skip printing output model param files, can be used for testing
--resolution [200] \  # resolution of data to output
--chroms [1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 X] \  # only use these chromosomes
--subsample [1] \  # fraction to subsample positions to initialize ChromGene emission parameters
--window [2000] \  # bases to go up/downstream of TSS/TES for flanking regions
[--output-tss] \  # output TSS binaries for baseline comparison
--out-dir ["."] \  # output directory
[--quiet]  # Disable progress messages
```

### 2. Train ChromGene model
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
-printposterior \
-nobrowser \
-noenrich \
./binaries/chromgene_binaries \
./out_dir \
total_num_states \
assembly
```

### 3. Create ChromGene assignments
Finally, create the ChromGene component assignments.
```
python chromgene_posteriors_to_components.py \
--posteriors-dir ["."] \  # directory with posterior probability files
--states-per-component [3] \  # number of states per mixture component
--out-dir ["."] \  # directory in which to save data
--ids-dir ["."] \  # directory that has ID files generated by the first step
--chroms [1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 X] \  # only use these chromosomes
[--use-default-name-mapping]  # Use name mapping for original model generated by Jaroszewicz & Ernst, 2023
```

This will create .npy files and .txt files, which are matrices of shape (num celltypes, num genes), where cell types are in sorted lexical order and genes are sorted by chromosome, then by order in the original BED file (which is also identical to the `{chrom}_ID.bed.gz` files created in the first step. Additionall, a tsv.gz file is created that contains all the gene regions used with per-celltype assignments.

## Annotations
We have generated ChromGene annotations for 127 cell types across 19,919 protein-coding genes using a single, unified model trained on 11 imputed histone marks and DNase. We have included these files in the main repository in the files `chromgene_assignments/chromgene_assts_by_eid.hg19.tsv.gz` for columns corresponding to EIDs, and the longer cell type name in `chromgene_assts_by_celltype_name.hg19.tsv.gz`. The correspondence between these two is available through Roadmap at https://docs.google.com/spreadsheets/d/1yikGx4MsO9Ei36b64yOy9Vb6oPC5IBGlFbYEt-N6gOM/edit#gid=15. This is based on hg19 and ENSEMBL v65/GENCODE v10. We have also added assignments for the hg38 assembly using the Gencode v41 annotation that was previously lifted over to hg19. Please see manuscript for details.

These are gzipped, tab delimited files containing the ChromGene annotations for the 127 cell types. Each row after the header row corresponds to one gene. The first five columns from left to right are the chromosome of the gene, the left-most coordinate of the gene, the right-most coordinate of the gene, the gene symbol, and strand of the gene. The remaining columns correspond to different cell types for which ChromGene annotations are reported as indicated by the names in the header.

## Example

#### 0a. Preparing your environment
Download Mamba (https://mamba.readthedocs.io/en/latest/installation.html)
```
conda create -n chromgene python=3.8
conda activate chromgene
mamba install -c conda-forge tqdm numpy pandas smart_open glob2 joblib 
```

#### 0b. Downloading the ChromGene code
```
cd
mkdir packages && cd packages
git clone https://github.com/ernstlab/ChromGene.git
```
Now, all the ChromGene scripts will be in `~/packages/ChromGene/scripts`

### 1. Create ChromGene binary files
#### 1a. Download ChromHMM binarized mark files
```
# Download whole directory (this will take a while)
wget --recursive https://egg2.wustl.edu/roadmap/data/byFileType/chromhmmSegmentations/binaryChmmInput/imputed12marks/binaryData/
# Move binaryData directory to the current directory
mv egg2.wustl.edu/roadmap/data/byFileType/chromhmmSegmentations/binaryChmmInput/imputed12marks/binaryData/ .
```

#### 1b. Create ChromGene binary files using ChromHMM binary files
```
# Get a annotation gtf file 
wget https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_29/gencode.v29.annotation.gtf.gz
# Convert to bed file
zcat gencode.v29.annotation.gtf.gz \
| awk 'BEGIN {OFS="\t"} ($3 == "gene") && ($0 ~ /protein_coding/) {split($10, gid, "\""); split($14, name, "\""); print $1, $4, $5, name[2], gid[2], $7}' \
> gencode.v29.annotation.bed

# Make ChromGene binaries directory
mkdir chromgene_binaries

# Create input files
python scripts/generate_chromgene_input_files.py \
gencode.v29.annotation.bed \
binaryData/* \
--out-dir chromgene_binaries  # output directory
```

### 2. Train ChromGene model on top of ChromHMM
```
# Download ChromHMM
wget http://compbio.mit.edu/ChromHMM/ChromHMM.zip
unzip ChromHMM.zip

# Train model
java -jar -mx24000M  ChromHMM/ChromHMM.jar LearnModel \
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
-printposterior \
-nobrowser \
-noenrich \
./chromgene_binaries \
./chromhmm_out \
37 \
hg19
```

### 3. Create ChromGene assignments
```
python scripts/chromgene_posteriors_to_components.py \
--posteriors-dir chromhmm_out/POSTERIOR \
--out-dir chromgene_assts \
--ids-dir chromgene_binaries \
--use-default-name-mapping
```
