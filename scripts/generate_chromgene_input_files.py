#!/usr/bin/python3.8 -tt

import sys
import gzip
import argparse
import random
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
from smart_open import open
import warnings
import itertools
from joblib import Parallel, delayed


class Gene:
    def __init__(self, chromosome=None, left=1e10, right=0, strand=None, name=None, gene_id=None, gene_type=None):
        self.chromosome = str(chromosome)
        self.left = int(left)
        self.right = int(right)
        self.strand = strand
        self.name = name
        self.gene_id = gene_id
        self.gene_type = gene_type
        self.exons = []

    def add_exon(self, start, end):
        # adds exon to list of exons and updates left-most and right-most coordinates
        self.exons.append((start, end))
        self.left = min(self.left, start)
        self.right = max(self.right, end)

    def get_idx(self, resolution=200):
        # returns start and end idxs for a gene
        return [int(self.left / resolution), int(self.right / resolution) + \
            bool(self.right % resolution != 0)]

    def get_tss_idx(self, resolution=200):
        if self.strand == "+":
            return int(self.left / resolution)
        else:
            assert self.strand == "-"
            return int(self.right / resolution)


def partition(num_elts, norm=1.):
    ### returns a random vector of length specified with sum of 1
    x = np.empty(num_elts)
    for k in range(num_elts):
        x[k] = random.uniform(.2, .8)
    return x/sum(x) * norm


def even_subsample(len_from, len_to):
    ### Take in a starting length and a desired length. Return an array of indexes corresponding
    ### to positions that are sampled with as even spacing as possible. For example,
    ### even_subsample(7,3) would return [0, 3, 6]
    if len_to <= len_from:
        raise ValueError(f"trying to sample {len_to} positions from a gene of length {len_from}")
        
    # Have a fractional step size so we can take different size (index) steps
    # Subtract 1 from len_from because that is the final index
    # Subtract 1 from len_to because we want (len_to - 1) divisions of the data and end on final idx
    step_size = float(len_from-1) / (len_to-1)
    # make a list of floats, convert them to integers
    return np.array([int(k) for k in [step_size*l for l in range(len_to)]])


def read_gtf(gtf_file, chroms, output_bed=None):
    ### Reads GTF into a list of genes
    seen_gene_ids = set()
    prev_gene_id = None
    gene_list = []
    with open(gtf_file) as infile:
        for line in infile:
            if line.startswith("#"):
                continue
            field = line.strip().split('\t')
            if (field[2] != 'exon') or ('_' in field[0]) or ('_dup' in field[-1]) or (field[0] not in chroms):
                continue
            desc_fields = field[-1].split()
            gene_id_idx = desc_fields.index("gene_id") + 1
            gene_type_idx = desc_fields.index("gene_type") + 1
            gene_name_idx = desc_fields.index("gene_name") + 1

            gene_id = desc_fields[gene_id_idx].lstrip('"').rstrip('";')
            gene_type = desc_fields[gene_type_idx].lstrip('"').rstrip('";')
            gene_name = desc_fields[gene_name_idx].lstrip('"').rstrip('";')

            if not gene_id == prev_gene_id:
                assert gene_id not in seen_gene_ids, f"{gene_id=} already encountered. GTF must not be sorted by gene id! Offending {line=}"
                prev_gene_id = gene_id
                seen_gene_ids.add(gene_id)
                gene_list.append(Gene(chromosome=field[0], strand=field[6], name=gene_name, gene_id=gene_id, gene_type=gene_type))

            gene_list[-1].add_exon(int(field[3]), int(field[4]))

    if output_bed is not None:
        with open(output_bed, "w") as outfile:
            for gene in gene_list:
                outfile.write(f"{gene.chromosome}\t{gene.left}\t{gene.right}\t{gene.name}\t.\t{gene.strand}\t{gene.gene_id}\t{gene.gene_type}\n")

    return gene_list


def read_bed(bed_file):
    ### Reads BED into a list of genes
    gene_list = []
    with open(bed_file) as infile:
        for line in infile:
            chrom, start, stop, gene_name, score, strand = line.strip().split('\t')
            if chrom == 'chrY':
                continue
            gene_list.append(Gene(chromosome=chrom, left=start, right=stop, strand=strand, name=gene_name))

    return gene_list


def print_binaries(cell_input_files, cell, chroms, gene_list, features, args):
    hist_dict = {}
    num_features = len(features)
    
    for mark_file in tqdm(cell_input_files[cell], desc=f"Reading data for cell type {cell}", disable=args.workers > 1):
        with open(mark_file) as infile:

            #    E007    chr1
            #    H3K27ac    H3K27me3    H3K36me3    H3K4me1    H3K4me3    H3K9me3 ...
            #    1    0    0    1    0    0
            #    1    0    1    0    1    0
            #    1    0    1    0    0    0

            # Read first line just for the name of the cell line and the chromosome number
            [_cell, chromosome] = infile.readline().split()
            hist_dict[chromosome] = pd.read_table(infile, skiprows=2, names=features)

    if not args.no_binary:
        
        # Set up a dict of open files with the chromosome as the key and print headers
        outfile_dict = {}
        outfile_ID_dict = {}
        for chrom in chroms:
            outfile_dict[chrom] = gzip.open(args.out_dir + cell + '_' + chrom + '_gene_binary.txt.gz','wb')
            outfile_dict[chrom].write(f"{cell}\t{chrom}\n".encode())
            outfile_dict[chrom].write(('\t'.join(features + ['dummy']) + '\n').encode())
            outfile_dict[chrom].write(('\t'.join(['0'] * (num_features) + ['1'] ) + '\n').encode())
        
        for chrom in chroms:
            outfile_ID_dict[chrom] = gzip.open(args.out_dir + chrom + '_ID.bed.gz', 'wb')

        # We generate the matrix on the fly and print it
        for gg, gene in tqdm(
            enumerate(gene_list), 
            desc=f'Printing gene-level histone marks for cell type {cell}',
            total=len(gene_list), 
            disable=args.workers > 1,
        ):

            if gene.chromosome in chroms:
                # Find values for features anchored on start of gene
                [left_idx, right_idx] = gene.get_idx(args.resolution)
                # Extend the window by a certain size (default 2kb)
                left_idx = left_idx - (args.window // args.resolution)
                right_idx = right_idx + (args.window // args.resolution)

                if gene.strand == '+':
                    # Print the gene's histone mark vales
                    for line in np.array(hist_dict[gene.chromosome][left_idx:right_idx]):
                        outfile_dict[gene.chromosome].write(
                            ('\t'.join([str(int(k)) for k in line] + ['0']) + '\n').encode()
                        )

                # If the strand is negative, have to use 'right_pos' instead as start of gene
                elif gene.strand == '-':
                    # Print the gene's histone mark vales
                    for line in np.array(hist_dict[gene.chromosome][right_idx:left_idx:-1]):
                        outfile_dict[gene.chromosome].write(
                            ('\t'.join([str(int(k)) for k in line] + ['0']) + '\n').encode()
                        )
                else:
                    raise ValueError(f"Invalid strand: {gene.strand}")
                
                # Print a 1 for the dummy, then 0s for the rest of the values
                outfile_dict[gene.chromosome].write(
                    ('\t'.join(['0'] * (num_features) + ['1'] ) + '\n').encode()
                )

                # Write to the ID file to keep track of the genes
                outfile_ID_dict[gene.chromosome].write(('\t'.join(
                    [gene.chromosome, str(gene.left), str(gene.right), gene.name, ".", gene.strand]
                ) + '\n').encode())


        # Close all files
        for outfile in outfile_dict.values():
            outfile.close()
        for file in outfile_ID_dict.values():
            outfile.close()

        if args.output_tss:
            outfile_dict = {}
            outfile_ID_dict = {}
            for chrom in chroms:
                outfile_dict[chrom] = gzip.open(args.out_dir + cell + '_' + chrom + '_tss_binary.txt.gz','wb')
                outfile_dict[chrom].write((cell + '\t' + chrom + '\n').encode())
                outfile_dict[chrom].write(('\t'.join(features + ['dummy']) + '\n').encode())
                outfile_dict[chrom].write(('\t'.join(['0'] * (num_features) + ['1'] ) + '\n').encode())
            
            for chrom in chroms:
                outfile_ID_dict[chrom] = gzip.open(args.out_dir + chrom + '_ID.bed.gz', 'wb')

            # we generate the matrix on the fly and print it
            for gg, gene in tqdm(enumerate(gene_list), total=len(gene_list), disable=args.workers > 1):
                if gene.chromosome in chroms:

                    # Find values for features anchored on start of gene
                    tss_idx = gene.get_tss_idx()

                    for cell in cells:
                        line = np.array(hist_dict[gene.chromosome][tss_idx])
                        outfile_dict[gene.chromosome].write(
                            ('\t'.join([str(int(k)) for k in line] + ['0']) + '\n').encode()
                            )

                        # Print a 1 for the dummy, then 0s for the rest of the values
                        outfile_dict[gene.chromosome].write(
                            ('\t'.join(['0'] * (num_features) + ['1'] ) + '\n').encode()
                            )

                    outfile_ID_dict[gene.chromosome].write(
                        ('\t'.join(
                            [gene.chromosome, str(gene.left), str(gene.right), gene.name, ".", gene.strand]
                        ) + '\n').encode()
                    )

        # Close all files
        for outfile in outfile_dict.values():
            outfile.close()
        for file in outfile_ID_dict.values():
            outfile.close()


def generate_emission_mat(args, features, sample_binary_emission_file=None):
    ### Create an emission matrix for each state based on random assignment to mixtures

    num_features = len(features)
    # total number of states besides dummy
    state_counts = np.array([0.] * (args.num_states * args.num_mixtures))
    # emission_mat excludes the gene and dummy features
    emission_mat = np.zeros((args.num_states * args.num_mixtures + 1, 
        num_features + 1))

    if sample_binary_emission_file is None:
        for file in os.listdir(args.out_dir):
            if 'chr1_gene_binary.txt' in file:
                sample_binary_emission_file = os.path.join(args.out_dir, file)
                break

    # loop over a binary emission files, then take each position and assign it to a state.
    if not os.path.exists(sample_binary_emission_file):
        raise ValueError(
            f"{sample_binary_emission_file=} is not found. "
            "If sample_binary_emission_file is not set, ensure that args.out_dir/*chr1_gene_binary.txt.gz exists. "
            "If sample_binary_emission_file is set, then the file does not exist."
        )

    with open(sample_binary_emission_file, 'r') as infile:
        # pass through first two lines
        infile.readline()
        infile.readline()

        for line in infile:
            emissions = [int(k) for k in line.split()]

            # if it's a dummy state, skip it
            if emissions[-1] == 1:
                mixture_choice = np.random.choice(np.arange(args.num_mixtures))
                continue
            else:
                # subsample
                if np.random.random() > args.subsample:
                    continue
                this_state = mixture_choice*args.num_states + np.random.choice(np.arange(args.num_states))
                emission_mat[this_state, :-1] += emissions[:-1]
                state_counts[this_state] += 1

    # Add pseudocount and normalize counts of emissions to generate probabilities
    emission_mat[:-1, :-1] += 1.
    state_counts[:] += 1.
    for state in range(args.num_states * args.num_mixtures):
        emission_mat[state] = emission_mat[state] / state_counts[state]
    # set emission for dummy variable
    emission_mat[-1, -1] = 1.
    return emission_mat


def main():

    ############################################################################################
    ############################################################################################
    ####                                                                                    ####
    ####    This script generates the necessary files to use ChromGene on top of            ####
    ####    ChromHMM: binarized emissions and initial probability files for training.       ####
    ####                                                                                    ####
    ####    Instead of running each gene as a separate observation or file, we              ####
    ####    concatenate all the genes on each chromosome, separated by a dummy variable     ####
    ####    which is always 0 except for the postions flanking each gene. Also, at the      ####
    ####    start of gene, an additional track emits '1' until the end of the region, so    ####
    ####    that we can separate the states upstream from the ones downstream.              ####
    ####                                                                                    ####
    ####    Step 0:                                                                         ####
    ####        Read in arguments and data                                                  ####
    ####                                                                                    ####
    ####    Step 1:                                                                         ####
    ####        Take binarized calls for each gene's histone marks                          ####
    ####        Concatenate them all and separate by dummy variable, which is 1             ####
    ####        in separations and at beginning and end                                     ####
    ####                                                                                    ####
    ####    Step 2:                                                                         ####
    ####        Print out prior probabilities for each state. Will be 0 for all states      ####
    ####        except for dummy state                                                      ####
    ####                                                                                    ####
    ####    Step 3:                                                                         ####
    ####        Print out transition matrix. The transition of each state to itself         ####
    ####        is p, and into the next state is 1-p. Can only transition into              ####
    ####        dummy state from last state in each mixture. From dummy state, must         ####
    ####        transition into one of the mixtures' first state. Combine into              ####
    ####        large transition matrix concatenated along diagonal                         ####
    ####                                                                                    ####
    ####    Step 4:                                                                         ####
    ####        Randomly initialize emission probabilities, or use emission probabilities   ####
    ####        from previously-learned ChromHMM run, but randomly chosen in order          ####
    ####                                                                                    ####
    ############################################################################################
    ############################################################################################

    parser = argparse.ArgumentParser()
    parser.add_argument('annotation', help='annotation file of genes to use to generate binary marks (gtf, bed)')
    parser.add_argument('mark_files', help='binary calls of histone marks via ChromHMM', nargs='+')
    parser.add_argument('--output_bed', '--output-bed', help='output BED file for annotation if reading a GTF')
    parser.add_argument('--num_states', '--num-states', help='number of states in each gene mixture', default=3, type=int)
    parser.add_argument('--num_mixtures', '--num-mixtures', help='number of gene mixtures', default=12, type=int)
    parser.add_argument('--no_binary', '--no-binary', help='skip printing output binary histone mark calls', action='store_true')
    parser.add_argument('--no_model_param', '--no-model-param', help='skip printing output model parameters', action='store_true')
    parser.add_argument('--resolution', help='resolution of binarized histone mark features', type=int, default=200)
    parser.add_argument('--chroms', help='chromosomes to include in analysis', nargs='+', default=list(range(1, 23)) + ['X'])
    parser.add_argument('--subsample', help='fraction to subsample for seeding emission parameters', default=1., type=float)
    parser.add_argument('--sample_binary_emission_file', '--sample-binary-emission-file', help='file to sample from when determining initial emission parameters')
    parser.add_argument('--window', help='bases to go in each of upstream/downstream of gene', type=int, default=2000)
    parser.add_argument('--output_tss', '--output-tss', help='output emissions at the TSS', default=False, action='store_true')
    parser.add_argument('--out_dir', '--out-dir', help='directory in which to save data', default='.')
    parser.add_argument('--workers', help='number of cores to use for writing', type=int, default=1)
    parser.add_argument('--verbose', default=False, action='store_true')

    args = parser.parse_args()

    if args.out_dir[-1] != '/':
        args.out_dir += '/'
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    chroms = []
    for chrom in args.chroms:
        chrom = str(chrom)
        if chrom.startswith('chr'):
            chroms.append(chrom)
        else:
            chroms.append(f'chr{chrom}')

    ############################################################################################
    ####    Step 1: Read in gene annotation file                                            ####
    ############################################################################################

    sys.stderr.write('Reading annotation file...\n')
    if '.gtf' in args.annotation:
        warnings.warn(
            "Warning! GTF files are difficult to parse, and may return unexpected results. Please use "
            "BED files wherever possible, or gtfs at your peril!",
            UserWarning,
        )
        gene_list = read_gtf(args.annotation, chroms, args.output_bed)
    elif ".bed" in args.annotation:
         gene_list = read_bed(args.annotation)
    else:
        raise ValueError(f"{args.annotation} is neither a GTF nor BED file")

    num_genes = len(gene_list)

    ############################################################################################
    ####    Step 2: Read binarized histone marks and print out binarized marks to file      ####
    ############################################################################################

    if (not args.no_binary) or args.output_tss:
        # First determine the cell types and chromosomes to loop over
        cells = set()
        input_chroms = set()
        cell_input_files = {}
        features = None
        for mark_file in args.mark_files:
            with open(mark_file) as infile:
                [cell, chrom] = infile.readline().split()
                _features = infile.readline().split()
                if features is not None and _features != features:
                    raise ValueError(
                        f"Features found in file {mark_file} do not correspond to previous files: " \
                        f"Previous {features=}, new features={_features}"
                    )

                features = _features
                if not cell in cells:
                    cells.add(cell)
                    cell_input_files[cell] = []
                if not chrom in input_chroms:
                    input_chroms.add(chrom)
                cell_input_files[cell].append(mark_file)
        num_features = len(features)

        if args.workers > 1:
            Parallel(args.workers)(delayed(print_binaries)(cell_input_files, cell, chroms, gene_list, features, args) for cell in tqdm(cells, total=len(cells), desc="Printing binaries"))
        else:
            for cc, cell in enumerate(cells):
                print_binaries(cell_input_files, cell, chroms, gene_list, features, args)

    ############################################################################################
    ####    Step 3: Print out initial probabilities                                         ####
    ############################################################################################

    if args.no_model_param:
        sys.exit()
    sys.stderr.write('Printing ChromHMM initial parameters\n')

    # print header: number of states, number of marks, 'E', model likelihood, number of iterations
    outfile = open(args.out_dir + 'model_' + str(args.num_states * args.num_mixtures + 1) + '.txt','w')
    outfile.write(str(args.num_states * args.num_mixtures + 1) + '\t' + str(num_features + 1) + '\tE\t-2E7\t200\n')

    # print initial probabilities. There are (mixture * states) states, followed by dummy state
    for state_idx in range(1, 1+(args.num_states * args.num_mixtures)):
        outfile.write('probinit\t'+str(state_idx)+'\t0.0\n')
    outfile.write('probinit\t' + str(1+(args.num_states * args.num_mixtures)) + '\t1.0\n')

    ############################################################################################
    ####    Step 4: Print out transition probabilities                                      ####
    ############################################################################################

    # There are a total of (mixture * states + 1) states. Can transition from dummy state 
    # to state (n mod states)==1
    # States (n mod states) == -1 can only transition to themselves or last state
    # All other states can transition to themselves or the next state

    # Print transitions from mixture states to other states
    sys.stderr.write('Outputting transition parameters\n')
    transition_mat = np.zeros((args.num_states * args.num_mixtures + 1, \
            args.num_states * args.num_mixtures + 1))
    # set transitions 
    for mixture_idx in range(args.num_mixtures):
        to_self = random.uniform(.85, .95)
        for state_idx in range(args.num_states):
            to_state = partition(args.num_states, norm=to_self)
            # transitions within mixture
            transition_mat[
                args.num_states*mixture_idx+state_idx, 
                args.num_states*mixture_idx:args.num_states*(mixture_idx+1)
            ] = to_state
            transition_mat[args.num_states*mixture_idx+state_idx, -1] = 1-to_self
    
    # set transitions from dummy to other states. Should be able to go to any state in any mixture
    dummy_to_state = partition(args.num_states*args.num_mixtures)
    transition_mat[-1, :-1] = dummy_to_state

    # print transition matrix
    for idx_from, line_from in enumerate(transition_mat):
        for idx_to, value in enumerate(line_from):
            outfile.write('\t'.join([str(k) for k in ["transitionprobs", idx_from + 1, idx_to + 1, value]]) + '\n')

    ############################################################################################
    ####    Step 5: Print out emission probabilities                                        ####
    ############################################################################################

    sys.stderr.write('Smartly initializing emission parameters\n')
    # emission_prob_mat is a (states, features) matrix with 
    # args.states * args.mixtures total states
    # features do not include 'dummy' and 'gene' states, only histone marks
    emission_prob_mat = generate_emission_mat(args, features, sample_binary_emission_file=args.sample_binary_emission_file)

    # Print emissions for each state, which is randomly generated
    # emissionprobs   1       0       N0-R    0       0.02042128
    # emissionprobs   1       0       N0-R    1       0.97957871
    sys.stderr.write('Printing emission parameters\n')
    features = features + ['dummy']
    for row_idx, row in enumerate(emission_prob_mat):
        for col_idx, prob in enumerate(row):
            outfile.write('\t'.join(
                [str(k) for k in ["emissionprobs", row_idx + 1, col_idx, features[col_idx], 0, 1-prob]]
            ) + "\n")
            outfile.write('\t'.join(
                [str(k) for k in ["emissionprobs", row_idx + 1, col_idx, features[col_idx], 1, prob]]
            ) + "\n")

    outfile.close()

if __name__ == '__main__':
    main()
