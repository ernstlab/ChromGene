#!/usr/bin/python3.8 -tt

import sys
import gzip
import argparse
import os
import numpy as np
import pandas as pd
from glob import glob


def main():
    '''
    This script takes segmentation files ChromHMM output and generates a matrix of ChromGene assignments
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--segmentation_dir', '--segmentation-dir', help='dir with segmentation files', default='.', required=True)
    parser.add_argument('--ids_dir', '--ids-dir', help='directory with ID files', default='.', required=True)
    parser.add_argument('--states_per_mixture', '--states-per-mixture', help='number of states per mixture', type=int, default=3)
    parser.add_argument('--chroms', help='chromosomes to include in analysis', nargs='+', default=list(range(1, 23)) + ['X'])
    parser.add_argument('--out_dir', '--out-dir', help='directory in which to save data', default='.')
    parser.add_argument('--verbose', action='store_true')

    args = parser.parse_args()

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    chroms = []
    for chrom in args.chroms:
        if str(chrom).startswith('chr'):
            chroms.append(chrom)
        else:
            chroms.append(f'chr{chrom}')

    per_celltype_asst_dict = {}
    for file in glob(os.path.join(args.segmentation_dir, "*_segments.bed*")):
        # Reads in segmentation files for each cell type
        if args.verbose:
            sts.stderr.write(f"Reading {file}...\n")
        celltype = file.split('_')[0]
        segmentation = pd.read_table(file, names=['chrom', 'start', 'end', 'state'])
        dummy_state = segmentation.iloc[0]['state']  # Should be the last state
        
        # Each chromosome gives back a list of gene assignments
        mixture_assts = {chrom: [] for chrom in chroms}
        pos_mixture = []
        for idx, row in segmentation.iterrows():
            chrom, start_pos, end_pos, state = row
            if state == dummy_state:  # A dummy state means the last gene ended, on to a new gene
                if not pos_mixture == []:
                    mixture_assts[chrom].append(max(set(pos_mixture), key=pos_mixture.count))
                pos_mixture = []
            else:
                # The state is given as something like "E1" or "E22", so we remove the first character and cast as int
                # Then, we integer divide by the states per mixture component to get the component number
                mixture = (int(state[1:])-1) // args.states_per_mixture
                region_length = (int(end_pos) - int(start_pos)) // 200
                pos_mixture.extend([mixture] * region_length)
        per_celltype_asst_dict[celltype] = mixture_assts

    # per_celltype_asst_dict is a dict of dicts, of the format per_celltype_asst_dict[celltype][chrom] = [2, 5, 3, 1, 0, 0, ...]
    # We transform this into a matrix out_mat of shape (num_celltypes, num_genes)
    num_genes = np.sum([len(mixture_assts[k]) for k in mixture_assts])
    celltype_names = sorted(per_celltype_asst_dict.keys())

    out_mat = []
    for celltype in celltype_names:
        celltype_assts = []
        for chrom in chroms:
            celltype_assts.extend(per_celltype_asst_dict[celltype][chrom])
        out_mat.append(celltype_assts)

    # Read in the gene regions from the IDs file
    gene_ids = {}
    for chrom in chroms:
        fpath = os.path.join(args.ids_dir, f'{chrom}_ID.bed.gz')
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"{fpath} does not exist. Are you sure you provided the right ids_dir?")
        gene_ids[chrom] = pd.read_table(fpath, names=['chrom', 'start', 'end', 'name', 'score', 'strand'])
        if len(gene_ids[chrom]) != len(mixture_assts[chrom]):
            raise ValueError(f"Found a different number of genes in ID files vs segmentations in {chrom=}")

    np.save(os.path.join(args.out_dir, 'celltype_gene_assts.npy'), np.array(out_mat))
    np.savetxt(os.path.join(args.out_dir, 'celltype_gene_assts.txt'), np.array(out_mat))

    ## Also write this to a pandas table using the gene ID files
    for chrom in chroms:
        for cc, celltype in enumerate(celltype_names):
            gene_ids[chrom][celltype] = per_celltype_asst_dict[celltype][chrom]

    asst_table = pd.concat([gene_ids[chrom] for chrom in chroms], ignore_index=True)
    asst_table.to_csv(os.path.join(args.out_dir, "chromgene_assignments.tsv.gz", sep="\t", compression="gzip", index=False))


if __name__ == '__main__':
    main()
