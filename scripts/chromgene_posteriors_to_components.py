#!/usr/bin/python3.8 -tt

import sys
import gzip
import argparse
import os
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm


def get_component_assts_from_posteriors(file_path, states_per_component):
    """
    Return a numpy array of argmax posterior probability assignments
    """
    # size: (num_pos, tot_num_states + 1)
    # The "+ 1" is for the dummy state
    posteriors_arr = np.loadtxt(file_path, skiprows=2)

    dummy_indicator = posteriors_arr[:, -1]
    per_component_posteriors = posteriors_arr[:, :-1].reshape(len(posteriors_arr), -1, states_per_component).sum(axis=-1)
    per_component_assts = np.argmax(per_component_posteriors, axis=1)

    assert set(dummy_indicator) in [{1.}, {0., 1.}], f"Found invalid values in last column: {set(dummy_indicator)}"
    dummy_indicator = dummy_indicator.astype(bool)
    assert set(per_component_assts[dummy_indicator]) == {0}  # argmax will return 0 if the row is all 0s

    per_gene_assts = []
    look_at_next_asst_val = False
    for dummy_val, asst_val in zip(dummy_indicator, per_component_assts):
        if dummy_val == 1:
            look_at_next_asst_val = True
        elif look_at_next_asst_val:
            per_gene_assts.append(asst_val)
            look_at_next_asst_val = False

    assert sum(dummy_indicator) == len(per_gene_assts) + 1, f"Found an unexpected number of dummy states {sum(dummy_indicator)} and genes {len(per_gene_assts)}"

    return np.array(per_gene_assts)


def main():
    '''
    This script takes posterior probability files ChromHMM output and generates a matrix of ChromGene assignments
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--posteriors_dir', '--posteriors-dir', help='dir with posterior prob files', default='.', required=True)
    parser.add_argument('--ids_dir', '--ids-dir', help='directory with ID files', default='.', required=True)
    parser.add_argument('--states_per_component', '--states-per-component', help='number of states per component', type=int, default=3)
    parser.add_argument('--chroms', help='chromosomes to include in analysis', nargs='+', default=list(range(1, 23)) + ['X'])
    parser.add_argument('--out_dir', '--out-dir', help='directory in which to save data', default='.')
    parser.add_argument('--quiet', action='store_true', help='suppress progress bar')
    parser.add_argument(
        '--use_default_name_mapping', '--use-default-name-mapping', action='store_true', 
        help='use original ChromGene component to annotation mapping. Should only be used if using the original model provided by authors'
    )

    args = parser.parse_args()

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    chroms = []
    for chrom in args.chroms:
        if str(chrom).startswith('chr'):
            chroms.append(chrom)
        else:
            chroms.append(f'chr{chrom}')

    per_celltype_and_chrom_asst_dict = {}
    file_paths = glob(os.path.join(args.posteriors_dir, "*_posterior.txt*"))
    if len(file_paths) == 0 and "POSTERIOR" in os.listdir(args.posteriors_dir):
        file_paths = glob(os.path.join(args.posteriors_dir, "POSTERIOR", "*_posterior.txt*"))

    for file_path in tqdm(file_paths, desc="Processing posterior files", disable=args.quiet):
        celltype, _, chrom = os.path.basename(file_path).split('_')[:3]
        if not chrom in chroms:
            continue

        component_assts = get_component_assts_from_posteriors(file_path, args.states_per_component)
        per_celltype_and_chrom_asst_dict[(celltype, chrom)] = component_assts


    # per_celltype_and_chrom_asst_dict is a dict with values such as [2, 5, 3, 1, 0, 0, ...]
    # We transform this into a matrix out_mat of shape (num_celltypes, num_genes)
    celltype_names = sorted(list(set([key[0] for key in per_celltype_and_chrom_asst_dict.keys()])))

    out_mat = []
    for celltype in celltype_names:
        celltype_assts = []
        for chrom in chroms:
            celltype_assts.extend(per_celltype_and_chrom_asst_dict[(celltype, chrom)])
        out_mat.append(celltype_assts)

    out_mat = np.array(out_mat).astype(int)

    # Read in the gene regions from the IDs file
    gene_ids = {}
    for chrom in chroms:
        fpath = os.path.join(args.ids_dir, f'{chrom}_ID.bed.gz')
        gene_ids[chrom] = pd.read_table(fpath, names=['chrom', 'start', 'end', 'name', 'score', 'strand'])
        if len(gene_ids[chrom]) != len(per_celltype_and_chrom_asst_dict[(celltype_names[0], chrom)]):
            raise ValueError(f"Found a different number of genes in ID files vs posterior prob files in {chrom=}")

    np.save(os.path.join(args.out_dir, 'celltype_gene_assts.npy'), out_mat.transpose())
    np.savetxt(os.path.join(args.out_dir, 'celltype_gene_assts.txt'), out_mat.transpose(), fmt='%i', delimiter="\t")

    ## Also write this to a pandas table using the gene ID files
    for chrom in chroms:
        for cc, celltype in enumerate(celltype_names):
            gene_ids[chrom][celltype] = per_celltype_and_chrom_asst_dict[(celltype, chrom)]

    asst_table = pd.concat([gene_ids[chrom] for chrom in chroms], ignore_index=True)
    asst_table.to_csv(os.path.join(args.out_dir, "chromgene_assignments.tsv.gz"), sep="\t", compression="gzip", index=False)

    if args.use_default_name_mapping:
        mapping = {
            0: 'strong_trans_enh', 
            1: 'strong_trans', 
            2: 'trans_enh', 
            3: 'trans_cons', 
            4: 'trans_K36me3', 
            5: 'trans_K79me2', 
            6: 'weak_trans_enh', 
            7: 'znf',
            8: 'poised', 
            9: 'bivalent',
            10: 'low',
            11: 'quiescent',
        }

        for celltype in celltype_names:
            asst_table[celltype] = asst_table[celltype].apply(lambda val: mapping[val])

        asst_table.to_csv(os.path.join(args.out_dir, "chromgene_assignments_named.tsv.gz"), sep="\t", compression="gzip", index=False)

if __name__ == '__main__':
    main()
