#!/usr/bin/python3.8 -tt

import sys
import gzip
import argparse
import os
import numpy as np
import pandas as pd


def main():
	'''
	This script takes segmentation files ChromHMM output and generates a matrix of ChromGene assignments
	'''
	parser = argparse.ArgumentParser()
	parser.add_argument('--segmentation_dir', help='dir with segmentation files', default='.')
	parser.add_argument('--states_per_mixture', help='number of states per mixture', type=int, default=4)
	parser.add_argument('--out_dir', help='directory in which to save data', default='.')

	args = parser.parse_args()

	if args.out_dir[-1] != '/':
		args.out_dir += '/'
	if args.segmentation_dir[-1] != '/':
		args.segmentation_dir += '/'
	if not os.path.exists(args.out_dir):
		os.makedirs(args.out_dir)

	cell_dict = {}
	for file in os.listdir(args.segmentation_dir):
		# Reads in segmentation files for each cell type
		if not (file.endswith('_segments.bed.gz') or file.endswith('_segments.bed')):
			continue
		print("Reading "+file)
		celltype = file.split('_')[0]
		segmentation = pd.read_table(args.segmentation_dir + file, names=['chrom', 'start', 'end', 'state'])
		dummy_state = segmentation.iloc[0]['state']  # Should be the last state
		# Each chromosome gives back a list of gene assignments
		mixture_assts = {'chr'+str(k):[] for k in list(range(1,23)) + ['X']}
		pos_mixture = []
		for idx, row in segmentation.iterrows():
			chrom, start_pos, end_pos, state = row
			if state == dummy_state:  # A dummy state means the last gene ended, on to a new gene
				if not pos_mixture == []:
					mixture_assts[chrom].append(max(set(pos_mixture), key=pos_mixture.count))
				pos_mixture = []
			else:
				mixture = (int(state[1:])-1) // args.states_per_mixture
				region_length = (int(end_pos) - int(start_pos)) // 200
				pos_mixture.extend([mixture] * region_length)
		cell_dict[celltype] = mixture_assts

	num_genes = np.sum([len(mixture_assts[k]) for k in mixture_assts])
	out_mat = []
	for cell in sorted(cell_dict):
		cell_assts = []
		for chrom in ['chr'+str(k) for k in list(range(1,23)) + ['X']]:
			cell_assts.extend(cell_dict[cell][chrom])
		out_mat.append(cell_assts)

	np.save(args.out_dir+'celltype_gene_assts.npy', np.array(out_mat))
	np.savetxt(args.out_dir+'celltype_gene_assts.txt', np.array(out_mat))

if __name__ == '__main__':
	main()
