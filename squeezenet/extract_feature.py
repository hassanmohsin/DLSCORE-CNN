# Script for extracting features from pdb-bind complexes
# Mahmudulla Hassan
# Last modified: 09/10/2018

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 
warnings.filterwarnings("ignore", category=FutureWarning) 

import htmd.ui as ht
import htmd.molecule.voxeldescriptors as vd
import os, sys
import pandas as pd
import numpy as np
from itertools import permutations

pdbbind_dir = "../dataset/refined-set-2016"

def get_pdb_feature(pdbid, voxel_size=(24, 24, 24), augmented=False, y_value=False):
    """ Returns voxel features for a pdb complex """

    def get_prop(mol, left_most_point):
        """ Returns atom occupancies """
        # Get the channels
        channels = vd._getAtomtypePropertiesPDBQT(mol)
        sigmas = vd._getRadii(mol)
        channels = sigmas[:, np.newaxis] * channels.astype(float)
        
        # Choose the grid centers
        centers = vd._getGridCenters(llc=left_most_point, N=voxel_size, resolution=1)
        centers = centers.reshape(np.prod(voxel_size), 3)
        
        # Extract the features and return
        features = vd._getOccupancyC(mol.coords[:, :, mol.frame], centers, channels)
        return features.reshape(*voxel_size, -1)
    
    def _rotate_sample(sample):
        output = np.zeros((24,)+sample.shape) #24 possible rotation
        counter = 0
        axes = [0, 1, 2]
        rotation_plane = permutations(axes, 2)
        rotated_sample = sample
        
        for plane in rotation_plane:
            #for angle in [0, 90, 180, 270]:
            output[counter] = rotated_sample
            counter = counter + 1
            for _ in range(3):
                rotated_sample = np.rot90(rotated_sample, axes=plane) #interpolation.rotate(input=sample, angle=angle, axes=plane, reshape=False)
                output[counter] = rotated_sample
                counter = counter + 1

        return output
    
    def get_augmented_data(x):
        aug_count = 24 # 24 possible rotation
        aug_data_x = np.zeros((x.shape[0]*aug_count,) + x.shape[1:])
        
        for i in range(x.shape[0]):
            aug_x = _rotate_sample(x[i])            
            aug_data_x[i*aug_count:i*aug_count+aug_count] = aug_x
            
        return aug_data_x
    
    # Find the files
    protein_file = os.path.join(pdbbind_dir, pdbid, pdbid + "_protein.pdbqt")
    ligand_file = os.path.join(pdbbind_dir, pdbid, pdbid + "_ligand.pdbqt")
    
    if not os.path.isfile(protein_file) or not os.path.isfile(ligand_file):
      raise FileNotFoundError("Protein or ligand file not found")
    
    # Generate the HTMD Molecule objects
    protein_mol = ht.Molecule(protein_file)
    ligand_mol = ht.Molecule(ligand_file)
    
    # Find the left most point. Half of the voxel's length is subtracted from the center of the ligand
    left_most_point = list(np.mean(ligand_mol.coords.reshape(-1, 3), axis=0) - 12.0)    
    
    # Get the features for both the protein and the ligand. Return those after concatenation.
    protein_featuers = get_prop(protein_mol, left_most_point)
    ligand_features = get_prop(ligand_mol, left_most_point)
    
    feature = np.concatenate((protein_featuers, ligand_features), axis=3)
    
    if not y_value:
        return feature if not augmented else get_augmented_data(feature.reshape(1, *feature.shape))
        
    # Read the affinity value
    affinity_df = pd.read_csv("PDBbind_refined16.txt", sep='\t', header=None, index_col=0)
    if pdbid in affinity_df.index:
        y = affinity_df.loc[pdbid].values[0] 
    else:
        raise ValueError("Invalid pdbid")
    
    return [feature, y] if not augmented else [get_augmented_data(feature.reshape(1, *feature.shape)), np.ones(24)*y]