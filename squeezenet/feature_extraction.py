# Script for extracting features from pdb-bind complexes
# Mahmudulla Hassan
# Last modified: 04/16/2019

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 
warnings.filterwarnings("ignore", category=FutureWarning) 

import htmd.ui as ht
import htmd.molecule.voxeldescriptors as vd
from htmd.molecule import vdw
import os, sys
import pandas as pd
import numpy as np
from itertools import permutations
import h5py
from oddt import toolkit
from oddt import datasets
from tqdm import *
import pickle
import dask.array as da
import json

# Directory paths
data_dir = "../dataset"
pdbbind_dir = os.path.join(data_dir, "refined-set-2016")

# from /home/mhassan/anaconda3/envs/dl/lib/python3.6/site-packages/htmd/molecule/voxeldescriptors.py
def _getRadii(mol):
    """ Gets vdW radius for each elem in mol.element. Source VMD.

    Parameters
    ----------
    mol :
        A Molecule object. Needs to be read from Autodock 4 .pdbqt format

    Returns
    -------
    radii : np.ndarray
        vdW radius for each element in mol.
    """

    mappings = {  # Mapping pdbqt representation to element.
        'HD': 'H',
        'HS': 'H',
        'A': 'C',
        'NA': 'N',
        'NS': 'N',
        'OA': 'O',
        'OS': 'O',
        'MG': 'Mg',
        'SA': 'S',
        'CL': 'Cl',
        'CA': 'Ca',
        'MN': 'Mn',
        'FE': 'Fe',
        'ZN': 'Zn',
        'BR': 'Br',
        'CS': 'Cs' # Extra added
    }
    atoms = ['H', 'C', 'N', 'O', 'F', 'Mg', 'P', 'S', 'Cl', 'Ca', 'Fe', 'Zn', 'Br', 'I']
    atoms.extend(['Co', 'Hg', 'Ni', 'Sr', 'Mn', 'K', 'Se', 'Cu', 'Cd', 'Li', 'Na']) # Extra added
    for el in atoms:
        mappings[el] = el

    res = np.zeros(mol.numAtoms)
    for a in range(mol.numAtoms):
        elem = mol.element[a]

        if elem not in mappings:
            raise ValueError('PDBQT element {} does not exist in mappings.'.format(elem))
        elem = mappings[elem]
        if elem in vdw.radiidict:
            rad = vdw.radiusByElement(elem)
        else:
            print('Unknown element -', mol.element[a], '- at atom index ', a)
            rad = 1.5
        res[a] = rad
    return res


def get_pdb_feature(pdbid, voxel_size=(24, 24, 24), augmented=False, y_value=False):
    """ Returns voxel features for a pdb complex """

    def get_prop(mol, left_most_point):
        """ Returns atom occupancies """
        # Get the channels
        channels = vd._getAtomtypePropertiesPDBQT(mol)
        sigmas = _getRadii(mol)
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
    if not os.path.isfile(protein_file): raise FileNotFoundError("{} NOT FOUND.".format(protein_file))
    ligand_file = os.path.join(pdbbind_dir, pdbid, pdbid + "_ligand.pdbqt")
    if not os.path.isfile(ligand_file): raise FileNotFoundError("{} NOT FOUND.".format(ligand_file))
    
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
    else:
        y = None
        affinity_df = pd.read_csv("PDBbind_refined16.txt", sep='\t', header=None, index_col=0)
        if pdbid in affinity_df.index:
            y = affinity_df.loc[pdbid].values[0] 
        else:
            raise ValueError("Invalid pdbid")
        return [feature, y] if not augmented else [get_augmented_data(feature.reshape(1, *feature.shape)), np.ones(24)*y]

    
def get_pdb_features(ids, sets, dirname):
    """ Returns features for given pdb ids"""
    pdb_ids = []
    labels = {}
    counter = 0
    pdb_map = {}
    
    for pdbid in tqdm(ids):
        try:
            [features, y_values] = get_pdb_feature(pdbid, y_value=True, augmented=True)
            for x, y in zip(features, y_values):
                np.save(os.path.join(dirname, "id_"+str(counter)), x.astype(np.float32))
                labels["id_"+str(counter)] = y
                pdb_map[pdbid] = counter
                counter = counter + 1
        except Exception as e:
            print("ERROR in ", pdbid , " ", str(e))
            continue

        pdb_ids.append(pdbid)
        #x.append(features)
        #y.append(y_values)
    
    label = "test" if sets == 'core' else "train"
    label_file = os.path.join(dirname, "labels.json")
    partition_file= os.path.join(dirname, "partition.json")
    pdbmap_file = os.path.join(dirname, "pdbmap.json")
    
    partition = {label: ['id_'+str(i) for i in range(counter)]}

    with open(label_file, 'w') as fp:
        json.dump(labels, fp)
    with open(partition_file, 'w') as fp:
        json.dump(partition, fp)
    with open(pdbmap_file, 'w') as fp:
        json.dump(pdb_map, fp)
    
    # save the successful pdb ids
    with open(os.path.join(dirname, 'successful_' + sets + '_pdb_ids.pickle'), 'wb') as f:
        pickle.dump(pdb_ids, f)

    
    
def to_hdf5(data):
    # Get the features
    [test_x, test_y, train_x, train_y] = data
    print("Shapes in the train and test set: ", train_x.shape, test_x.shape)
    print("Saving the data in data.h5")
    h5f = h5py.File(os.path.join(data_dir, "data.h5"), 'w')
    h5f.create_dataset('train_x', data=train_x)
    h5f.create_dataset('train_y', data=train_y)
    h5f.create_dataset('test_x', data=test_x)
    h5f.create_dataset('test_y', data=test_y)
    h5f.close()   
    
    
def main():
    """ Returns features for all the complexes in the dataset. """ 
    # List ids of the core set and the refined set
    core_ids = open('../core_ids.txt', 'r').readlines()[0].split(',')
    refined_ids = open('../refined_ids.txt', 'r').readlines()[0].split(',')
    
    # Get the features 
    #print("Extracting features for the core set")
    #core_x, core_y = get_pdb_features(core_ids, sets="core")
    #print("Extracting features for the refined set")
    #refined_x, refined_y = get_pdb_features(refined_ids, sets='refined')    
    
    #return core_x, core_y, refined_x, refined_y
    
    # Write the features to the directory
    train_dir = os.path.join(data_dir, 'training_data')
    if not os.path.isdir(train_dir): os.makedirs(train_dir)
    test_dir = os.path.join(data_dir, 'test_data')
    if not os.path.isdir(test_dir): os.makedirs(test_dir)
    
    get_pdb_features(core_ids, 'core', test_dir)
    get_pdb_features(refined_ids, 'refined', train_dir)


if __name__=="__main__": main()
