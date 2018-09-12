import numpy as np
import glob
import csv
import pickle
from tqdm import *
import os
from scipy import spatial
import pybel
from oddt import toolkit
from oddt import datasets
# ODDT documentation: https://pythonhosted.org/oddt/
import h5py
import sys
sys.path.append("scripts")
from elements import ELEMENTS


# Directory paths
data_dir = "data"
pdbbind_dir = "/home/PDBbind/pdbbind_2016/refined-set-2016"
binding_pocket_dir = "/home/PDBbind/pdbbind_2016/binding_pockets_refined"
pdbbind_dataset = datasets.pdbbind(home=pdbbind_dir, default_set='refined', version=2016)


# Overriding the datasets._pdbbind_id class to get the different pocket files than the original one
class PDB(datasets._pdbbind_id):
    def __init__(self, home, pocket_dir, pdbid, opt=None):
        self.pocket_dir = pocket_dir
        datasets._pdbbind_id.__init__(self, home, pdbid, opt=None)
    
    @property
    def pocket(self):
        f = os.path.join(self.pocket_dir, '%s_complex2.pdb' % self.id)
        if os.path.isfile(f):
            pocket = next(toolkit.readfile('pdb', f, lazy=True, opt=self.opt))
            if pocket is not None:
                pocket.protein = True
            return pocket
        else:
            return None        


# In[4]:


def _reshape(x):
    """
    Method for reshaping a sample to the desired one.
    :param x
    """
    new_shape = [24, 24, 24, 16]
    dim_diff = np.array([i-j for i, j in zip(new_shape, x.shape)])
    pad_dim = np.round(dim_diff / 2).astype(int)
    x = np.pad(x, [(pad_dim[0], dim_diff[0]-pad_dim[0]),
                   (pad_dim[1], dim_diff[1]-pad_dim[1]),
                   (pad_dim[2], dim_diff[2]-pad_dim[2]),
                   (0, 0)],
               'constant')
    
    return x
    #return x[13:37, 13:37, 13:37, :]
    #return x[5:45, 5:45, 5:45, :]


# In[5]:


# Van daar Waals radii for all the elements
element_radii = {'Ac': 2.0, 'Ag': 1.72, 'Al': 2.0, 'Am': 2.0, 'Ar': 1.88, 'As': 1.85, 'At': 2.0, 'Au': 1.66,
                 'B': 2.0, 'Ba': 2.0, 'Be': 2.0, 'Bh': 2.0, 'Bi': 2.0, 'Bk': 2.0, 'Br': 1.85, 'C': 1.7, 'Ca': 1.37,
                 'Cd': 1.58, 'Ce': 2.0, 'Cf': 2.0, 'Cl': 2.27, 'Cm': 2.0, 'Co': 2.0, 'Cr': 2.0, 'Cs': 2.1, 'Cu': 1.4,
                 'Db': 2.0, 'Ds': 2.0, 'Dy': 2.0, 'Er': 2.0, 'Es': 2.0, 'Eu': 2.0, 'F': 1.47, 'Fe': 2.0, 'Fm': 2.0,
                 'Fr': 2.0, 'Ga': 1.07, 'Gd': 2.0, 'Ge': 2.0, 'H': 1.2, 'He': 1.4, 'Hf': 2.0, 'Hg': 1.55, 'Ho': 2.0,
                 'Hs': 2.0, 'I': 1.98, 'In': 1.93, 'Ir': 2.0, 'K': 1.76, 'Kr': 2.02, 'La': 2.0, 'Li': 1.82, 'Lr': 2.0,
                 'Lu': 2.0, 'Md': 2.0, 'Mg': 1.18, 'Mn': 2.0, 'Mo': 2.0, 'Mt': 2.0, 'N': 1.55, 'Na': 1.36, 'Nb': 2.0,
                 'Nd': 2.0, 'Ne': 1.54, 'Ni': 1.63, 'No': 2.0, 'Np': 2.0, 'O': 1.52, 'Os': 2.0, 'P': 1.8, 'Pa': 2.0,
                 'Pb': 2.02, 'Pd': 1.63, 'Pm': 2.0, 'Po': 2.0, 'Pr': 2.0, 'Pt': 1.72, 'Pu': 2.0, 'Ra': 2.0, 'Rb': 2.0,
                 'Re': 2.0, 'Rf': 2.0, 'Rg': 2.0, 'Rh': 2.0, 'Rn': 2.0, 'Ru': 2.0, 'S': 1.8, 'Sb': 2.0, 'Sc': 2.0,
                 'Se': 1.9, 'Sg': 2.0, 'Si': 2.1, 'Sm': 2.0, 'Sn': 2.17, 'Sr': 2.0, 'Ta': 2.0, 'Tb': 2.0, 'Tc': 2.0,
                 'Te': 2.06, 'Th': 2.0, 'Ti': 2.0, 'Tl': 1.96, 'Tm': 2.0, 'U': 1.86, 'V': 2.0, 'W': 2.0, 'X': 1.5,
                 'Xe': 2.16, 'Y': 2.0, 'Yb': 2.0, 'Zn': 1.39, 'Zr': 2.0}


# ## Generating Nearest Neighbor features

# In[44]:


# Get voxel features (Nearest neighbor feature)

def get_voxel_features(pdbid, pdbbind_set='refined'):
    if pdbbind_set == 'refined':
        # Don't use the core set
        if pdbid in pdbbind_dataset.sets['core'].keys():
            #print("ERROR: PDBID {} IS IN CORE SET.".format(pdbid))
            return None
    
    # Read the binding pocket file
    pdb_object = PDB(home=pdbbind_dir, pocket_dir=binding_pocket_dir, pdbid=pdbid)
    binding_pocket = pdb_object.pocket    
    
    # If the binding pocket file has any error, return None
    if binding_pocket == None:
        #print("ERROR: INVALID BINDING POCKET FOR PDBID {}.".format(pdbid))
        return None
    
    # Get the filters for protein and ligands
    filter_proteins = [atom.residue.name != 'HOH' for atom in binding_pocket.atoms]
    filter_ligands = [atom.residue.name == 'HOH' for atom in binding_pocket.atoms]
    
    # Get the properties
    is_hydrophobic = binding_pocket.atom_dict['ishydrophobe'].reshape((-1, 1))
    is_aromatic = binding_pocket.atom_dict['isaromatic'].reshape((-1, 1))
    is_hbond_acceptor = binding_pocket.atom_dict['isacceptor'].reshape((-1, 1))
    is_hbond_donor = binding_pocket.atom_dict['isdonor'].reshape((-1, 1))
    charges = binding_pocket.atom_dict['charge']
    is_positive = (charges < 0.0).reshape((-1, 1))
    is_negative = (charges >= 0.0).reshape((-1, 1))
    is_metal = binding_pocket.atom_dict['ismetal'].reshape((-1, 1))
    is_halogen = binding_pocket.atom_dict['ishalogen'].reshape((-1, 1))
    atom_coords = binding_pocket.atom_dict['coords']
    properties = np.concatenate((is_hydrophobic,
                                 is_aromatic, 
                                 is_hbond_acceptor, 
                                 is_hbond_donor, 
                                 is_positive, 
                                 is_negative, 
                                 is_metal, 
                                 is_halogen), axis=1)
    
    
    # Now get the Van Dar Wals redii for each of the atoms
    vdw_radii = np.array([element_radii[ELEMENTS[a.atomicnum].symbol] for a in binding_pocket.atoms], dtype=np.float32)
        
    # Multiply the vdw radii with the properties. False's will be zeros and True's will be the vdw radii
    properties = vdw_radii[:, np.newaxis] * properties
    
    # Get the features for proteins and ligands
    features = np.zeros((len(binding_pocket.atoms), 16))
    features[filter_proteins, :8] = properties[filter_proteins]
    features[filter_ligands, 8:] = properties[filter_ligands]
    
    # Get the bounding box for the molecule
    max_coord = np.max(atom_coords, axis=0) # np.squeeze?
    min_coord = np.min(atom_coords, axis=0) # np.squeeze?

    # Calculate the number of voxels required
    voxel_side = 2
    N = np.ceil((max_coord - min_coord) / voxel_side).astype(int) + 1

    # Get the centers of each descriptors
    xrange = [min_coord[0] + voxel_side * x for x in range(0, N[0])]
    yrange = [min_coord[1] + voxel_side * x for x in range(0, N[1])]
    zrange = [min_coord[2] + voxel_side * x for x in range(0, N[2])]
    centers = np.zeros((N[0], N[1], N[2], 3))

    for i, x in enumerate(xrange):
        for j, y in enumerate(yrange):
            for k, z in enumerate(zrange):
                centers[i, j, k, :] = np.array([x, y, z])

    centers = centers.reshape((-1, 3))
    
    voxel_features = np.zeros((len(centers), features.shape[1]), dtype=np.float32)
    for i in range(len(binding_pocket.atoms)):
        # Get the coordinates of the current atom
        atom_coordinate = atom_coords[i]
        
        # Get the closest voxel
        c_voxel_id = spatial.distance.cdist(atom_coordinate.reshape((-1, 3)), centers).argmin()
        c_voxel = centers[c_voxel_id] # nearest voxel
        
        # Calculate the potential
        voxel_distance = np.linalg.norm(atom_coordinate - c_voxel)
        x = features[i] / voxel_distance
        n = 1.0 - np.exp(-np.power(x, 12))
        voxel_features[c_voxel_id] = n #features[i]
        
    voxel_features = voxel_features.reshape((N[0], N[1], N[2], -1))
    
    return _reshape(voxel_features)


# In[52]:


pdb_ids = []
pdb_features = []

# Get the features for all the pdbbind refined complexes
for pdbid in tqdm_notebook(pdbbind_dataset.sets['refined'].keys()):
    feat = get_voxel_features(pdbid=pdbid, pdbbind_set='refined')
    if feat == None: continue    

    # Save features
    pdb_ids.append(pdbid)
    pdb_features.append(feat)


# In[56]:


# Convert the list of features as numpy array
data_x = np.array(pdb_features, dtype=np.float32)
data_y = np.array([pdbbind_dataset.sets['refined'][_id] for _id in pdb_ids], dtype=np.float32)

print(data_x.shape, data_y.shape)


# In[85]:


# Split into train, test and validation
from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, test_size=0.2, random_state=1)
test_x, valid_x, test_y, valid_y = train_test_split(test_x, test_y, test_size=0.5, random_state=1)

print(train_x.shape, valid_x.shape, test_x.shape)


# In[86]:


# Save the data
h5f = h5py.File(os.path.join(data_dir, "data_nearest_neighbor.h5"), 'w')
h5f.create_dataset('train_x', data=train_x)
h5f.create_dataset('train_y', data=train_y)
h5f.create_dataset('valid_x', data=valid_x)
h5f.create_dataset('valid_y', data=valid_y)
h5f.create_dataset('test_x', data=test_x)
h5f.create_dataset('test_y', data=test_y)
h5f.close()


# ## Genrating distributed features

# In[7]:


# Get voxel features (distributed features)

def get_voxel_features2(pdbid, pdbbind_set='refined'):
    if pdbbind_set == 'refined':
        # Don't use the core set
        if pdbid in pdbbind_dataset.sets['core'].keys():
            #print("ERROR: PDBID {} IS IN CORE SET.".format(pdbid))
            return None
    
    # Read the binding pocket file
    pdb_object = PDB(home=pdbbind_dir, pocket_dir=binding_pocket_dir, pdbid=pdbid)
    binding_pocket = pdb_object.pocket  
    
    # If the binding pocket file has any error, return None
    if binding_pocket == None:
        #print("ERROR: INVALID BINDING POCKET FOR PDBID {}.".format(pdbid))
        return None
    
    # Get the filters for protein and ligands
    filter_proteins = [atom.residue.name != 'HOH' for atom in binding_pocket.atoms]
    filter_ligands = [atom.residue.name == 'HOH' for atom in binding_pocket.atoms]
    
    # Get the properties
    is_hydrophobic = binding_pocket.atom_dict['ishydrophobe'].reshape((-1, 1))
    is_aromatic = binding_pocket.atom_dict['isaromatic'].reshape((-1, 1))
    is_hbond_acceptor = binding_pocket.atom_dict['isacceptor'].reshape((-1, 1))
    is_hbond_donor = binding_pocket.atom_dict['isdonor'].reshape((-1, 1))
    charges = binding_pocket.atom_dict['charge']
    is_positive = (charges < 0.0).reshape((-1, 1))
    is_negative = (charges >= 0.0).reshape((-1, 1))
    is_metal = binding_pocket.atom_dict['ismetal'].reshape((-1, 1))
    is_halogen = binding_pocket.atom_dict['ishalogen'].reshape((-1, 1))
    atom_coords = binding_pocket.atom_dict['coords']
    properties = np.concatenate((is_hydrophobic,
                                 is_aromatic, 
                                 is_hbond_acceptor, 
                                 is_hbond_donor, 
                                 is_positive, 
                                 is_negative, 
                                 is_metal, 
                                 is_halogen), axis=1)
    
    
    # Now get the Van Dar Wals redii for each of the atoms
    vdw_radii = np.array([element_radii[ELEMENTS[a.atomicnum].symbol] for a in binding_pocket.atoms], dtype=np.float32)
        
    # Multiply the vdw radii with the properties. False's will be zeros and True's will be the vdw radii
    properties = vdw_radii[:, np.newaxis] * properties
    
    # Get the features for proteins and ligands
    features = np.zeros((len(binding_pocket.atoms), 16))
    features[filter_proteins, :8] = properties[filter_proteins]
    features[filter_ligands, 8:] = properties[filter_ligands]
    
    # Get the bounding box for the molecule
    max_coord = np.max(atom_coords, axis=0) # np.squeeze?
    min_coord = np.min(atom_coords, axis=0) # np.squeeze?

    # Calculate the number of voxels required
    voxel_side = 2
    N = np.ceil((max_coord - min_coord) / voxel_side).astype(int) + 1

    # Get the centers of each descriptors
    xrange = [min_coord[0] + voxel_side * x for x in range(0, N[0])]
    yrange = [min_coord[1] + voxel_side * x for x in range(0, N[1])]
    zrange = [min_coord[2] + voxel_side * x for x in range(0, N[2])]
    centers = np.zeros((N[0], N[1], N[2], 3))

    for i, x in enumerate(xrange):
        for j, y in enumerate(yrange):
            for k, z in enumerate(zrange):
                centers[i, j, k, :] = np.array([x, y, z])

    centers = centers.reshape((-1, 3))
    
    voxel_features = np.zeros((len(centers), features.shape[1]), dtype=np.float32)
    
    for i in range(len(binding_pocket.atoms)):
        # Get the coordinates of the current atom
        atom_coordinate = atom_coords[i]
        
        # Get the closest voxel and it's 26 neighbors
        voxel_distances = spatial.distance.cdist(atom_coordinate.reshape((-1, 3)), centers).reshape(-1)
        c_voxel_ids = voxel_distances.argsort()[:27]
        c_voxel_dist = np.sort(voxel_distances)[:27]
                
        # Calculate the potential
        x = features[i] / c_voxel_dist.reshape(-1)[:, np.newaxis]
        n = 1.0 - np.exp(-np.power(x, 12))
        
        # Get the maximum and assign
        max_feat = np.maximum(voxel_features[c_voxel_ids], n)
        
        voxel_features[c_voxel_ids] = n #features[i]
        
    voxel_features = voxel_features.reshape((N[0], N[1], N[2], -1))
    
    return _reshape(voxel_features)


# In[8]:


pdb_ids = []
pdb_features = []

# Get the features for all the pdbbind refined complexes
for pdbid in tqdm_notebook(pdbbind_dataset.sets['refined'].keys()):
    feat = get_voxel_features2(pdbid=pdbid, pdbbind_set='refined')
    if feat == None: continue    

    # Save features
    pdb_ids.append(pdbid)
    pdb_features.append(feat)


# In[9]:


# Convert the list of features as numpy array
data_x = np.array(pdb_features, dtype=np.float32)
data_y = np.array([pdbbind_dataset.sets['refined'][_id] for _id in pdb_ids], dtype=np.float32)

print(data_x.shape, data_y.shape)


# In[10]:


# Split into train, test and validation
from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, test_size=0.2, random_state=1)
test_x, valid_x, test_y, valid_y = train_test_split(test_x, test_y, test_size=0.5, random_state=1)

print(train_x.shape, valid_x.shape, test_x.shape)


# In[11]:


# Save the data
h5f = h5py.File(os.path.join(data_dir, "data_distributed2.h5"), 'w')
h5f.create_dataset('train_x', data=train_x)
h5f.create_dataset('train_y', data=train_y)
h5f.create_dataset('valid_x', data=valid_x)
h5f.create_dataset('valid_y', data=valid_y)
h5f.create_dataset('test_x', data=test_x)
h5f.create_dataset('test_y', data=test_y)
h5f.close()


# ## Generating Canonically Oriented Nearest Neighbor Features

# In[14]:


from sklearn.decomposition import PCA


# In[15]:


import pybel


# In[16]:


error_count = 0


# In[17]:


def transform_coordinates(binding_pocket):
    # Get the filters for protein and ligands
    filter_proteins = [atom.residue.name != 'HOH' for atom in binding_pocket.atoms]
    filter_ligands = [atom.residue.name == 'HOH' for atom in binding_pocket.atoms]
    
    # Get the coordinates for proteins and ligands
    coords = binding_pocket.atom_dict['coords']
    protein_coords = coords[filter_proteins]
    ligand_coords = coords[filter_ligands]
    
    #Find the centroid of the ligand structure and move the points around that.
    l_centroid = np.mean(ligand_coords, axis=0)
    protein_coords = protein_coords - l_centroid
    ligand_coords = ligand_coords - l_centroid
    
    try:
        # Perform PCA on ligand points
        pca = PCA(n_components=3)
        pca.fit(ligand_coords)

        # x_axis is the first principal component
        x_axis = pca.components_[:, 0]

        # Get the centroid of the protein points
        p_centroid = np.mean(protein_coords, axis=0)

        # Projection of p_centroid vector on x_axis
        proj_x = np.matmul(np.transpose(x_axis), p_centroid)/np.matmul(np.transpose(x_axis), x_axis) * x_axis
        # Get y-axis
        y_axis = p_centroid - proj_x

        #Normalize
        y_axis = y_axis / np.linalg.norm(y_axis)

        # z_axis is perpendicular to both x_axis and y_axis. Not sure about the direction yet (+ or - ?)
        z_axis = np.cross(x_axis, y_axis)

        # Transformation matrix (as column vector and normalized)
        R = np.transpose(np.array([x_axis, y_axis, z_axis]))

        # Transform all protein and ligand points
        mol_coords = coords.reshape((-1, 3))
        transformed_coords = np.transpose(np.matmul(R, np.transpose(mol_coords))).reshape((-1, 3))

        # Save the coordinates back to the original binding pocket
        binding_pocket.coords = transformed_coords

        return binding_pocket
    except:
        global error_count 
        error_count = error_count + 1
        return binding_pocket


# In[118]:


# Get voxel features (Nearest neighbor feature)

def get_voxel_features3(pdbid, pdbbind_set='refined'):
    if pdbbind_set == 'refined':
        # Don't use the core set
        if pdbid in pdbbind_dataset.sets['core'].keys():
            #print("ERROR: PDBID {} IS IN CORE SET.".format(pdbid))
            return None
    
    # Read the binding pocket file
    pdb_object = PDB(home=pdbbind_dir, pocket_dir=binding_pocket_dir, pdbid=pdbid)
    
    # If the binding pocket file has any error, return None
    if pdb_object == None or pdb_object.pocket == None:
        #print("ERROR: INVALID BINDING POCKET FOR PDBID {}.".format(pdbid))
        return None
    
    # Get the Canonically transformed binding pocket
    binding_pocket = transform_coordinates(pdb_object.pocket)
    
    # Get the filters for protein and ligands
    filter_proteins = [atom.residue.name != 'HOH' for atom in binding_pocket.atoms]
    filter_ligands = [atom.residue.name == 'HOH' for atom in binding_pocket.atoms]
    
    # Get the properties
    is_hydrophobic = binding_pocket.atom_dict['ishydrophobe'].reshape((-1, 1))
    is_aromatic = binding_pocket.atom_dict['isaromatic'].reshape((-1, 1))
    is_hbond_acceptor = binding_pocket.atom_dict['isacceptor'].reshape((-1, 1))
    is_hbond_donor = binding_pocket.atom_dict['isdonor'].reshape((-1, 1))
    charges = binding_pocket.atom_dict['charge']
    is_positive = (charges < 0.0).reshape((-1, 1))
    is_negative = (charges >= 0.0).reshape((-1, 1))
    is_metal = binding_pocket.atom_dict['ismetal'].reshape((-1, 1))
    is_halogen = binding_pocket.atom_dict['ishalogen'].reshape((-1, 1))
    atom_coords = binding_pocket.atom_dict['coords']
    properties = np.concatenate((is_hydrophobic,
                                 is_aromatic, 
                                 is_hbond_acceptor, 
                                 is_hbond_donor, 
                                 is_positive, 
                                 is_negative, 
                                 is_metal, 
                                 is_halogen), axis=1)
    
    
    # Now get the Van Dar Wals redii for each of the atoms
    vdw_radii = np.array([element_radii[ELEMENTS[a.atomicnum].symbol] for a in binding_pocket.atoms], dtype=np.float32)
        
    # Multiply the vdw radii with the properties. False's will be zeros and True's will be the vdw radii
    properties = vdw_radii[:, np.newaxis] * properties
    
    # Get the features for proteins and ligands
    features = np.zeros((len(binding_pocket.atoms), 16))
    features[filter_proteins, :8] = properties[filter_proteins]
    features[filter_ligands, 8:] = properties[filter_ligands]
    
    # Get the bounding box for the molecule
    max_coord = np.max(atom_coords, axis=0) # np.squeeze?
    min_coord = np.min(atom_coords, axis=0) # np.squeeze?

    # Calculate the number of voxels required
    voxel_side = 2
    N = np.ceil((max_coord - min_coord) / voxel_side).astype(int) + 1

    # Get the centers of each descriptors
    xrange = [min_coord[0] + voxel_side * x for x in range(0, N[0])]
    yrange = [min_coord[1] + voxel_side * x for x in range(0, N[1])]
    zrange = [min_coord[2] + voxel_side * x for x in range(0, N[2])]
    centers = np.zeros((N[0], N[1], N[2], 3))

    for i, x in enumerate(xrange):
        for j, y in enumerate(yrange):
            for k, z in enumerate(zrange):
                centers[i, j, k, :] = np.array([x, y, z])

    centers = centers.reshape((-1, 3))
    
    voxel_features = np.zeros((len(centers), features.shape[1]), dtype=np.float32)
    for i in range(len(binding_pocket.atoms)):
        # Get the coordinates of the current atom
        atom_coordinate = atom_coords[i]
        
        # Get the closest voxel
        c_voxel_id = spatial.distance.cdist(atom_coordinate.reshape((-1, 3)), centers).argmin()
        c_voxel = centers[c_voxel_id] # nearest voxel
        
        # Calculate the potential
        voxel_distance = np.linalg.norm(atom_coordinate - c_voxel)
        x = features[i] / voxel_distance
        n = 1.0 - np.exp(-np.power(x, 12))
        voxel_features[c_voxel_id] = n #features[i]
        
    voxel_features = voxel_features.reshape((N[0], N[1], N[2], -1))
    
    return _reshape(voxel_features)


# It's not possible to do PCA on some of the pdb complexes. Because sometimes the number of ligand atoms are not enough to perform that operation.

# In[120]:


pdb_ids = []
pdb_features = []

# Get the features for all the pdbbind refined complexes
for pdbid in tqdm_notebook(pdbbind_dataset.sets['refined'].keys()):
    try:
        feat = get_voxel_features3(pdbid=pdbid, pdbbind_set='refined') # Get the features from Canonically oriented molecule
    except:
        continue
    if feat == None: continue    

    # Save features
    pdb_ids.append(pdbid)
    pdb_features.append(feat)


# In[121]:


# Convert the list of features as numpy array
data_x = np.array(pdb_features, dtype=np.float32)
data_y = np.array([pdbbind_dataset.sets['refined'][_id] for _id in pdb_ids], dtype=np.float32)

print(data_x.shape, data_y.shape)


# In[122]:


# Split into train, test and validation
from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, test_size=0.2, random_state=1)
test_x, valid_x, test_y, valid_y = train_test_split(test_x, test_y, test_size=0.5, random_state=1)

print(train_x.shape, valid_x.shape, test_x.shape)


# In[123]:


# Save the data
h5f = h5py.File(os.path.join(data_dir, "data_cano_nearest_neighbor.h5"), 'w')
h5f.create_dataset('train_x', data=train_x)
h5f.create_dataset('train_y', data=train_y)
h5f.create_dataset('valid_x', data=valid_x)
h5f.create_dataset('valid_y', data=valid_y)
h5f.create_dataset('test_x', data=test_x)
h5f.create_dataset('test_y', data=test_y)
h5f.close()


# ## Generating Canonically Oriented  Distributed Features

# In[18]:


# Get voxel features (distributed features)

def get_voxel_features4(pdbid, pdbbind_set='refined'):
    if pdbbind_set == 'refined':
        # Don't use the core set
        if pdbid in pdbbind_dataset.sets['core'].keys():
            #print("ERROR: PDBID {} IS IN CORE SET.".format(pdbid))
            return None
    
    # Read the binding pocket file
    pdb_object = PDB(home=pdbbind_dir, pocket_dir=binding_pocket_dir, pdbid=pdbid)
    
    # If the binding pocket file has any error, return None
    if pdb_object == None or pdb_object.pocket == None:
        #print("ERROR: INVALID BINDING POCKET FOR PDBID {}.".format(pdbid))
        return None
    
    # Get the Canonically transformed binding pocket
    binding_pocket = transform_coordinates(pdb_object.pocket)
    
    # Get the filters for protein and ligands
    filter_proteins = [atom.residue.name != 'HOH' for atom in binding_pocket.atoms]
    filter_ligands = [atom.residue.name == 'HOH' for atom in binding_pocket.atoms]
    
    # Get the properties
    is_hydrophobic = binding_pocket.atom_dict['ishydrophobe'].reshape((-1, 1))
    is_aromatic = binding_pocket.atom_dict['isaromatic'].reshape((-1, 1))
    is_hbond_acceptor = binding_pocket.atom_dict['isacceptor'].reshape((-1, 1))
    is_hbond_donor = binding_pocket.atom_dict['isdonor'].reshape((-1, 1))
    charges = binding_pocket.atom_dict['charge']
    is_positive = (charges < 0.0).reshape((-1, 1))
    is_negative = (charges >= 0.0).reshape((-1, 1))
    is_metal = binding_pocket.atom_dict['ismetal'].reshape((-1, 1))
    is_halogen = binding_pocket.atom_dict['ishalogen'].reshape((-1, 1))
    atom_coords = binding_pocket.atom_dict['coords']
    properties = np.concatenate((is_hydrophobic,
                                 is_aromatic, 
                                 is_hbond_acceptor, 
                                 is_hbond_donor, 
                                 is_positive, 
                                 is_negative, 
                                 is_metal, 
                                 is_halogen), axis=1)
    
    
    # Now get the Van Dar Wals redii for each of the atoms
    vdw_radii = np.array([element_radii[ELEMENTS[a.atomicnum].symbol] for a in binding_pocket.atoms], dtype=np.float32)
        
    # Multiply the vdw radii with the properties. False's will be zeros and True's will be the vdw radii
    properties = vdw_radii[:, np.newaxis] * properties
    
    # Get the features for proteins and ligands
    features = np.zeros((len(binding_pocket.atoms), 16))
    features[filter_proteins, :8] = properties[filter_proteins]
    features[filter_ligands, 8:] = properties[filter_ligands]
    
    # Get the bounding box for the molecule
    max_coord = np.max(atom_coords, axis=0) # np.squeeze?
    min_coord = np.min(atom_coords, axis=0) # np.squeeze?

    # Calculate the number of voxels required
    voxel_side = 2
    N = np.ceil((max_coord - min_coord) / voxel_side).astype(int) + 1

    # Get the centers of each descriptors
    xrange = [min_coord[0] + voxel_side * x for x in range(0, N[0])]
    yrange = [min_coord[1] + voxel_side * x for x in range(0, N[1])]
    zrange = [min_coord[2] + voxel_side * x for x in range(0, N[2])]
    centers = np.zeros((N[0], N[1], N[2], 3))

    for i, x in enumerate(xrange):
        for j, y in enumerate(yrange):
            for k, z in enumerate(zrange):
                centers[i, j, k, :] = np.array([x, y, z])

    centers = centers.reshape((-1, 3))
    
    voxel_features = np.zeros((len(centers), features.shape[1]), dtype=np.float32)
    
    for i in range(len(binding_pocket.atoms)):
        # Get the coordinates of the current atom
        atom_coordinate = atom_coords[i]
        
        # Get the closest voxel and it's 8 neighbors
        voxel_distances = spatial.distance.cdist(atom_coordinate.reshape((-1, 3)), centers).reshape(-1)
        c_voxel_ids = voxel_distances.argsort()[:27]
        c_voxel_dist = np.sort(voxel_distances)[:27]
                
        # Calculate the potential
        x = features[i] / c_voxel_dist.reshape(-1)[:, np.newaxis]
        n = 1.0 - np.exp(-np.power(x, 12))
        
        # Get the maximum and assign
        max_feat = np.maximum(voxel_features[c_voxel_ids], n)
        
        voxel_features[c_voxel_ids] = n #features[i]
        
    voxel_features = voxel_features.reshape((N[0], N[1], N[2], -1))
    
    return _reshape(voxel_features)


# In[19]:


pdb_ids = []
pdb_features = []

# Get the features for all the pdbbind refined complexes
for pdbid in tqdm_notebook(pdbbind_dataset.sets['refined'].keys()):
    try:
        feat = get_voxel_features4(pdbid=pdbid, pdbbind_set='refined') # Get the features from Canonically oriented molecule
    except:
        continue
    if feat == None: continue    

    # Save features
    pdb_ids.append(pdbid)
    pdb_features.append(feat)


# In[20]:


# Convert the list of features as numpy array
data_x = np.array(pdb_features, dtype=np.float32)
data_y = np.array([pdbbind_dataset.sets['refined'][_id] for _id in pdb_ids], dtype=np.float32)

print(data_x.shape, data_y.shape)


# In[21]:


# Split into train, test and validation
from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, test_size=0.2, random_state=1)
test_x, valid_x, test_y, valid_y = train_test_split(test_x, test_y, test_size=0.5, random_state=1)

print(train_x.shape, valid_x.shape, test_x.shape)


# In[22]:


# Save the data
h5f = h5py.File(os.path.join(data_dir, "data_cano_distributed2.h5"), 'w')
h5f.create_dataset('train_x', data=train_x)
h5f.create_dataset('train_y', data=train_y)
h5f.create_dataset('valid_x', data=valid_x)
h5f.create_dataset('valid_y', data=valid_y)
h5f.create_dataset('test_x', data=test_x)
h5f.create_dataset('test_y', data=test_y)
h5f.close()

