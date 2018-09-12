# Predicting protein-ligand binding affinities using Convolutional Neural Networks (CNN)
This is an implementation of the CNN network architecture described in the following paper 

**KDEEP: Protein–Ligand Absolute Binding Affinity Prediction via 3D-Convolutional Neural Networks** <br>
José Jiménez , Miha Škalič , Gerard Martínez-Rosell , and Gianni De Fabritiis <br>
DOI: 10.1021/acs.jcim.7b00650 <br>

### Requirements
  * Tensorflow : `pip install tensorflow-gpu` or `pip install tensorflow`
  * Keras: `pip install keras`
  * Scikit-learn: `pip install -U scikit-learn`
  * oddt: `conda install -c oddt oddt`
  * tqdm: `pip install tqdm`
  * htmd: 
  ```
  conda config --add channels acellera 
  conda config --add channels psi4 
  conda install htmd 
  ```
