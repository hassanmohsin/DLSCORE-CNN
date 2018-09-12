
# coding: utf-8

# ## Training

# In[1]:


from keras.models import Sequential,Model
from keras.layers import Dropout, Flatten, Dense, Input, Add, merge, concatenate
from keras.layers.convolutional import Conv3D
from keras.layers.pooling import MaxPooling3D, GlobalAveragePooling3D, AveragePooling3D
from keras import metrics
from keras import optimizers
from keras.utils import plot_model
from keras import backend as K
from keras.utils.training_utils import multi_gpu_model
from keras.utils.data_utils import Sequence
from keras.callbacks import ModelCheckpoint
from keras.initializers import he_uniform
from keras.initializers import glorot_uniform

import numpy as np
import sys
import h5py
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

sys.path.append("models/")
sys.path.append("scripts/")


# In[2]:


import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="4, 5, 6, 7"


# In[4]:


# Load the data
h5f = h5py.File('../CNN_experiments/data/data_nearest_neighbor.h5', 'r')
train_x, train_y = h5f['train_x'][:], h5f['train_y'][:]
valid_x, valid_y = h5f['valid_x'][:], h5f['valid_y'][:]
test_x, test_y = h5f['test_x'][:], h5f['test_y'][:]
h5f.close()

print(train_x.shape, valid_x.shape, test_x.shape)


# In[5]:


from models import Squeeze_model


# In[6]:


# from IPython.display import SVG
# from keras.utils.vis_utils import plot_model, model_to_dot
# model_input = Input(shape=(24, 24, 24, 16))
# squeeze_model = Model(inputs=model_input, outputs=Squeeze_model(model_input))
# #plot_model(squeeze_model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
# SVG(model_to_dot(squeeze_model, show_layer_names=True, show_shapes=True).create(prog='dot', format='svg'))
# # plot_model(get_model4((24, 24, 24, 16)))


# In[7]:


# Training parameters
nb_gpus = 4
nb_batch = nb_gpus*3
nb_epochs = 200
l_rate = 1e-4
decay_rate = l_rate / nb_epochs


# In[8]:


# checkpoint
outputFolder = './weights'
# if not os.path.exists(outputFolder):
#     os.makedirs(outputFolder)

filepath=outputFolder+"/weights-rotated_random.h5"

#from keras.callbacks import EarlyStopping, ReduceLROnPlateau
callbacks_list = [ModelCheckpoint(filepath, 
                                  monitor='val_loss',
                                  verbose=1,
                                  save_best_only=True,
                                  save_weights_only=True,
                                  mode='auto', period=1)]


# In[9]:


model_input = Input(shape=(28, 28, 28, 16))
squeeze_model = Model(inputs=model_input, outputs=Squeeze_model(model_input))
model = multi_gpu_model(squeeze_model, gpus=nb_gpus)

model.compile(optimizer=optimizers.adam(lr=l_rate),# beta_1=0.99, beta_2=0.999),
              loss='mean_absolute_error')


# In[10]:


from my_classes import AugmentedDataGeneratorRandom


# In[11]:


rotation_count = 32
aug_data_gen = AugmentedDataGeneratorRandom(x=train_x, y=train_y, batch_size=nb_batch, rotation_count=rotation_count)
aug_val_gen = AugmentedDataGeneratorRandom(x=valid_x, y=valid_y, batch_size=nb_batch, rotation_count=rotation_count)


# In[12]:


for x, y in aug_data_gen:
    print(x.shape, y.shape)
    break


# In[13]:


history = model.fit_generator(generator=aug_data_gen, validation_data=aug_val_gen,
                              use_multiprocessing=False, 
                              epochs=nb_epochs, 
                              max_queue_size=10, 
                              workers=56, 
                              verbose=1, callbacks=callbacks_list)


# In[14]:


# Save the history
import pickle

with open(os.path.join(outputFolder, "history_rotated-random.pickle"), 'wb') as f:
    pickle.dump(history.history, f)


# In[16]:



plt.figure()
plt.plot(history.history['loss'][:100])
plt.plot(history.history['val_loss'][:100])
plt.xlabel("Epochs")
plt.ylabel("Loss (MAE)")
plt.legend(['Train Loss', 'Validation Loss'])
plt.savefig('nn-rotated-random.png', format='png', dpi=1000)
plt.show()


# In[15]:


# Save the weights
model.load_weights(filepath)


# In[16]:


from sklearn.metrics import r2_score


# In[17]:


rotation_count = 32
sample_size = 200


# In[18]:


train_aug_data_gen = AugmentedDataGeneratorRandom(x=train_x[:sample_size],
                                            y=train_y[:sample_size],
                                            batch_size=sample_size,
                                            rotation_count=rotation_count)

test_aug_data_gen = AugmentedDataGeneratorRandom(x=test_x[:sample_size],
                                            y=test_y[:sample_size],
                                            batch_size=sample_size,
                                            rotation_count=rotation_count)


# In[19]:


# Train r2

for x, y in train_aug_data_gen:
    # Without average
    train_r2 = r2_score(y_true=y, y_pred=model.predict(x))
    print("Train r2 (without average): ", train_r2)
    
    # With average
    y_pred = model.predict(x)[:, 0]
    
    sample_y = np.zeros(sample_size)
    sample_ypred = np.zeros(sample_size)
    for i in range(sample_size):
        start = i*(rotation_count+1)
        end = i*(rotation_count+1) + (rotation_count+1)
        mean_ypred = np.mean(y_pred[start:end])
        mean_y = np.mean(y[start:end])
        sample_ypred[i] = mean_ypred
        sample_y[i] = mean_y
    
    train_r2 = r2_score(y_true=sample_y, y_pred=sample_ypred)
    print("Train r2 (with average): ", train_r2)
    break


# In[20]:


# Test r2

for x, y in test_aug_data_gen:
    # Without average
    test_r2 = r2_score(y_true=y, y_pred=model.predict(x))
    print("Test r2 (without average): ", test_r2)
    
    # With average
    y_pred = model.predict(x)[:, 0]
    
    sample_y = np.zeros(sample_size)
    sample_ypred = np.zeros(sample_size)
    for i in range(sample_size):
        start = i*(rotation_count+1)
        end = i*(rotation_count+1) + (rotation_count+1)
        mean_ypred = np.mean(y_pred[start:end])
        mean_y = np.mean(y[start:end])
        sample_ypred[i] = mean_ypred
        sample_y[i] = mean_y
    
    test_r2 = r2_score(y_true=sample_y, y_pred=sample_ypred)
    print("Test r2 (with average): ", test_r2)
    break

