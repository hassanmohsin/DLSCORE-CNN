
# coding: utf-8

# ## Training

# In[2]:


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

import os
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
os.environ["CUDA_VISIBLE_DEVICES"]="0, 1, 2, 3"


# In[3]:


from my_classes import DataGenerator, AugmentedDataGenerator


# In[4]:


# Load the data
h5f = h5py.File('data/data_nearest_neighbor.h5', 'r')
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
nb_batch = nb_gpus*16
nb_epochs = 200
l_rate = 1e-4
decay_rate = l_rate / nb_epochs


# In[8]:


data_gen = DataGenerator(x=train_x, y=train_y, batch_size=nb_batch)
val_gen = DataGenerator(x=valid_x, y=valid_y, batch_size=nb_batch)


# In[9]:


# Build the model and train
model_input = Input(shape=(24, 24, 24, 16))
squeeze_model = Model(inputs=model_input, outputs=Squeeze_model(model_input))
model = multi_gpu_model(squeeze_model, gpus=nb_gpus)
decay_rate = l_rate / nb_epochs
model.compile(optimizer=optimizers.adam(lr=l_rate),# decay=decay_rate, beta_1=0.99, beta_2=0.999),
              loss='mean_absolute_error')
              #target_tensors=[staging_area_callback.target_tensor],
              #fetches=staging_area_callback.extra_ops)


# In[10]:


# checkpoint
outputFolder = './weights'
# if not os.path.exists(outputFolder):
#     os.makedirs(outputFolder)

filepath=outputFolder+"/weights-original.h5"

#from keras.callbacks import EarlyStopping, ReduceLROnPlateau
callbacks_list = [ModelCheckpoint(filepath, 
                                  monitor='val_loss',
                                  verbose=1,
                                  save_best_only=True,
                                  save_weights_only=True,
                                  mode='auto', period=1)]


# In[11]:


history = model.fit_generator(generator=data_gen, validation_data=val_gen,
                              use_multiprocessing=False, 
                              epochs=nb_epochs, 
                              max_queue_size=10, 
                              workers=56, 
                              verbose=1, 
                              callbacks=callbacks_list)


# In[12]:


# Save the history
import pickle

with open(os.path.join(outputFolder, "history_original.pickle"), 'wb') as f:
    pickle.dump(history.history, f)


# In[13]:


# Save the weights
# model.save_weights('weights/weights_original.h5')


# In[14]:


model.load_weights(filepath)


# In[15]:


# First 100 epochs
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.show()


# In[41]:


import pickle
with open("weights/history_cano_rotated_random.pickle", 'rb') as f:
    history = pickle.load(f)
    
plt.figure()
plt.plot(history['loss'])#[:100])
plt.plot(history['val_loss'])#[:100])
plt.xlabel("Epochs")
plt.ylabel("Loss (MAE)")
plt.legend(['Train Loss', 'Validation Loss'])
plt.savefig('nn-cano-rotated-random.png', format='png', dpi=1000)
plt.show()


# In[16]:


from sklearn.metrics import r2_score


# In[17]:


#train_r2 = r2_score(y_true=train_y, y_pred=model.predict(train_x))
train_r2 = r2_score(y_true=train_y[:2000], y_pred=model.predict(train_x[:2000]))
print("Train r2: ", train_r2)


# In[18]:


#train_r2 = r2_score(y_true=train_y, y_pred=model.predict(train_x))
test_r2 = r2_score(y_true=test_y, y_pred=model.predict(test_x))
print("Test r2: ", test_r2)

