# %% [markdown]
# <p style="text-align:right;">Mario Stanke, Felix Becker, University of Greifswald, Germany</p>
# 
# ## CNN to distinguish cats, dogs, wolves and plants
# In this notebook we will perform image classification with a convolutional neural network. We will use a dataset of photos of 3 animals (tabby cats, German shepherds and grey woves) and plants seedlings of 9 plant species. The animal photos were downloaded from [IMAGENET](http://www.image-net.org/). The plant dataset is an excerpt from https://www.kaggle.com/vbookshelf/a-simple-keras-solution. All input images are in subfolders of ```data/cats-dogs-plants``` respective to their class.
# 
# What you will learn: 
#  - convolutional layers (Conv2D)
#  - handling of image datasets (```tf.data.Dataset```)
#  - image augmentation
#  - monitoring training progress (callbacks)
#  - saving and reading a model to and from a file
#  - classification error analysis

# %%
import random
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Input, MaxPooling2D, Dropout, BatchNormalization
import pandas as pd
import pandas.util
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import math
                         

# %% [markdown]
# ## Image Data Preparation - Make Data Frames
# We will create **Pandas data frames** to hold the *file names and class labels only*. These data structures are small as they do not contain the actual image data. The images are on the drive and never are they all loaded to memory, so this approach scales to very large training sets of images.

# %%
# directory with image class subdirectories
img_path = '../data/cats-dogs-plants'

# Create a pandas dataframe from a tab separated file 
df = pd.read_csv(img_path + "/classes-and-fnames.txt", sep = '\t', names = ['classname', 'fname'])
df['path'] = img_path + '/' + df['classname'] + "/" + df['fname']

df.head()

# %%
#check how many images we have in total
df.shape

# %%
# associate class names with a class (0 .. K-1)
classnames = df['classname'].unique() # all 12 species names

K = classnames.size  # 12
name2class = dict(zip(classnames, range(K))) # dictionary that maps a name to its index in classnames array
name2class

# %%
# Add a column 'class' to data frame  with the number representing the species name
df['class'] = df['classname'].map(name2class) # new column class with number representing plant name

# print a few random example lines
df.sample(n=5)

# %%
# Plot a few sample images for each class
examples_per_class = 4
_, ax = plt.subplots(nrows = K, ncols = examples_per_class, 
                     figsize = (3 * examples_per_class, 3 * K)) # adjust size here
    
for i in range(K): # loop over classes = rows
    sample = df[df['class'] == i].sample(n = examples_per_class)
    for j in range(examples_per_class):
        path = sample.iloc[j]['path']
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # cv2 has color order BGR rather than RGB
        ax[i, j].imshow(img)
        ax[i, j].set_title(classnames[i], fontsize=15)  
plt.tight_layout()
plt.show()

# %%
num_animals = 3 # assume that the first classes are animals
# percentages for the relative frequencies of each plant
100 * df['classname'].value_counts() / df.shape[0]

# %% [markdown]
# **Question:** What is the fraction of correct guesses (accuracy) you are expected to get with a good random guessing method, i.e. without even looking at a photo?
# 
# ### Training, Validation and Test Split

# %%
# randomly split data frame into training, validation and test data frames
num_imgs  = df.shape[0] # total number of examples
num_test  = 400 # size of test set, used only once at end
num_val   = 300 # size of validation set, used to monitor training progress
num_train = num_imgs - num_test - num_val # size of training set, the (large) rest

assert num_train > 0, "Error: examples consumed by test and validation sets alone"

# construct an array [0, 1, ..., num_imgs] with the indices = row numbers of df
all_idxs = np.array(range(num_imgs))
np.random.shuffle(all_idxs) # random order, so there is no bias in any partition

# construct training and testing data frames 
test_df = df.iloc[all_idxs[0 : num_test]]
val_df = df.iloc[all_idxs[num_test : num_test + num_val]]
train_df = df.iloc[all_idxs[num_test + num_val :]] # the rest

assert train_df.shape[0] == num_train, "Internal error of 3-way split into train, test, val"
print("Sizes: train =", num_train, "\tvalidation =", num_val, "\ttest =", num_test)

# %% [markdown]
# ## Make TensorFlow Datasets

# %%
img_size = 96    # width and height of images when input to the model (an image will be resized, if required)
batch_size = 32  # for training and prediction

# %% [markdown]
# ### Image Augmentation
# From one image on the drive one can generate many training examples, e.g.
# 
#  - by mirroring and rotating the image where it preserves the label
#  - by random cropping followed by resizing
#  - by random changes to colors.
#     
# **Goals:**
#  - Have a much larger effective training set (the bigger the better).
#  - Make it more robust with respect to biases, e.g. the direction of light during photography may have
#     had a systematic bias, say, if maize was on one side of the greenhouse and sugar beets on the other. 

# %%
def path_to_array(filename, label):
    """ Map a filename to an actual image tensor using image augmentation, one-hot encode label.
        Args:
            filename: String filepath to an image
            label: Integer representation of the class the image belongs to
        Returns:
            img: A 3D tensor representing the image.
            label: A one-hot representation of the label.
    """
    img = tf.io.read_file(filename)
    tf.image.decode_png(img, channels=3)  # Attempt decoding
    
    img = tf.io.read_file(filename)
    img = tf.image.decode_png(img, channels = 3)
    # now img is 3 dim array of numbers in {0,..., 255}
    img = tf.cast(img, dtype = tf.float32) / 255. # scale to floating point number in [0,1]
    
    # resize to fixed input size
    img = tf.image.resize(img, [img_size, img_size])
    
    # image augmentation, make one of 8 congruent or mirrored images
    img = tf.image.random_flip_left_right(img) # 50% chance to mirror vertically
    num_rot = np.random.randint(0, 4) # rotate 0-3 times counter-clockwise by a 90 degrees  
    img = tf.where(label >= num_animals, # assume that plants photos are bird's-eye view and may be rotated
                   tf.image.rot90(img, k = num_rot),
                   img)

    # randomly change colors
    img = tf.image.random_brightness(img, max_delta = 0.1)
    img = tf.image.random_hue(img, max_delta = 0.02)
    img = tf.image.random_saturation(img, lower = 0.95, upper = 1.05)
    
    # make sure the pixel values are still in the correct range
    img = tf.clip_by_value(img, 0., 1.)
    
    # one-hot encode the label, e.g. 3 becomes [0,0,0,1,0,0,0,0,0,0,0,0]
    label = tf.one_hot(label, depth = K)
    return img, label

# %%
#lets test this method
sample_img, label = path_to_array( '../data/cats-dogs-plants/cat/1000.png', 0 )
sample_img.shape, label

# %% [markdown]
# Next, we will use the `map` function to apply a callable (in this case `path_to_array`) to all examples in a dataset and get a new dataset. 

# %%
def make_dataset(df):
    """ Make a tf dataset of images from a pd data frame of file paths """

    # first, make dataset with just the relevant: path and class
    ds_path = tf.data.Dataset.from_tensor_slices((df['path'], df['class']))

    # convert to data set with actual images
    # mapping without num_parallel_calls significantly slows down training (can you see why?) 
    ds = ds_path.map(path_to_array, num_parallel_calls=4)

    # group images together
    ds = ds.batch(batch_size)
    
    # prefetching means running the preprocessing and model execution in parallel
    # this is very helpful because loading an image from disk and augmenting it can be expensive
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

test_ds  = make_dataset(test_df)
val_ds   = make_dataset(val_df)
train_ds = make_dataset(train_df)
train_ds = train_ds.repeat() # infinitely repeat, learn on these images as long as we say

# %% [markdown]
# A `tf.dataset` is a generator. We can use a simple for-loop to iterate over elements:

# %%
# print and display first example from first batch
for (x, y) in train_ds.take(1):
    print("data frame:\n", train_df.iloc[0])
    print("\ndata set:")
    _, ax = plt.subplots(ncols = 2, figsize = (12, 5))
    
    # plot original image as on drive
    path = train_df.iloc[0]['path']
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
    ax[0].imshow(img)
    ax[0].set_title(classnames[np.argmax(y[0])] + "\noriginal image")
    
    # plot augmented image
    ax[1].imshow(x[0])
    ax[1].set_title("augmented image")

# %% [markdown]
# ## Define and Train a Convolutional Neural Network
# **Layer types:**
#  - Conv2D convolutional layer, here inputs are 3dim and the convolution sums over the 3rd dim.
#  - [MaxPooling2D](https://www.tensorflow.org/api_docs/python/tf/keras/layers/MaxPool2D) takes local maxima over a rectangular region.
#  - [Dropout](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dropout) randomly deactivates neurons during training, which can make training more robust.
#  - [BatchNormalization](https://arxiv.org/pdf/1502.03167.pdf), centralizes and scales its input to have approximately 0 mean and variance 1.
# 
# Model architecture modified from [this source](https://www.kaggle.com/fmarazzi/baseline-keras-cnn-roc-fast-5min-0-8253-lb)

# %%
tf.random.set_seed(24092021) # so we all get the same pseudorandom results

# width and height of a filter
kernel_size = (3, 3)

# width and height of a pool
pool_size   = (2, 2)

# number of filters (channel depth) in the first "block" of the model
first_filters  = 32

second_filters = 64
third_filters  = 128

# dropout probabilities
dropout_conv  = 0.3
dropout_dense = 0.3

model = tf.keras.models.Sequential() # sequential stack of layers
model.add(Input(shape = (img_size, img_size, 3)))

model.add( BatchNormalization())
model.add( Conv2D (first_filters, kernel_size, activation = 'relu'))
model.add( Conv2D (first_filters, kernel_size, activation = 'relu'))
model.add( Conv2D (first_filters, kernel_size, activation = 'relu'))
model.add( MaxPooling2D (pool_size = pool_size)) 
model.add( Dropout (dropout_conv))

model.add( Conv2D (second_filters, kernel_size, activation ='relu'))
model.add( Conv2D (second_filters, kernel_size, activation ='relu'))
model.add( Conv2D (second_filters, kernel_size, activation ='relu'))
model.add( MaxPooling2D (pool_size = pool_size))
model.add( Dropout (dropout_conv))

model.add( Conv2D (third_filters, kernel_size, activation ='relu'))
model.add( Conv2D (third_filters, kernel_size, activation ='relu'))
model.add( Conv2D (third_filters, kernel_size, activation ='relu'))
model.add( MaxPooling2D (pool_size = pool_size))
model.add( Dropout (dropout_conv))

model.add( Flatten())
model.add( Dense (256, activation = "relu", kernel_regularizer = tf.keras.regularizers.l2(0.005)))
model.add( Dropout (dropout_dense))
model.add( Dense (K, activation = "softmax"))

model.summary()
# draw an image with the layers
# tf.keras.utils.plot_model(model) # can be helpful to look at but does not work on brain

# %%
# define the loss, optimization algorithm and prepare the model for gradient computation 
model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0005), 
              loss = tf.keras.losses.categorical_crossentropy,
              metrics = ['accuracy'])

# %%
# Callbacks: What should be done during (long) training?
modelfname = "animals_and_plants.keras"
# Function to store model to file
# Will always keep the model with the lowest validation loss
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    modelfname, monitor = 'val_loss', mode = 'min', 
    save_best_only = True, verbose = 1)

# Function to decrease learning rate by 'factor'
# when there has been no significant improvement in the last 'patience' epochs.
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor = 'val_loss', mode = 'min', factor = 0.75, patience = 4, verbose = 1)
                         
# fit the model
# note that we have to state steps_per_epoch because our training dataset loops indefinetely
history = model.fit(
    train_ds, epochs = 1, #50, 
    steps_per_epoch = 600, # num_train / batch_size would use each example once on average
    validation_data = val_ds, verbose = 1,
    callbacks = [checkpoint, reduce_lr])

# %%
# plot the training history as loss and accuracy curves
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

_, ax = plt.subplots(ncols = 2, figsize = (15, 6))
ax[0].plot(epochs, loss, 'bo', label = 'Training loss')
ax[0].plot(epochs, val_loss, 'g', label = 'Validation loss')
ax[0].set_title('Training and validation loss')
ax[0].legend()

ax[1].plot(epochs, acc, 'bo', label = 'Training acc')
ax[1].plot(epochs, val_acc, 'g', label = 'Validation acc')
ax[1].set_title('Training and validation accuracy')
ax[1].legend();

# %% [markdown]
# We can see that the validation results stagnate at epoch 20 while the training results improve. This is overfitting. In practice we should now seek to
# 1. reduce the complexity of our model (drop some layers or decrease the number of filters)
# 
# 2. or regularize more (increase dropout or do more image augmentation).

# %% [markdown]
# ## Evaluation after Training

# %%
# Load the parameters with the best validation accuracy during training.
# This works also if you interruped the training, provided modelfname has been written to yet!
model.load_weights(modelfname)

test_loss, test_acc = model.evaluate(test_ds, verbose = 0)
print("Loss on test set:", test_loss, "\nAccuracy on test set:", test_acc)

# %%
# make a prediction
prediction = model.predict(test_ds)
yhat = prediction.argmax(axis = 1)
if 'pred' not in test_df:
    test_df.insert(4, 'pred',  prediction.argmax(axis = 1))
if 'confidence' not in test_df:
    test_df.insert(5, 'confidence',  prediction.max(axis = 1))

print("Predicted probabilities for first 5 examples:\n", np.round(prediction[0:5], 3))
print("Predicted classes for first 5 examples:\n", test_df['pred'][0:5])
print("Probability of prediction:\n", test_df['confidence'][0:5])
test_df.head()

# %%
confusion_matrix(test_df['class'], test_df['pred'])

# %%
print(classification_report(test_df['class'], test_df['pred'], target_names = classnames))

# %% [markdown]
# ### Display misclassified photos of animals

# %%
# Make data frame of all misclassifications that involve an animal
false_animals_df = test_df[(test_df['class'] != test_df['pred']) # misclassified and
                           & ((test_df['class'] < num_animals) # is an animal
                              | (test_df['pred'] < num_animals))] # or was predicted to be an animal
numfalse = false_animals_df.shape[0]
print("number of misclassifications that involve an animal :", numfalse)

# %%
num_show = min(numfalse, 50) # show at most 50 false examples
ncols = 4
nrows = math.ceil(num_show / ncols) # round up
nrows = min(nrows, 15) # at most 15 rows
f, ax = plt.subplots(nrows, ncols, figsize = (3 * ncols, 3 * nrows))
for k in range(num_show):
    i = math.floor(k / ncols) # row
    j = k % ncols # column
    record = false_animals_df.iloc[k]
    path = record['path']
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
    ax[i, j].imshow(img)
    ax[i, j].set_title(classnames[record['class']] + " predicted as\n"
                       + str(classnames[record['pred']]) + " with conf. "
                       + str(np.round(record['confidence'], 3)),
                      fontsize=12)

plt.tight_layout()

# %%



