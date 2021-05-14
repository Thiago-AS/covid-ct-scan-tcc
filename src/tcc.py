#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import nibabel as nib
import imgaug as ia
import imgaug.augmenters as iaa
import tqdm
import warnings
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, BatchNormalization, Activation, Dropout
from tensorflow.keras.layers import concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
from tensorflow.keras.utils import plot_model
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

warnings.filterwarnings('ignore')
print("Version: ", tf.version.VERSION)
if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TF")


# ## 1. Load metada

# In[2]:


metadata = pd.read_csv('../input/covid19-ct-scans/metadata.csv')
metadata.head()


# In[3]:


ORIGINAL_IMG_SIZE = 512
TRAIN_IMG_SIZE = 128
TRAIN_IMG_CHANNELS = 1
SEED = 42
ia.seed(SEED)
np.random.seed = SEED
tf.seed = SEED
CLAHE = cv.createCLAHE(clipLimit=3.0)


# In[4]:


def show_image(img, title, show_hist, axes):
    axes[0].imshow(img, cmap='bone')
    axes[0].set_title(title)
    axes[0].set_xticks([]); axes[0].set_yticks([])
    
    if show_hist:
        axes[1].hist(img.flatten(), bins=256)
        axes[1].set_title(f"{title} histograma")
        axes[1].set_xticks([]); axes[1].set_yticks([])

def show_data(cts, lungs, infections, index, axes):        
    axes[0].imshow(cts[index], cmap='bone')
    axes[0].set_title("Tomografia")
    axes[0].set_xticks([]); axes[0].set_yticks([])
    
    axes[1].imshow(lungs[index], cmap='bone')
    axes[1].set_title("Máscara Pulmonar")
    axes[1].set_xticks([]); axes[1].set_yticks([])

    axes[2].imshow(infections[index], cmap='bone')
    axes[2].set_title("Máscara das lesões")
    axes[2].set_xticks([]); axes[2].set_yticks([])
    
def show_overlay_infection(cts, infections, percent, index, axes) :
    axes[0].imshow(cts[index], cmap='bone')
    axes[0].set_xticks([]); axes[0].set_yticks([])
    axes[0].set_title('Tomografia')

    axes[1].imshow(cts[index], cmap='bone')
    axes[1].imshow(infections[index], alpha=0.65, cmap='nipy_spectral')
    axes[1].set_xticks([]); axes[1].set_yticks([])
    axes[1].set_title('Lesões')
    axes[1].set_xlabel(f"Porcentagem do pulmão afetado: {percent}%")
    
def calculate_infection_percent(lung, infection):
    lung = (lung > 0.25).astype(np.uint8)
    infection = (infection > 0.5).astype(np.uint8)
    pix_lung = np.count_nonzero(lung)
    pix_infection = np.count_nonzero(infection)
    percentage = (pix_infection * 100)/ pix_lung
    return "{:.2f}".format(percentage)


# In[5]:


def get_contours(img):
    img = np.uint8(img*255)
    img = cv.GaussianBlur(img, (5,5), 0)
    
    _, binary_img = cv.threshold(img, 50, 255, cv.THRESH_BINARY)
    contours, _ = cv.findContours(binary_img, 2, 1)

    x, y = img.shape
    contours = [cc for cc in contours if is_contour_ok(cc, x * y)]
    return contours

def is_contour_ok(cc, size):
    x, y, w, h = cv.boundingRect(cc)
    if ((w < 50 and h > 150) or (w > 150 and h < 50)) : 
        return False
    area = cv.contourArea(cc)
    return area < (size * 0.5) and area > 200

def find_boundaries(img, contours):
    min_y, min_x = img.shape
    max_x, max_y = 0, 0

    for cc in contours:
        x, y, w, h = cv.boundingRect(cc)
        if x < min_x: min_x = x
        if y < min_y: min_y = y
        if x + w > max_x: max_x = x + w
        if y + h > max_y: max_y = y + h

    return (min_x, min_y, max_x, max_y)

def crop(img, boundaries):
    min_x, min_y, max_x, max_y = boundaries
    return img[min_y:max_y, min_x:max_x]

def clahe_enhancer(img):
    img = np.uint8(img*255)  
    clahe_img = CLAHE.apply(img)
        
    return clahe_img


# In[6]:


def normalize_image(img):
    if img.max() == img.min():
        return img

    return (img - img.min())/(img.max() - img.min())

def load_nii_file(file_name, mask=False):
    all_imgs = []
    raw_imgs = nib.load(file_name)
    raw_imgs = raw_imgs.get_fdata()
    raw_imgs = np.rot90(np.array(raw_imgs))
    raw_imgs = np.reshape(np.rollaxis(raw_imgs, 2), (raw_imgs.shape[2], raw_imgs.shape[0], raw_imgs.shape[1], 1))

    for i in tqdm.tqdm(range(raw_imgs.shape[0])):
        img = raw_imgs[i]
        img = cv.resize(img, dsize=(ORIGINAL_IMG_SIZE, ORIGINAL_IMG_SIZE), interpolation=cv.INTER_AREA)
        img = normalize_image(img)
        if mask:
            img = np.where(img > 0, 1.0, 0.0)
        all_imgs.append(img)

    return all_imgs

def apply_clahe(imgs):
    enhanced_imgs = []
    for i in tqdm.tqdm(range(len(imgs))):
        enhanced_imgs.append(clahe_enhancer(imgs[i]))
    return enhanced_imgs

def crop_images_to_lung_bounds(cts, lungs, infections):
    all_cropped_cts, all_cropped_lungs, all_cropped_infections = [], [], []
    for i in tqdm.tqdm(range(len(cts))):
        lung_contours = get_contours(lungs[i])
        lung_bounds = find_boundaries(lungs[i], lung_contours)
        
        cropped_ct = crop(cts[i], lung_bounds)
        cropped_lung = crop(lungs[i], lung_bounds)
        cropped_infection = crop(infections[i], lung_bounds)
        
        all_cropped_cts.append(cropped_ct)
        all_cropped_lungs.append(cropped_lung)
        all_cropped_infections.append(cropped_infection)
        
    return all_cropped_cts, all_cropped_lungs, all_cropped_infections

def resize_and_reshape_to_train_values(cts, lungs, infections):
    all_resized_cts, all_resized_lungs, all_resized_infections, errors_index = cts[:], lungs[:], infections[:], []
    for i in tqdm.tqdm(range(len(cts))) :
        try:
            all_resized_cts[i] = cv.resize(all_resized_cts[i], dsize=(TRAIN_IMG_SIZE, TRAIN_IMG_SIZE), interpolation=cv.INTER_AREA)
            all_resized_cts[i] = np.reshape(all_resized_cts[i], (TRAIN_IMG_SIZE, TRAIN_IMG_SIZE, TRAIN_IMG_CHANNELS))

            all_resized_lungs[i] = cv.resize(all_resized_lungs[i], dsize=(TRAIN_IMG_SIZE, TRAIN_IMG_SIZE), interpolation=cv.INTER_AREA)
            all_resized_lungs[i] = np.reshape(all_resized_lungs[i], (TRAIN_IMG_SIZE, TRAIN_IMG_SIZE, TRAIN_IMG_CHANNELS))

            all_resized_infections[i] = cv.resize(all_resized_infections[i], dsize=(TRAIN_IMG_SIZE, TRAIN_IMG_SIZE), interpolation=cv.INTER_AREA)
            all_resized_infections[i] = np.reshape(all_resized_infections[i], (TRAIN_IMG_SIZE, TRAIN_IMG_SIZE, TRAIN_IMG_CHANNELS))
        except Exception as e:
            errors_index.append(i)
            
    return all_resized_cts, all_resized_lungs, all_resized_infections, errors_index

def clean_nulls(errors_index, *args):
    for arg in args:
        for i in reversed(range(len(errors_index))):
            del arg[errors_index[i]]


# In[7]:


cts, lungs, infections = [], [], []
for i in range(metadata.shape[0]):
    print(f"Loading CT [{metadata.loc[i, 'ct_scan']}], LUNG [{metadata.loc[i, 'lung_mask']}] and INFECTION [{metadata.loc[i, 'infection_mask']}]")
    partial_cts = load_nii_file(metadata.loc[i, 'ct_scan'])
    partial_lungs = load_nii_file(metadata.loc[i, 'lung_mask'])
    partial_infections = load_nii_file(metadata.loc[i, 'infection_mask'])
    cts = cts + partial_cts
    lungs = lungs + partial_lungs
    infections = infections + partial_infections    


# In[8]:


enhanced_cts = apply_clahe(cts)
cropped_cts, cropped_lungs, cropped_infections = crop_images_to_lung_bounds(enhanced_cts, lungs, infections)
final_cts, final_lungs, final_infections, errors_index = resize_and_reshape_to_train_values(cropped_cts, cropped_lungs, cropped_infections)
clean_nulls(errors_index, cts, lungs, infections, enhanced_cts, cropped_cts, cropped_lungs, cropped_infections, final_cts, final_lungs, final_infections)


# In[9]:


i = np.random.randint(low=0, high=len(cts))
fig, axes = plt.subplots(3, 4, figsize=(20,10))    
show_data(cts, lungs, infections, i, list(axes[:, 0]))
show_data(enhanced_cts, lungs, infections, i, list(axes[:, 1]))
show_data(cropped_cts, cropped_lungs, cropped_infections, i, list(axes[:, 2]))
show_data(final_cts, final_lungs, final_infections, i, list(axes[:, 3]))


# In[10]:


total = len(final_infections)
infect_bool = np.ones(total)
for i in range(total):
    if np.unique(final_infections[i]).size == 1:
        infect_bool[i] = 0
        
print(f"Total CTs - {total}")
print(f"Infected CTs - {int(infect_bool.sum())}")
print(f"Non Infected CTs - {int(total - infect_bool.sum())}")


# ## Data augmentation pipeline

# In[11]:


seq = iaa.Sequential([
    iaa.Fliplr(0.5),
    iaa.Flipud(0.5),
    iaa.Affine(
        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
        rotate=(-15, 15),
        shear=(-15, 15)
    )
], random_order=True)


# In[12]:


seq_det = seq.to_deterministic()
augmented_cts = seq_det.augment_images(final_cts)
augmented_lungs = seq_det.augment_images(final_lungs)
augmented_infections = seq_det.augment_images(final_infections)


# In[13]:


i = np.random.randint(low=0, high=len(final_cts), size=2)
fig, axes = plt.subplots(3, 4, figsize=(20,10))

show_data(final_cts, final_lungs, final_infections, i[0], list(axes[:, 0]))
show_data(augmented_cts, augmented_lungs, augmented_infections, i[0], list(axes[:, 1]))
show_data(final_cts, final_lungs, final_infections, i[1], list(axes[:, 2]))
show_data(augmented_cts, augmented_lungs, augmented_infections, i[1], list(axes[:, 3]))


# In[14]:


all_ct_data = final_cts + augmented_cts
all_lung_data = final_lungs + augmented_lungs
all_infection_data = final_infections + augmented_infections

all_ct_data = np.asarray(all_ct_data)
all_lung_data = np.asarray(all_lung_data)
all_infection_data = np.asarray(all_infection_data)

all_ct_data, all_lung_data, all_infection_data = shuffle(all_ct_data, all_lung_data, all_infection_data, random_state=SEED)


# In[15]:


i = np.random.randint(low=0, high=len(all_ct_data), size=4)
fig, axes = plt.subplots(3, 4, figsize=(20,10))

show_data(all_ct_data, all_lung_data, all_infection_data, i[0], list(axes[:, 0]))
show_data(all_ct_data, all_lung_data, all_infection_data, i[1], list(axes[:, 1]))
show_data(all_ct_data, all_lung_data, all_infection_data, i[2], list(axes[:, 2]))
show_data(all_ct_data, all_lung_data, all_infection_data, i[3], list(axes[:, 3]))


# ## Split data into train and validation sets

# In[16]:


train_size = int(0.8 * all_ct_data.shape[0])
test_size = int(0.1 * all_ct_data.shape[0])

X_train, y_lung_train, y_infection_train = (all_ct_data[:train_size]/255, 
                               all_lung_data[:train_size], 
                               all_infection_data[:train_size])

X_validation, y_lung_validation, y_infection_validation = (all_ct_data[train_size:train_size+test_size]/255, 
                               all_lung_data[train_size:train_size+test_size],
                               all_infection_data[train_size:train_size+test_size])

X_test, y_lung_test, y_infection_test = (all_ct_data[train_size+test_size:]/255, 
                            all_lung_data[train_size+test_size:],
                            all_infection_data[train_size+test_size:])

print(X_train.shape, y_lung_train.shape, y_infection_train.shape)
print(X_validation.shape, y_lung_validation.shape, y_infection_validation.shape)
print(X_test.shape, y_lung_test.shape, y_infection_test.shape)


# ## Model

# In[20]:


def unet_block_down(filter_value, input_ts):
    c = Conv2D(filter_value, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(input_ts)
    c = Dropout(0.1)(c)
    c = Conv2D(filter_value, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c)
    p = MaxPooling2D((2, 2))(c)
    
    return c, p

def unet_block_up(filter_value, input_ts, concat_ts):
    u = Conv2DTranspose(filter_value, (2, 2), strides=(2, 2), padding='same')(input_ts)
    u = concatenate([u, concat_ts])
    c = Conv2D(filter_value, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u)
    c = Dropout(0.1)(c)
    c = Conv2D(filter_value, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c)
    
    return u, c

def unet(input_shape) :
    
    inputs = Input(input_shape)
    
    c1, p1 = unet_block_down(64, inputs)
    c2, p2 = unet_block_down(128, p1)
    c3, p3 = unet_block_down(256, p2)
    c4, p4 = unet_block_down(512, p3)

    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = Dropout(0.1)(c5)
    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

    u6, c6 = unet_block_up(512, c5, c4)
    u7, c7 = unet_block_up(256, c6, c3)
    u8, c8 = unet_block_up(128, c7, c2)
    u9, c9 = unet_block_up(64, c8, c1)

    lung_seg = Conv2D(1, (1, 1), activation='sigmoid', name='lung_output')(c9)
                
    c1, p1 = unet_block_down(64, inputs)
    c2, p2 = unet_block_down(128, p1)
    c3, p3 = unet_block_down(256, p2)
    c4, p4 = unet_block_down(512, p3)

    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = Dropout(0.1)(c5)
    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

    u6, c6 = unet_block_up(512, c5, c4)
    u7, c7 = unet_block_up(256, c6, c3)
    u8, c8 = unet_block_up(128, c7, c2)
    u9, c9 = unet_block_up(64, c8, c1)

    infect_seg = Conv2D(1, (1, 1), activation='sigmoid', name='infect_output')(c9)

    model = Model(inputs=inputs, outputs=[lung_seg, infect_seg], name='unet_model')
    
    return model


# In[22]:


def cts_block_down(filter_value, input_ts, pool_size=(2, 2)) :
    c = Conv2D(filter_value, (3,3), activation='relu', padding='same', kernel_initializer="he_normal")(input_ts)
    c = Conv2D(filter_value, (3, 3), activation='relu', padding='same', kernel_initializer="he_normal")(c)
    p = BatchNormalization()(c)
    c = MaxPooling2D(pool_size)(p) 
    c = Dropout(0.2)(c) 
    
    return c, p

def cts_block_up(filter_value, input_ts) :
    c = BatchNormalization() (input_ts)
    c = Conv2D(filter_value, (3, 3), activation='relu', padding='same', kernel_initializer="he_normal") (c)
    c = Conv2D(filter_value, (3, 3), activation='relu', padding='same', kernel_initializer="he_normal") (c) 
    
    return c

def cts_model(input_shape) :
    
    inputs = Input(input_shape)
    
    c1, p1 = cts_block_down(32, inputs)
    c2, p2 = cts_block_down(64, c1)
    c3, _ = cts_block_down(128, c2, pool_size=(1, 1))
    c4, _ = cts_block_down(256, c3, pool_size=(1, 1))

    u5 = cts_block_up(256, c4)
    u5 = Conv2DTranspose(128, (2, 2), strides=(2,2), padding='same')(u5) 
    
    u6 = cts_block_up(128, u5)
    u6 = Conv2DTranspose(64, (2, 2), padding='same')(u6)
    u6 = concatenate([u6, p2])
    
    u7 = cts_block_up(64, u6)
    u7 = Conv2DTranspose(32, (2, 2), strides=(2,2), padding='same')(u7)
    u7 = concatenate([u7, p1], axis=3)
    
    u8 = cts_block_up(32, u7)

    lung_seg = Conv2D(1, (1, 1), activation='sigmoid', name='lung_output') (u8) 

    c1, p1 = cts_block_down(32, lung_seg)
    c2, p2 = cts_block_down(64, c1)
    c3, _ = cts_block_down(128, c2, pool_size=(1, 1))
    c4, _ = cts_block_down(256, c3, pool_size=(1, 1))

    u5 = cts_block_up(256, c4)
    u5 = Conv2DTranspose(128, (2, 2), strides=(2,2), padding='same')(u5) 
    
    u6 = cts_block_up(128, u5)
    u6 = Conv2DTranspose(64, (2, 2), padding='same')(u6)
    u6 = concatenate([u6, p2])
    
    u7 = cts_block_up(64, u6)
    u7 = Conv2DTranspose(32, (2, 2), strides=(2,2), padding='same')(u7)
    u7 = concatenate([u7, p1], axis=3)
    
    u8 = cts_block_up(32, u7)
    
    infect_seg = Conv2D(1, (1, 1), activation='sigmoid', name='infect_output') (u8)

    model = Model(inputs=inputs, outputs=[lung_seg, infect_seg], name='cts_model')
    
    return model


# In[23]:


ctsModel = cts_model((TRAIN_IMG_SIZE, TRAIN_IMG_SIZE, TRAIN_IMG_CHANNELS))
ctsModel.summary()


# In[24]:


callbacks = [EarlyStopping(patience=6, monitor='val_loss'), 
             ReduceLROnPlateau(factor=np.sqrt(0.1), patience=2, min_lr=0.5e-6)]


# ## Training

# In[25]:


ctsModel.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

ctsRes = ctsModel.fit(x=X_train, 
                      y={'lung_output': y_lung_train, 'infect_output': y_infection_train}, 
                      batch_size=16, 
                      epochs=100,
                      verbose=1,
                      validation_data=(X_validation, 
                                       {'lung_output': y_lung_validation, 'infect_output': y_infection_validation}),
                      callbacks = callbacks)


# In[26]:


plt.style.use('ggplot')

fig, axes = plt.subplots(1, 2, figsize=(20,10))

axes[0].plot(ctsRes.history['lung_output_accuracy'], color='b', label='acurácia pulmão treinamento')
axes[0].plot(ctsRes.history['infect_output_accuracy'], color='c', label='acurácia infecção treinamento')
axes[0].plot(ctsRes.history['val_lung_output_accuracy'], color='r', label='acurácia pulmão validação')
axes[0].plot(ctsRes.history['val_infect_output_accuracy'], color='m', label='acurácia infecção validação')
axes[0].set_ylabel('Acurácia')
axes[0].set_xlabel('Época')
axes[0].legend()
axes[0].set_ylim([0.5, 1])
axes[0].set_facecolor('w')
axes[0].grid(color='grey', linestyle=':', linewidth=0.5)

axes[1].plot(ctsRes.history['loss'], color='g', label='custo geral')
axes[1].plot(ctsRes.history['lung_output_loss'], color='b', label='custo pulmão treinamento')
axes[1].plot(ctsRes.history['infect_output_loss'], color='c', label='custo infecção treinamento')
axes[1].plot(ctsRes.history['val_loss'], color='y', label='validação geral')
axes[1].plot(ctsRes.history['val_lung_output_loss'], color='r', label='custo pulmão validação')
axes[1].plot(ctsRes.history['val_infect_output_loss'], color='m', label='custo infecção validação')
axes[1].set_ylabel('Custo')
axes[1].set_xlabel('Época')
axes[1].legend()
axes[1].set_ylim([0,1])
axes[1].set_facecolor('w')
axes[1].grid(color='grey', linestyle=':', linewidth=0.5)


# In[27]:


test_result = ctsModel.evaluate(x=X_test, 
                                y={'lung_output': y_lung_test, 'infect_output': y_infection_test},
                                batch_size=32)

print('Test data:\n\t Lung: %.4f loss, %.4f dice coeff\n\t Infection: %.4f loss, %.4f dice coeff' 
      %(test_result[1], test_result[3], test_result[2], test_result[4]))


# In[28]:


y_lung_pred, y_infection_pred = ctsModel.predict(X_test)
y_lung_pred_bin = (y_lung_pred > 0.25).astype(np.uint8)
y_infection_pred_bin = (y_infection_pred > 0.5).astype(np.uint8)


# In[32]:


i = np.random.randint(low=0, high=len(X_test))
fig = plt.figure(figsize=(12,7))

plt.subplot(2,3,1)
plt.imshow(tf.reshape(X_test[i], [TRAIN_IMG_SIZE, TRAIN_IMG_SIZE]), cmap='bone')
plt.axis('off')
plt.title('Tomografia')

plt.subplot(2,3,2)
plt.imshow(tf.reshape(y_lung_test[i], [TRAIN_IMG_SIZE, TRAIN_IMG_SIZE]), cmap='bone')
plt.axis('off')
plt.title('Máscara pulmonar')

plt.subplot(2,3,3)
plt.imshow(tf.reshape(y_infection_test[i], [TRAIN_IMG_SIZE, TRAIN_IMG_SIZE]), cmap='bone')
plt.axis('off')
plt.title('Máscara das lesões')

plt.subplot(2,3,5)
plt.imshow(tf.reshape(y_lung_pred_bin[i], [TRAIN_IMG_SIZE, TRAIN_IMG_SIZE]), cmap='bone')
plt.axis('off')
plt.title('Máscara pulmonar prevista')

plt.subplot(2,3,6)
plt.imshow(tf.reshape(y_infection_pred_bin[i], [TRAIN_IMG_SIZE, TRAIN_IMG_SIZE]), cmap='bone')
plt.axis('off')
plt.title('Máscara das lesões prevista')


# In[30]:


i = np.random.randint(low=0, high=len(X_test), size=4)
fig, axes = plt.subplots(2, 4, figsize=(20,10))

show_overlay_infection(X_test, y_infection_pred_bin, calculate_infection_percent(y_lung_pred[0], y_infection_pred[0]), 0, list(axes[:, 0]))
show_overlay_infection(X_test, y_infection_pred_bin, calculate_infection_percent(y_lung_pred[160], y_infection_pred[160]), 160, list(axes[:, 1]))
show_overlay_infection(X_test, y_infection_pred_bin, calculate_infection_percent(y_lung_pred[313], y_infection_pred[313]), 313, list(axes[:, 2]))
show_overlay_infection(X_test, y_infection_pred_bin, calculate_infection_percent(y_lung_pred[154], y_infection_pred[154]), 154, list(axes[:, 3]))


# In[ ]:




