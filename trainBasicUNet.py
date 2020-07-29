import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow.keras import backend
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from kerastuner.engine import hypermodel
from keras.callbacks import ModelCheckpoint
from utils import *
import tqdm
from sklearn.model_selection import train_test_split

from keras_unet.models import custom_unet
from keras_unet.losses import bce_dice_loss

from keras.optimizers import Adam
from keras_unet.metrics import iou, iou_thresholded

import numpy as np 

#Upload and prepare data 
def main():
    #trainiing settings 

    #Multi GPU strategy
    strategy = tf.distribute.MirroredStrategy() 
    
    #load dataset
    image_paths,mask_paths = load_images("Data/lgg-mri-segmentation/kaggle_3m/")

    #check to see if each image has a mask if so, read in the images
    if image_mask_check(image_paths, mask_paths):
        masks,images = read_in_images(image_paths,mask_paths)
    
    #reshape mask image to (256,256) -- binarize
    for i in range(0,len(masks)):
        m = masks[i]
        m = m[:,:,0]
        m.reshape(m.shape[0],m.shape[1])
        masks[i] = m

    #make images and masks into numpy arrays and normalize the masks
    images = np.asarray(images)
    masks = np.asarray(masks)
    masks = masks / 255
    masks = masks.reshape(masks.shape[0],masks.shape[1],masks.shape[2],1)


    #split the data
    img_overall_train, img_test, mask_overall_train, mask_test = train_test_split(images, masks, test_size=0.16667, random_state=42)
    img_train, img_val, mask_train, mask_val = train_test_split(img_overall_train, mask_overall_train, test_size = 0.166667, random_state = 32)
   
    #create data generator for running model
    #batch size of 16
    train_datagen = ImageDataGenerator()

    train_generator = train_datagen.flow(
        img_train, mask_train,
        batch_size=16)

    val_generator = train_datagen.flow(
        img_val, mask_val)


    #steps within each epoch is the total number of training images divided by the batch size
    STEPS_PER_EPOCH = len(img_train) // 16
    
    
    #Get U Net

    input_shape = img_train[0].shape

    #allow model to be trained over multiple GPUs
    with strategy.scope():
        
    #load in the custom u net from the models section 
        model = custom_unet(
            input_shape,
            filters=32,
            use_batch_norm=True,
            dropout=0.3,
            dropout_change_per_layer=0.0,
            num_layers=4
        )

        print(model.summary())

        ##Compile and Train

        #save each model if validation loss improves
        model_filename = 'UNet_Model'
        callback_checkpoint = ModelCheckpoint(
            model_filename, 
            verbose=1, 
            monitor='val_loss', 
            save_best_only=True,
        )

        opt = keras.optimizers.Adam(learning_rate=0.001)
        #set compile and run for model
    
        model.compile(
            optimizer=opt, 
            #loss='binary_crossentropy',
            loss = bce_dice_loss,
            metrics=[iou, iou_thresholded]
        )

        #use fit_generator because using data generator
        #fit the model for 
        history = model.fit_generator(
            train_generator,
            steps_per_epoch=STEPS_PER_EPOCH,
            epochs=300,
            validation_data=(img_val, mask_val),
            callbacks=[callback_checkpoint]
    )

    # serialize model to JSON
    #save model as JSON and weight file
    model_json = model.to_json()
    with open("UNetModel.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model.h5")
    print("Saved model to disk")
    

if __name__ == '__main__':
    main()
