import os
import shutil
import time
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from skimage import img_as_ubyte
from random import shuffle
from keras.regularizers import l2
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, Input, AveragePooling2D
from keras.optimizers import RMSprop, SGD
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping, LearningRateScheduler

# Create train/val/test dataset
# val set is 20% of train's
data_prepared = False # set this to True when train/val/test dataset NOT exist
prefix = '../dataset/food-101/'
train_meta_path = prefix + 'meta/train.txt'
test_meta_path = prefix + 'meta/test.txt'
train_image_path = prefix + 'train/'
val_image_path = prefix + 'val/'
test_image_path = prefix + 'test/'
if not os.path.exists(prefix+'train'):
    data_prepared = True

t1 = time.time()
if data_prepared:
    print 'Preparing dataset...'
    with open(train_meta_path, mode='r') as f:
        lines = f.readlines()
        lines = [l.strip('\n') for l in lines]
        shuffle(lines)
        percentage = int(len(lines)*0.8)
        train_names = lines[:percentage]
        val_names = lines[percentage:]
        with open(prefix+'gen_train_meta.txt', mode='w') as f:
            for name in train_names:
                f.write('%s\n'%name)
        with open(prefix+'gen_val_meta.txt', mode='w') as f:
            for name in val_names:
                f.write('%s\n'%name)
        for i, l in enumerate(train_names):
            class_names, _ = l.split('/')
            if not os.path.exists(train_image_path+class_names):
                os.makedirs(train_image_path+class_names)
            shutil.copy(prefix + 'images/' + l + '.jpg', train_image_path+class_names)

        for i, l in enumerate(val_names):
            class_names, _ = l.split('/')
            if not os.path.exists(val_image_path+class_names):
                os.makedirs(val_image_path+class_names)
            shutil.copy(prefix + 'images/' + l + '.jpg', val_image_path+class_names)


    with open(test_meta_path, mode='r') as f:
        lines = f.readlines()
        lines = [l.strip('\n') for l in lines]
        shuffle(lines)
        with open(prefix+'gen_test_meta.txt', mode='w') as f:
            for name in lines:
                f.write('%s\n'%name)
        for i, l in enumerate(lines):
            class_names, _ = l.split('/')
            if not os.path.exists(test_image_path+class_names):
                os.makedirs(test_image_path+class_names)
            shutil.copy(prefix + 'images/' + l + '.jpg', test_image_path+class_names)

    t2 = time.time()
    print 'Time for creating dataset: %s s'%(t2 - t1)
else:
    print 'Dataset exists, skip preparing dataset.'


# Training configuration
height, width, channels = 224, 224, 3
num_classes = 101
batch_sz = 4
num_epoches = 32
num_train_samples = 60600
num_val_samples = 15150
num_test_samples = 25250


print 'Building model...'
base_model = InceptionV3(weights='imagenet', include_top=False, input_tensor=Input(shape=(224, 224, 3)))
x = base_model.output
x = AveragePooling2D(pool_size=(5, 5))(x)
x = Dropout(.4)(x)
x = Flatten()(x)
predictions = Dense(num_classes, init='glorot_uniform', W_regularizer=l2(.0005), activation='softmax')(x)

model = Model(input=base_model.input, output=predictions)

opt = SGD(lr=.01, momentum=.9)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

checkpointer = ModelCheckpoint(filepath='model-{val_loss:.2f}.hdf5', verbose=1, save_best_only=True)
csv_logger = CSVLogger('training_log.txt')

def schedule(epoch):
    if epoch < 15:
        return .01
    elif epoch < 28:
        return .002
    else:
        return .0004
lr_scheduler = LearningRateScheduler(schedule)

train_datagen = ImageDataGenerator(
    width_shift_range=0.1,
    height_shift_range=0.1,
    rescale=1./255,
    zoom_range=0.1,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(
    rescale=1./255
)

train_generator = train_datagen.flow_from_directory(
    train_image_path,
    target_size=(height, width),
    class_mode='categorical',
    batch_size=batch_sz
)

val_generator = val_datagen.flow_from_directory(
    val_image_path,
    target_size=(height, width),
    class_mode='categorical',
    batch_size=batch_sz
)

print 'Training...'
model.fit_generator(train_generator,
                    validation_data=val_generator,
                    steps_per_epoch = num_train_samples // batch_sz,
                    validation_steps = num_val_samples // batch_sz,
                    epochs=num_epoches,
                    verbose=1,
                    callbacks=[lr_scheduler, csv_logger, checkpointer])

print 'Testing...'
test_datagen = ImageDataGenerator(
    rescale=1./255
)

test_generator = test_datagen.flow_from_directory(
    test_image_path,
    target_size=(height, width),
    class_mode='categorical',
    batch_size=batch_sz
)

results = model.evaluate_generator(test_generator, num_test_samples // batch_sz)
print 'Testing resluts:', resluts
