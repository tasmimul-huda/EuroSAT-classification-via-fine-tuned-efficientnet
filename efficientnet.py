import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.efficientnet import EfficientNetB0

## Efficient net without augmentation
class EfficientNetModel:
    
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self.build_model()
        
    def build_model(self):
        base_model = EfficientNetB0(input_shape=self.input_shape, include_top=False, weights='imagenet')
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        predictions = Dense(self.num_classes, activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=predictions)
        return model
    
    def train(self, train_data, val_data, epochs, batch_size, model_save_path, log_dir):
        self.model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='min', restore_best_weights=True)
        checkpoint = tf.keras.callbacks.ModelCheckpoint(model_save_path + 'best_model.h5', monitor='val_loss', mode='min', save_best_only=True, verbose=1)
        tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True, write_images=True)
        callbacks = [early_stop, checkpoint, tensorboard]
        history = self.model.fit(train_data, 
                                 epochs=epochs,
                                 batch_size=batch_size, 
                                 validation_data=val_data, 
                                 callbacks=callbacks)
        return history
    
    def predict(self, data):
        return self.model.predict(data)
    
    
################# with cutmix data augmantation
# class EfficientNet:
#     def __init__(self, input_shape, num_classes):
#         self.input_shape = input_shape
#         self.num_classes = num_classes
#         self.model = self._build_model()

#     def _build_model(self):
#         inputs = Input(shape=self.input_shape)
#         x = tf.keras.applications.EfficientNetB0(include_top=False, weights=None)(inputs)
#         x = GlobalAveragePooling2D()(x)
#         outputs = Dense(self.num_classes, activation='softmax')(x)
#         model = tf.keras.Model(inputs=inputs, outputs=outputs)
#         return model

#     def _cutmix(self, image, label, crop_size=64, beta=1.0):
#         """Applies CutMix augmentation to the given image and label."""
#         batch_size = tf.shape(image)[0]
#         indices = tf.range(batch_size, dtype=tf.int32)
#         shuffled_indices = tf.random.shuffle(indices)

#         # Generate random bounding box coordinates
#         w, h, c = tf.shape(image)[1], tf.shape(image)[2], tf.shape(image)[3]
#         cx = tf.random.uniform([], 0, w, tf.int32)
#         cy = tf.random.uniform([], 0, h, tf.int32)
#         w_ = tf.cast(w * tf.math.sqrt(1 - beta), tf.int32)
#         h_ = tf.cast(h * tf.math.sqrt(1 - beta), tf.int32)
#         x1 = tf.clip_by_value(cx - w_ // 2, 0, w)
#         y1 = tf.clip_by_value(cy - h_ // 2, 0, h)
#         x2 = tf.clip_by_value(cx + w_ // 2, 0, w)
#         y2 = tf.clip_by_value(cy + h_ // 2, 0, h)

#         # Apply CutMix to the batch
#         new_images = tf.identity(image)
#         for i in range(batch_size):
#             new_images[i, x1[i]:x2[i], y1[i]:y2[i], :] = image[shuffled_indices[i], x1[i]:x2[i], y1[i]:y2[i], :]

#         # Adjust the labels and mix the CutMixed examples
#         new_labels = beta * label + (1 - beta) * tf.gather(label, shuffled_indices)
#         return new_images, new_labels

#     def train(self, train_ds, val_ds, epochs, batch_size, cutmix=True, callbacks=None):
#         if cutmix:
#             train_ds = train_ds.map(self._cutmix, num_parallel_calls=tf.data.AUTOTUNE)

#         self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#         if callbacks is None:
#             callbacks = [EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)]

#         history = self.model.fit(
#             train_ds.prefetch(tf.data.AUTOTUNE),
#             validation_data=val_ds.prefetch(tf.data.AUTOTUNE),
#             epochs=epochs,
#             batch_size=batch_size,
#             callbacks=callbacks
#         )

#         return history
   
#############################
# class CustomCallbacks:
#     def __init__(self, model_save_path, tensorboard_logs_path):
#         self.model_save_path = model_save_path
#         self.tensorboard_logs_path = tensorboard_logs_path

#     def get_callbacks(self):
#         callbacks = [
#             EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
#             ModelCheckpoint(self.model_save_path, save_best_only=True),
#             TensorBoard(self.tensorboard_logs_path)
#         ]
#         return callbacks
# class EfficientNet:
#     def __init__(self, input_shape, num_classes):
#         self.input_shape = input_shape
#         self.num_classes = num_classes
#         self.model = self._build_model()

#     def _build_model(self):
#         inputs = Input(shape=self.input_shape)
#         x = tf.keras.applications.EfficientNetB0(include_top=False, weights=None)(inputs)
#         x = GlobalAveragePooling2D()(x)
#         outputs = Dense(self.num_classes, activation='softmax')(x)
#         model = tf.keras.Model(inputs=inputs, outputs=outputs)
#         return model

#     def train(self, train_ds, val_ds, epochs, batch_size, cutmix_prob=0.5, alpha=1.0, callbacks=None):
#         def cutmix(image_batch, label_batch):
#             # implementation of cutmix function here...

#         data_augmentation = tf.keras.Sequential([
#             layers.experimental.preprocessing.RandomFlip("horizontal"),
#             layers.experimental.preprocessing.RandomRotation(0.1),
#         ])

#         if callbacks is None:
#             callbacks = CustomCallbacks('path/to/save/model.h5', 'path/to/save/tensorboard/logs').get_callbacks()

#         self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#         history = self.model.fit(
#             train_ds.map(lambda x, y: (data_augmentation(x), y)).map(lambda x, y: cutmix(x, y) if tf.random.uniform([]) < cutmix_prob else (x, y)).prefetch(tf.data.AUTOTUNE),
#             validation_data=val_ds.prefetch(tf.data.AUTOTUNE),
#             epochs=epochs,
#             batch_size=batch_size,
#             callbacks=callbacks
#         )

#         return history
    
######################################################################3
import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.callbacks import *

class EfficientNet:
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self._build_model()

    def _build_model(self):
        inputs = Input(shape=self.input_shape)
        x = tf.keras.applications.EfficientNetB0(include_top=False, weights=None)(inputs)
        x = GlobalAveragePooling2D()(x)
        outputs = Dense(self.num_classes, activation='softmax')(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        return model

    def train(self, train_ds, val_ds, epochs, batch_size, data_augmentation=None, callbacks=None):
        if data_augmentation is None:
            data_augmentation = tf.keras.Sequential([
                layers.experimental.preprocessing.RandomFlip("horizontal"),
                layers.experimental.preprocessing.RandomRotation(0.1),
            ])
        if callbacks is None:
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
                ModelCheckpoint('path/to/save/model.h5', save_best_only=True),
                TensorBoard('path/to/save/tensorboard/logs')
            ]

        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        history = self.model.fit(
            train_ds.map(lambda x, y: (data_augmentation(x), y)).prefetch(tf.data.AUTOTUNE),
            validation_data=val_ds.prefetch(tf.data.AUTOTUNE),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks
        )

        return history
####################################################################    
    

""" 
from tensorflow.keras.layers.experimental.preprocessing import RandomFlip, RandomRotation
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers, models

class EfficientNetModel(tf.keras.Model):
    
    def __init__(self, num_classes):
        super(EfficientNetModel, self).__init__()
        self.num_classes = num_classes
    
    def build_model(self):
        inputs = tf.keras.Input(shape=(224, 224, 3))
        x = RandomFlip()(inputs)
        x = RandomRotation(0.2)(x)
        base_model = EfficientNetB0(include_top=False, input_tensor=x, weights='imagenet')
        x = layers.GlobalAveragePooling2D()(base_model.output)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        self.backbone = models.Model(inputs, base_model.output)
        self.pooling = layers.GlobalAveragePooling2D()
        self.dropout = layers.Dropout(0.5)
        self.fc = layers.Dense(self.num_classes, activation='softmax')
    
    def compile_model(self):
        self.build_model()
        optimizer = tf.keras.optimizers.Adam(lr=0.001)
        self.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

"""

'''
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import Model

class EfficientNetModel(Model):
    
    def __init__(self, num_classes):
        super(EfficientNetModel, self).__init__()
        self.num_classes = num_classes
        self.base_model = EfficientNetB0(include_top=False, weights='imagenet')
        self.global_pooling = GlobalAveragePooling2D()
        self.dropout = Dropout(0.5)
        self.classifier = Dense(num_classes, activation='softmax')
        
    def call(self, inputs):
        x = inputs
        x = self.base_model(x)
        x = self.global_pooling(x)
        x = self.dropout(x)
        outputs = self.classifier(x)
        return outputs
    
    def compile_model(self):
        optimizer = tf.keras.optimizers.Adam(lr=0.001)
        self.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

'''

'''
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.efficientnet import EfficientNetB0

class EfficientNetModel:
    
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self.build_model()
        
    def build_model(self):
        base_model = EfficientNetB0(input_shape=self.input_shape, include_top=False, weights='imagenet')
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        predictions = Dense(self.num_classes, activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=predictions)
        return model
    
    def train(self, train_data, val_data, epochs, batch_size):
        self.model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
        early_stop = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='min', restore_best_weights=True)
        checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', save_best_only=True, verbose=1)
        tensorboard = TensorBoard(log_dir='logs', histogram_freq=1, write_graph=True, write_images=True)
        callbacks = [early_stop, checkpoint, tensorboard]
        self.model.fit(train_data, epochs=epochs, batch_size=batch_size, validation_data=val_data, callbacks=callbacks)
    
    def predict(self, data):
        return self.model.predict(data)

'''

'''4
import tensorflow as tf

class AugmentationLayer(tf.keras.layers.Layer):
    
    def __init__(self, rotation_range=0, width_shift_range=0., height_shift_range=0., horizontal_flip=False):
        super(AugmentationLayer, self).__init__()
        self.datagen = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=rotation_range,
                                                                       width_shift_range=width_shift_range,
                                                                       height_shift_range=height_shift_range,
                                                                       horizontal_flip=horizontal_flip)
    
    def call(self, inputs, training=None):
        if training:
            inputs = tf.stack([self.datagen.random_transform(image.numpy()) for image in inputs])
        return inputs

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.efficientnet import EfficientNetB0

class EfficientNetModel:
    
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self.build_model()
        
    def build_model(self):
        input_layer = tf.keras.layers.Input(shape=self.input_shape)
        x = AugmentationLayer(rotation_range=20, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)(input_layer)
        base_model = EfficientNetB0(include_top=False, weights='imagenet')
        x = base_model(x, training=False)
        x = GlobalAveragePooling2D()(x)
        output_layer = Dense(self.num_classes, activation='softmax')(x)
        model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)
        return model
    
    def train(self, train_data, val_data, epochs, batch_size):
        self.model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
        early_stop = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='min', restore_best_weights=True)
        checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', save_best_only=True, verbose=1)
        tensorboard = TensorBoard(log_dir='logs', histogram_freq=1, write_graph=True, write_images=True)
        callbacks = [early_stop, checkpoint, tensorboard]
        self.model.fit(train_data, epochs=epochs, batch_size=batch_size, validation_data=val_data, callbacks=callbacks)
    
    def predict(self, data):
        return self.model.predict(data)


'''

'''
def train(self, train_data, val_data, epochs, batch_size):
    self.model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    
    # define data augmentation pipeline
    def augment_data(image, label):
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, max_delta=0.1)
        image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
        return image, label
    
    # apply data augmentation to training dataset
    train_data = train_data.map(augment_data)
    
    early_stop = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='min', restore_best_weights=True)
    checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', save_best_only=True, verbose=1)
    tensorboard = TensorBoard(log_dir='logs', histogram_freq=1, write_graph=True, write_images=True)
    callbacks = [early_stop, checkpoint, tensorboard]
    
    self.model.fit(train_data, epochs=epochs, batch_size=batch_size, validation_data=val_data, callbacks=callbacks)

'''

'''
# for cutmix
def augment_data(images, labels, alpha=1.0):
    # apply CutMix to a batch of images and labels
    batch_size = tf.shape(images)[0]
    indices = tf.range(batch_size, dtype=tf.int32)
    shuffled_indices = tf.random.shuffle(indices)
    lam = tfd.Beta(alpha, alpha).sample()
    cutmix_images = lam * images + (1 - lam) * tf.gather(images, shuffled_indices)
    cutmix_labels = lam * labels + (1 - lam) * tf.gather(labels, shuffled_indices)
    return cutmix_images, cutmix_labels

cutmix_image = lam * image_i + (1 - lam) * image_j
cutmix_label = lam * label_i + (1 - lam) * label_j


def train(self, train_data, val_data, epochs, batch_size):
    self.model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    
    # apply CutMix to training dataset
    train_data = train_data.map(lambda x, y: augment_data(x, y, alpha=1.0))
    
    early_stop = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='min', restore_best_weights=True)
    checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', save_best_only=True, verbose=1)
    tensorboard = TensorBoard(log_dir='logs', histogram_freq=1, write_graph=True, write_images=True)
    callbacks = [early_stop, checkpoint, tensorboard]
    
    self.model.fit(train_data, epochs=epochs, batch_size=batch_size, validation_data=val_data, callbacks=callbacks)

def cross_entropy_with_mixing(y_true, y_pred, lam):
    loss = lam * tf.keras.losses.categorical_crossentropy(y_true, y_pred) \
           + (1 - lam) * tf.keras.losses.categorical_crossentropy(tf.gather(y_true, tf.random.shuffle(tf.range(tf.shape(y_true)[0]))), y_pred)
    return loss

def cross_entropy_with_mixing(y_true, y_pred, lam):
    loss = lam * tf.keras.losses.categorical_crossentropy(y_true, y_pred) \
           + (1 - lam) * tf.keras.losses.categorical_crossentropy(tf.gather(y_true, tf.random.shuffle(tf.range(tf.shape(y_true)[0]))), y_pred)
    return loss


'''