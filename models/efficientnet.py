import tensorflow as tf

# Efficient net without augmentation4
class EfficientNetModel:
    
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self.build_model()
        
    def build_model(self):
        base_model = tf.keras.applications.efficientnet.EfficientNetB0(input_shape=self.input_shape, include_top=False, weights='imagenet')
        for layer in base_model.layers[:30]:
            layer.trainable = False
        
        x = base_model.output
        
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        predictions = tf.keras.layers.Dense(self.num_classes, activation='softmax')(x)
        model = tf.keras.models.Model(inputs=base_model.input, outputs=predictions)
        return model
     
    def compile(self, learning_rate = 0.0001):
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate =learning_rate), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    def train(self, train_data, val_data, epochs, batch_size, model_save_path, log_dir):
        # self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate =learning_rate), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='min', restore_best_weights=True)
        checkpoint = tf.keras.callbacks.ModelCheckpoint(model_save_path , monitor='val_loss', mode='min', save_weights_only=True,save_best_only=True, verbose=1)
        ReduceLR = tf.keras.callbacks.ReduceLROnPlateau(monitpr = 'val_loss',factor=0.1,patience=3, verbose=1)
        tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True, write_images=True)
        callbacks = [ReduceLR, early_stop, checkpoint, tensorboard]
        history = self.model.fit(train_data, 
                                 epochs=epochs,
                                 batch_size=batch_size, 
                                 validation_data=val_data, 
                                 callbacks=callbacks)
        return history
    def evaluate(self, data):
        loss, accuracy = self.model.evaluate(data)
        return loss, accuracy
    
    def predict(self, data):
        return self.model.predict(data)
    
    def summary(self):
        return self.model.summary()
    
    # def save_model(self, filepath):
    #     self.model.save(filepath)
        
    def load_model(self, model_path):
        self.model.load_weights(model_path)
        
    # def plot_model_architecture(self, file_path):
    #     tf.keras.utils.plot_model(self, to_file=file_path, show_shapes=True)
        
        
class AugmentedEfficientNet:
    
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.data_augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal_and_vertical"),
            tf.keras.layers.RandomContrast(0.2),
            tf.keras.layers.RandomZoom(0.2),
            tf.keras.layers.RandomRotation(factor=0.15),
            tf.keras.layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
            tf.keras.layers.RandomContrast(factor=0.1),
            ],
            name="img_augmentation",
            )
        self.base_model = tf.keras.applications.efficientnet.EfficientNetB2(input_shape=self.input_shape, include_top=False, weights='imagenet')

        self.model = self.build_model()
        
    def build_model(self):
        inputs = tf.keras.layers.Input(shape=self.input_shape)
        x = self.data_augmentation(inputs)
        x = self.base_model(x)
        
        for layer in self.base_model.layers[:40]:  #-20
            # if not isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
                
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        predictions = tf.keras.layers.Dense(self.num_classes, activation='softmax')(x)
        model = tf.keras.models.Model(inputs=inputs, outputs=predictions)
        return model
     
    def compile(self, learning_rate = 0.0001):
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate =learning_rate), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    def train(self, train_data, val_data, epochs, batch_size, model_save_path, log_dir):
        # self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate =learning_rate), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=7, verbose=1, mode='min', restore_best_weights=True)
        checkpoint = tf.keras.callbacks.ModelCheckpoint(model_save_path , monitor='val_loss', mode='min', save_weights_only=True,save_best_only=True, verbose=1)
        ReduceLR = tf.keras.callbacks.ReduceLROnPlateau(monitpr = 'val_loss',factor=0.1,patience=3, verbose=1)
        tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True, write_images=True)
        callbacks = [ReduceLR, early_stop, checkpoint, tensorboard]
        history = self.model.fit(train_data, 
                                 epochs=epochs,
                                 batch_size=batch_size, 
                                 validation_data=val_data, 
                                 callbacks=callbacks)
        return history
    def evaluate(self, data):
        loss, accuracy = self.model.evaluate(data)
        return loss, accuracy
    
    def predict(self, data):
        return self.model.predict(data)
    
    def summary(self):
        return self.model.summary()
    
    # def save_model(self, filepath):
    #     self.model.save(filepath)
        
    def load_model(self, model_path):
        self.model.load_weights(model_path)
        
    # def plot_model_architecture(self, file_path):
    #     tf.keras.utils.plot_model(self, to_file=file_path, show_shapes=True)