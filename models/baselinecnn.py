import tensorflow as tf
class BaselineCNN:
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self.build_model()
    
    def build_model(self):
        inputs = tf.keras.layers.Input(shape=self.input_shape)
        x = tf.keras.layers.Conv2D(32, (3,3), activation='relu',padding='same')(inputs)
        x = tf.keras.layers.MaxPooling2D((2,2))(x)
        x = tf.keras.layers.Conv2D(64, (3,3), activation='relu',padding='same')(x)
        x = tf.keras.layers.MaxPooling2D((2,2))(x)
        x = tf.keras.layers.Conv2D(128, (3,3), activation='relu',padding='same')(x)
        x = tf.keras.layers.MaxPooling2D((2,2))(x)
        x = tf.keras.layers.Dropout(0.3) (x)
        x = tf.keras.layers.Conv2D(128, (3,3), activation='relu',padding='same')(x)
        x = tf.keras.layers.MaxPooling2D((2,2))(x)
        x = tf.keras.layers.Dropout(0.3) (x)
        x = tf.keras.layers.Conv2D(64, (3,3), activation='relu',padding='same')(x)
        x = tf.keras.layers.MaxPooling2D((2,2))(x)
        x = tf.keras.layers.Conv2D(32, (3,3), activation='relu',padding='same')(x)
        x = tf.keras.layers.MaxPooling2D((2,2))(x)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        outputs = tf.keras.layers.Dense(self.num_classes, activation='softmax')(x)
        model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
        return model
    
    def compile(self, learning_rate = 0.0001):
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate =learning_rate), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    def train(self, train_data, val_data, epochs, batch_size, model_save_path, log_dir):
        # self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate =learning_rate), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='min', restore_best_weights=True)
        checkpoint = tf.keras.callbacks.ModelCheckpoint(model_save_path , monitor='val_loss', mode='min', save_weights_only=True,save_best_only=True, verbose=1)
        tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True, write_images=True)
        callbacks = [early_stop, checkpoint, tensorboard]
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
    
    def save_model(self, filepath):
        self.model.save(filepath)
        
    def load_model(self, model_path):
        self.model.load_weights(model_path)
        
    def plot_model_architecture(self, file_path):
        tf.keras.utilsplot_model(self, to_file=file_path, show_shapes=True)
        
