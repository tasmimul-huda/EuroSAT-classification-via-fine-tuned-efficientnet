import tensorflow as tf

## Efficient net without augmentation
class EfficientNetModel:
    
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self.build_model()
        
    def build_model(self):
        base_model = tf.keras.applications.efficientnet.EfficientNetB0(input_shape=self.input_shape, include_top=False, weights='imagenet')
        x = base_model.output
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        predictions = tf.keras.layers.Dense(self.num_classes, activation='softmax')(x)
        model = tf.keras.models.Model(inputs=base_model.input, outputs=predictions)
        return model
    
    def train(self, train_data, val_data, epochs, batch_size,learning_rate, model_save_path, log_dir):
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate =learning_rate), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='min', restore_best_weights=True)
        checkpoint = tf.keras.callbacks.ModelCheckpoint(model_save_path + 'effnet_best_model.h5', monitor='val_loss', mode='min', save_weights_only=True,save_best_only=True, verbose=1)
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
