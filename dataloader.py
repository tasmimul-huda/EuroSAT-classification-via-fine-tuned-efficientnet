import tensorflow as tf
import pandas as pd
from PIL import Image
import os
import tensorflow_io as tfio

class EuroSatLoader:
    def __init__(self, csv_path, image_folder, batch_size = 32, img_size = (32, 32), buffer_size = 100, shuffle = True):
        self.csv_path = csv_path
        self.image_folder = image_folder
        self.batch_size = batch_size
        self.img_size = img_size
        self.shuffle = shuffle
        self.buffer_size = buffer_size
        
        # Load the CSV files using pandas
        self.df = pd.read_csv(self.csv_path)
        # Create a list of image paths and labels for train, validation, and test datasets
        self.image_paths = [os.path.join(self.image_folder, filename) for filename in self.df.Filename] 
        self.labels = self.df.Label.values
        
    def load_image(self, image_path, label):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3) #
        image = tf.image.resize(image, self.img_size)
        image = tf.cast(image, tf.float32) / 255.0
        return image, label
    
    def get_dataset(self):
        dataset = tf.data.Dataset.from_tensor_slices((self.image_paths, self.labels))
        dataset = dataset.map(self.load_image, num_parallel_calls=tf.data.AUTOTUNE)
        if self.shuffle:
            dataset = dataset.shuffle(buffer_size=self.buffer_size, reshuffle_each_iteration=True)
        dataset = dataset.batch(batch_size=self.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset