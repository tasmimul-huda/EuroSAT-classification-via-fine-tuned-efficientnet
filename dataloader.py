import tensorflow as tf
import pandas as pd
from PIL import Image
import os

class EuroSatLoader:
    def __init__(self, csv_path, image_folder, batch_size = 32, img_size = (32, 32), buffer_size = 100, shuffle = True):
        self.csv_path = csv_path
        self.image_folder = image_folder
        self.batch_size = batch_size
        self.img_size = img_size
        self.shuffle = shuffle
        self.buffer_size = buffer_size
        
        self.df = pd.read_csv(self.csv_path)
        
        self.image_paths = [os.path.join(self.image_folder, filename) for filename in self.df.Filename] 
        self.labels = self.df.Label.values
        
    def load_image(self, image_path, label):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
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
    
    
    
    
    
    
    
    
    
    
    
'''
class ImageDataLoader:
    def __init__(self, train_csv_path, val_csv_path, test_csv_path, image_folder, batch_size=32, img_size=(224, 224), shuffle=True, buffer_size=10000):
        self.train_csv_path = train_csv_path
        self.val_csv_path = val_csv_path
        self.test_csv_path = test_csv_path
        self.image_folder = image_folder
        self.batch_size = batch_size
        self.img_size = img_size
        self.shuffle = shuffle
        self.buffer_size = buffer_size
        
        # Load the CSV files using pandas
        self.train_df = pd.read_csv(self.train_csv_path)
        self.val_df = pd.read_csv(self.val_csv_path)
        self.test_df = pd.read_csv(self.test_csv_path)
        
        # Create a list of image paths and labels for train, validation, and test datasets
        self.train_image_paths = [os.path.join(self.image_folder, class_name, path) for class_name, path in zip(self.train_df["class"], self.train_df["image_path"])]
        self.train_labels = self.train_df["label"].values
        self.val_image_paths = [os.path.join(self.image_folder, class_name, path) for class_name, path in zip(self.val_df["class"], self.val_df["image_path"])]
        self.val_labels = self.val_df["label"].values
        self.test_image_paths = [os.path.join(self.image_folder, class_name, path) for class_name, path in zip(self.test_df["class"], self.test_df["image_path"])]
        self.test_labels = self.test_df["label"].values
        
        # Create TensorFlow datasets using the image paths and labels for train, validation, and test datasets
        self.train_dataset = tf.data.Dataset.from_tensor_slices((self.train_image_paths, self.train_labels))
        self.val_dataset = tf.data.Dataset.from_tensor_slices((self.val_image_paths, self.val_labels))
        self.test_dataset = tf.data.Dataset.from_tensor_slices((self.test_image_paths, self.test_labels))
        
        # Use the map function to apply the load_image function to each element of the train, validation, and test datasets
        self.train_dataset = self.train_dataset.map(self.load_image)
        self.val_dataset = self.val_dataset.map(self.load_image)
        self.test_dataset = self.test_dataset.map(self.load_image)
        
        # Shuffle and batch the train dataset
        if self.shuffle:
            self.train_dataset = self.train_dataset.shuffle(buffer_size=self.buffer_size)
        self.train_dataset = self.train_dataset.batch(batch_size=self.batch_size)
        
        # Batch the validation and test datasets
        self.val_dataset = self.val_dataset.batch(batch_size=self.batch_size)
        self.test_dataset = self.test_dataset.batch(batch_size=self.batch_size)
        
        # Prefetch the datasets for faster training
        self.train_dataset = self.train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        self.val_dataset = self.val_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        self.test_dataset = self.test_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    
    # Define a function to load an image file given its path
    def load_image(self, image_path, label):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, self.image_size)
        image = tf.cast(image, tf.float32) / 255.0
        return image, label
        
'''







'''
class DataLoader:
    def __init__(self, csv_file, img_folder, image_size, batch_size):
        self.img_folder = img_folder
        self.image_paths, self.labels = self._load_data(csv_file)
        self.image_size = image_size
        self.batch_size = batch_size

    def _load_data(self, csv_file):
        data = pd.read_csv(csv_file)
        image_paths = data['Filename'].tolist()
        labels = data['Label'].tolist()
        return image_paths, labels

    def _process_image(self, image_path, label):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, self.image_size)
        image = tf.cast(image, tf.float32) / 255.0
        return image, label

    def get_dataset(self):
        dataset = tf.data.Dataset.from_tensor_slices((self.image_paths, self.labels))
        dataset = dataset.map(self._process_image)
        dataset = dataset.batch(self.batch_size)
        return dataset
'''