from config import CFG
from dataloader import EuroSatLoader
from models import BaselineCNN, AugmentedBaselineCNN
from models import EfficientNetModel
from pathlib import Path
from utils import ModelUtils
import tensorflow as tf
import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings('ignore')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'



def get_model(model_type, input_shape, num_classes):
    if model_type == 'baseline':
        model = BaselineCNN(input_shape, num_classes)
    elif model_type == 'augmented_baseline':
        model = AugmentedBaselineCNN(input_shape, num_classes)
    else:
        raise ValueError('Invalid model type')
        
    return model

if __name__ == '__main__':
    cfg = CFG()
    Path(cfg.Path.model_save_path).mkdir(parents = True, exist_ok = True)
    Path(cfg.Path.tensorboard_logs_path).mkdir(parents = True, exist_ok = True)
    Path(cfg.Path.figure_save_path).mkdir(parents = True, exist_ok = True)

    train_loader = EuroSatLoader(csv_path = cfg.Path.train_df, 
                                image_folder = cfg.Path.image_folder,
                                batch_size = cfg.Dataset.batch_size,
                                img_size = cfg.Dataset.img_size,
                                buffer_size = cfg.Dataset.buffer_size, 
                                shuffle = True)
    train_dataset = train_loader.get_dataset()


    val_loader = EuroSatLoader(csv_path = cfg.Path.val_df, 
                                image_folder = cfg.Path.image_folder,
                                batch_size = cfg.Dataset.batch_size,
                                img_size = cfg.Dataset.img_size,
                                buffer_size = cfg.Dataset.buffer_size, 
                                shuffle = False)
    val_dataset = val_loader.get_dataset()
    test_loader = EuroSatLoader(csv_path = cfg.Path.test_df, 
                                image_folder = cfg.Path.image_folder,
                                batch_size = cfg.Dataset.batch_size,
                                img_size = cfg.Dataset.img_size,
                                buffer_size = cfg.Dataset.buffer_size, 
                                shuffle = False)
    test_dataset = test_loader.get_dataset()
    
    model = get_model(model_type = cfg.model_type, 
                      input_shape = cfg.HyperParameter.input_shape, 
                      num_classes = cfg.HyperParameter.num_classes)
    
    print(f"INFO ===========Training Started===============")
    model.compile(learning_rate= cfg.HyperParameter.learning_rate)
    history = model.train(train_data = train_dataset, 
                          val_data = val_dataset,
                          epochs = cfg.HyperParameter.epochs,
                          batch_size= cfg.HyperParameter.batch_size, 
                          model_save_path= cfg.Path.model_save_path,
                          log_dir= cfg.Path.tensorboard_logs_path)
    print(f"INFO ===========Training Finished===============")
    print(f"INFO ===========Plot Curve===============")
    ModelUtils.plot_loss_accuracy(cfg.Path.loss_acc_save_path,history)
    
    model.load_model('D:/EuroSAT/weights/best_model_2.h5')
    
    train_loss, train_acc = model.evaluate(train_dataset)
    print(f"train_loss:{train_loss}:: train_accuracy: {train_acc}")
    
    val_loss, val_acc = model.evaluate(val_dataset)
    print(f"val_loss:{val_loss}:: val_acc: {val_acc}")
    
    test_loss, test_acc = model.evaluate(test_dataset)
    print(f"test_loss:{test_loss}:: test_acc: {test_acc}")
        
    predictions = model.predict(test_dataset)
    predicted_label = np.argmax(predictions, axis = 1)

    class_names = ["AnnualCrop","Forest","HerbaceousVegetation","Highway","Industrial",
                   "Pasture","PermanentCrop","Residential","River","SeaLake"]
    
    true_labels = []
    for image, label in test_dataset:
        true_labels += list(label.numpy())
    
    # Plot confusion Matrix
    ModelUtils.plot_cm(y_true = true_labels, y_pred = predicted_label, class_names = class_names,save_path=cfg.Path.cm_save_path)
    #Plot Roc Auc Curve
    ModelUtils.multiclass_roc_auc_score(y_true= true_labels, model_predicted_label= predicted_label,class_names = class_names,save_path=cfg.Path.roc_save_path)

    # print("Train Dataset")
    # for i, (x, y) in enumerate(train_dataset):
    #     print(x.shape, y.shape)
    #     if i == 10:
    #         break