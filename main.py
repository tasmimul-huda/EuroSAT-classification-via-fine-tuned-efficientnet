from config import CFG
from dataloader import EuroSatLoader
from models import BaselineCNN, AugmentedBaselineCNN
from models import EfficientNetModel, AugmentedEfficientNet
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
    print(f"Training started with {model_type}")
    if model_type == 'baseline':
        model = BaselineCNN(input_shape, num_classes)
    elif model_type == 'augmented_baseline':
        model = AugmentedBaselineCNN(input_shape, num_classes)
    elif model_type == 'efficientnet':
        model = EfficientNetModel(input_shape, num_classes)
    elif model_type == 'augmented_efficientnet':
        model = AugmentedEfficientNet(input_shape, num_classes)
    else:
        raise ValueError('Invalid model type')
        
    return model

if __name__ == '__main__':
    cfg = CFG()
    Path(cfg.Path.saved_model_folder).mkdir(parents = True, exist_ok = True)
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
    
    print(model.summary())
    
    if os.path.isfile(cfg.Path.model_save_path):
        print("INFO ===========Running the Partially Trained Model===============")
        #This code is implemented to load the partly trained model which was stopped due to some reason
        model.load_model(cfg.Path.model_save_path)
        history = model.train(train_data = train_dataset, 
                          val_data = val_dataset,
                          epochs = cfg.HyperParameter.epochs,
                          batch_size= cfg.HyperParameter.batch_size, 
                          model_save_path= cfg.Path.model_save_path,
                          log_dir= cfg.Path.tensorboard_logs_path)
    else:
        print("INFO ===========Running the Training of Model from Scratch===============")
        # model.compile(learning_rate= cfg.HyperParameter.learning_rate)
        history = model.train(train_data = train_dataset, 
                          val_data = val_dataset,
                          epochs = cfg.HyperParameter.epochs,
                          batch_size= cfg.HyperParameter.batch_size, 
                          model_save_path= cfg.Path.model_save_path,
                          log_dir= cfg.Path.tensorboard_logs_path)
        
    print(f"INFO ===========Training Finished===============")
    print(f"INFO ===========Plot Curve===============")
    ModelUtils.plot_loss_accuracy(cfg.Path.loss_acc_save_path,history)
    
    model.load_model(cfg.Path.model_save_path)
    
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
    
    
'''
INFO ===========Training Finished=============== BaselineCNN
INFO ===========Plot Curve===============
148/148 [==============================] - 49s 328ms/step - loss: 0.1734 - accuracy: 0.9451
train_loss:0.17337167263031006:: train_accuracy: 0.9450793862342834
43/43 [==============================] - 14s 311ms/step - loss: 0.2517 - accuracy: 0.9157
val_loss:0.2517469525337219:: val_acc: 0.9157407283782959
22/22 [==============================] - 6s 281ms/step - loss: 0.2454 - accuracy: 0.9189
test_loss:0.24541887640953064:: test_acc: 0.9188888669013977
22/22 [==============================] - 7s 287ms/step
    
    '''
    
"""  
    INFO ===========Plot Curve=============== augment baseline
296/296 [==============================] - 92s 310ms/step - loss: 0.2587 - accuracy: 0.9079
train_loss:0.2587067484855652:: train_accuracy: 0.9079365134239197
85/85 [==============================] - 27s 319ms/step - loss: 0.2856 - accuracy: 0.8961
val_loss:0.28564929962158203:: val_acc: 0.8961111307144165
43/43 [==============================] - 25s 564ms/step - loss: 0.2576 - accuracy: 0.9063
test_loss:0.2576492726802826:: test_acc: 0.9062963128089905
43/43 [==============================] - 14s 311ms/step


=======================Efficient net B1

296/296 [==============================] - 12s 38ms/step - loss: 0.0354 - accuracy: 0.9920
train_loss:0.03542402386665344:: train_accuracy: 0.9920105934143066
85/85 [==============================] - 3s 33ms/step - loss: 0.3069 - accuracy: 0.9033
val_loss:0.3068503737449646:: val_acc: 0.903333306312561
43/43 [==============================] - 2s 56ms/step - loss: 0.2612 - accuracy: 0.9104
test_loss:0.26120659708976746:: test_acc: 0.9103703498840332
43/43 [==============================] - 3s 32ms/step



###EFFb2
296/296 [==============================] - 11s 35ms/step - loss: 0.0105 - accuracy: 0.9977
train_loss:0.010462253354489803:: train_accuracy: 0.9976719617843628
85/85 [==============================] - 3s 29ms/step - loss: 0.2332 - accuracy: 0.9302
val_loss:0.23319227993488312:: val_acc: 0.9301851987838745
43/43 [==============================] - 3s 61ms/step - loss: 0.2064 - accuracy: 0.9352
test_loss:0.20635396242141724:: test_acc: 0.9351851940155029
43/43 [==============================] - 3s 28ms/step
"""