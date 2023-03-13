from config import CFG
from dataloader import EuroSatLoader
from models import EfficientNetModel, BaselineCNN
from pathlib import Path
from utils import ModelUtils
import tensorflow as tf
import numpy as np
import json


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

    # print("Train Dataset")
    # for i, (x, y) in enumerate(train_dataset):
    #     print(x.shape, y.shape)
    #     if i == 10:
    #         break

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
    
    model = BaselineCNN(input_shape = cfg.HyperParameter.input_shape, 
                        num_classes = cfg.HyperParameter.num_classes)
    
    # print(f"INFO ===========Training Started===============")
    model.compile()
    # history = model.train(train_data = train_dataset, 
    #                       val_data = val_dataset,
    #                       epochs = cfg.HyperParameter.epochs,
    #                       batch_size= cfg.HyperParameter.batch_size, 
    #                       learning_rate= cfg.HyperParameter.learning_rate,
    #                       model_save_path= cfg.Path.model_save_path,
    #                       log_dir= cfg.Path.tensorboard_logs_path)
    # print(f"INFO ===========Training Finished===============")
    # print(f"INFO ===========Plot Curve===============")
    # ModelUtils.plot_loss_accuracy(cfg.Path.figure_save_path,history)
    
    model.load_model('D:/EuroSAT/weights/best_model.h5')
    
    # train_loss, train_acc = model.evaluate(train_dataset)
    # print(f"train_loss:{train_loss}:: train_accuracy: {train_acc}")
    
    # val_loss, val_acc = model.evaluate(val_dataset)
    # print(f"val_loss:{val_loss}:: val_acc: {val_acc}")
    
    # test_loss, test_acc = model.evaluate(test_dataset)
    # print(f"test_loss:{test_loss}:: test_acc: {test_acc}")
    
    # for image, label in test_dataset.as_numpy_iterator():
    #     print(image, label)
        
    predictions = model.predict(test_dataset)
    predicted_label = np.argmax(predictions, axis = 1)
    # # print(predictions)
    class_names = ["AnnualCrop","Forest","HerbaceousVegetation","Highway","Industrial",
                   "Pasture","PermanentCrop","Residential","River","SeaLake"]
    true_labels = []
    # print('predicted_label',len(predicted_label))
    for image, label in test_dataset: #.as_numpy_iterator()
        true_labels += list(label.numpy())
        # print(list(label.numpy()))
    print('true_labels', len(true_labels))
        
    ModelUtils.plot_cm(y_true = true_labels, y_pred = predicted_label, class_names = class_names,save_path=cfg.Path.figure_save_path)
