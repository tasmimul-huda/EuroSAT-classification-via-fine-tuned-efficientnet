from config import CFG
from dataloader import EuroSatLoader
from models import EfficientNetModel, BaselineCNN
from pathlib import Path
from utils import ModelUtils
import tensorflow as tf
import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report


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
        
    predictions = model.predict(test_dataset)
    predicted_label = np.argmax(predictions, axis = 1)

    class_names = ["AnnualCrop","Forest","HerbaceousVegetation","Highway","Industrial",
                   "Pasture","PermanentCrop","Residential","River","SeaLake"]
    
    true_labels = []
    for image, label in test_dataset: #.as_numpy_iterator()
        true_labels += list(label.numpy())
    
    # Plot confusion Matrix
    # ModelUtils.plot_cm(y_true = true_labels, y_pred = predicted_label, class_names = class_names,save_path=cfg.Path.figure_save_path)
    #Plot Roc Auc Curve
    # ModelUtils.multiclass_roc_auc_score(y_true= true_labels, model_predicted_label= predicted_label,class_names = class_names,save_path=cfg.Path.figure_save_path)
    
    report = classification_report(true_labels, predicted_label, target_names=class_names)
    report_dict = classification_report(true_labels, predicted_label, target_names=class_names, output_dict=True, zero_division=0)
    print(report_dict.keys())
    for key in report_dict.keys():
        if isinstance(report_dict[key], dict):
            print(f'{key}: {report_dict[key]["precision"]}')
    # plt.bar(report_dict.keys(), [report_dict[key]['precision'] for key in report_dict.keys()])
    # plt.title('Classification Report')
    # plt.xlabel('Class')
    # plt.ylabel('Precision')
    # plt.show()
    
    # fig, axs = plt.subplots(1, 3, figsize=(12, 6), sharey=True)

    # metrics = ['precision', 'recall', 'f1-score']
    # for i, metric in enumerate(metrics):
    #     axs[i].bar(report_dict.keys(), [report_dict[key][metric] for key in report_dict.keys()])
    #     axs[i].set_title(metric.title())
    #     axs[i].set_xlabel('Class')
        
    # axs[0].set_ylabel('Score')
    # plt.show()
