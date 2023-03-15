class CFG:  
    model_type = "efficientnet" # augmented_baseline,baseline, efficientnet
    # path defination
    class Path:
        model_type = "effnet"
        
        train_df = 'D:/EuroSAT/EuroSAT/train.csv'
        val_df = 'D:/EuroSAT/EuroSAT/validation.csv'
        test_df = 'D:/EuroSAT/EuroSAT/test.csv'
        image_folder = 'D:/EuroSAT/EuroSAT/'
        saved_model_folder = 'D:/EuroSAT/weights/'
        model_save_path = f'D:/EuroSAT/weights/{model_type}_best_model.h5'
        tensorboard_logs_path = "D:/EuroSAT/tensorboard/logs/"
        figure_save_path = "D:/EuroSAT/Figures/"
        
        model_architecture_path = f"D:/EuroSAT/Figures/{model_type}_.png"
        
        cm_save_path = f"D:/EuroSAT/Figures/{model_type}_confusion_matrix.png"
        roc_save_path = f"D:/EuroSAT/Figures/{model_type}_roc_auc.png"
        loss_acc_save_path = f"D:/EuroSAT/Figures/{model_type}_loss_acc_curve.png"
    # dataset parameter
    class Dataset:
        batch_size = 64
        img_size = (64, 64)
        buffer_size = 1000
    
    class HyperParameter:
        batch_size = 64
        learning_rate = 0.00003
        input_shape = (64, 64, 3)
        num_classes = 10
        epochs = 100
        

class CFGTIFF:  
    model_type = "allband_baseline" # augmented_baseline,baseline, efficientnet
    # path defination
    class Path:
        model_type = "allband_baseline"
        
        train_df = 'D:/EuroSAT/EuroSATallBands/train.csv'
        val_df = 'D:/EuroSAT/EuroSATallBands/validation.csv'
        test_df = 'D:/EuroSAT/EuroSATallBands/test.csv'
        image_folder = 'D:/EuroSAT/EuroSATallBands/'
        saved_model_folder = 'D:/EuroSAT/weights/'
        model_save_path = f'D:/EuroSAT/weights/{model_type}_best_model.h5'
        tensorboard_logs_path = "D:/EuroSAT/tensorboard/logs/"
        figure_save_path = "D:/EuroSAT/Figures/"
        
        model_architecture_path = f"D:/EuroSAT/Figures/{model_type}_.png"
        
        cm_save_path = f"D:/EuroSAT/Figures/{model_type}_confusion_matrix.png"
        roc_save_path = f"D:/EuroSAT/Figures/{model_type}_roc_auc.png"
        loss_acc_save_path = f"D:/EuroSAT/Figures/{model_type}_loss_acc_curve.png"
    # dataset parameter
    class Dataset:
        batch_size = 128
        img_size = (64, 64)
        buffer_size = 100
    
    class HyperParameter:
        batch_size = 128
        learning_rate = 0.0003
        input_shape = (64, 64, 3)
        num_classes = 10
        epochs = 100