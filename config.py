class CFG:  
    # path defination
    class Path:
        train_df = 'D:/EuroSAT/EuroSAT/train.csv'
        val_df = 'D:/EuroSAT/EuroSAT/validation.csv'
        test_df = 'D:/EuroSAT/EuroSAT/test.csv'
        image_folder = 'D:/EuroSAT/EuroSAT/'
        model_save_path = 'D:/EuroSAT/weights/'
        tensorboard_logs_path = "D:/EuroSAT/tensorboard/logs/"
        figure_save_path = "D:/EuroSAT/Figures/"
        
    # dataset parameter
    class Dataset:
        batch_size = 64
        img_size = (64, 64)
        buffer_size = 1000
    
    class HyperParameter:
        batch_size = 64
        learning_rate = 0.0001
        input_shape = (64, 64, 3)
        num_classes = 10
        epochs = 100
        
        
        
        