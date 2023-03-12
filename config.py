class CFG:  
    # path defination
    class Path:
        train_df = 'D:/EuroSAT/EuroSAT/train.csv'
        val_df = 'D:/EuroSAT/EuroSAT/validation.csv'
        test_df = 'D:/EuroSAT/EuroSAT/test.csv'
        image_folder = 'D:/EuroSAT/EuroSAT/'
        model_save_path = 'D:/EuroSAT/weights/'
        tensorboard_logs_path = "D:/EuroSAT/tensorboard/logs/"
        
    # dataset parameter
    class Dataset:
        batch_size = 32
        img_size = (32, 32)
        buffer_size = 1000
    
    class HyperParameter:
        batch_size = 64
        learning_rate = 0.001
        input_shape = (224, 224, 3)
        num_classes = 10
        
        
        
        