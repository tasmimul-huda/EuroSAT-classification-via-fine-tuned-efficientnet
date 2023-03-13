from config import CFG
from dataloader import EuroSatLoader
from models import EfficientNetModel, BaselineCNN
from pathlib import Path
from utils import ModelUtils

cfg = CFG()


if __name__ == '__main__':
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
    print(f"INFO ===========Training Started===============")
    history = model.train(train_data = train_dataset, 
                          val_data = val_dataset,
                          epochs = cfg.HyperParameter.epochs,
                          batch_size= cfg.HyperParameter.batch_size, 
                          learning_rate= cfg.HyperParameter.learning_rate,
                          model_save_path= cfg.Path.model_save_path,
                          log_dir= cfg.Path.tensorboard_logs_path)
    print(f"INFO ===========Training Finished===============")
    print(f"INFO ===========Plot Curve===============")
    ModelUtils.plot_loss_accuracy(cfg.Path.figure_save_path,history)
    
    
    
