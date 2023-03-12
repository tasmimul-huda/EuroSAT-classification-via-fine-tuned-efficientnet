from config import CFG
from dataloader import EuroSatLoader

cfg = CFG()

train_loader = EuroSatLoader(csv_path = cfg.Path.train_df, 
                             image_folder = cfg.Path.image_folder,
                             batch_size = cfg.Dataset.batch_size,
                             img_size = cfg.Dataset.img_size,
                             buffer_size = cfg.Dataset.buffer_size, 
                             shuffle = True)
train_dataset = train_loader.get_dataset()

print("Train Dataset")
for i, (x, y) in enumerate(train_dataset):
    print(x.shape, y.shape)
    if i == 10:
        break
    

val_loader = EuroSatLoader(csv_path = cfg.Path.val_df, 
                             image_folder = cfg.Path.image_folder,
                             batch_size = cfg.Dataset.batch_size,
                             img_size = cfg.Dataset.img_size,
                             buffer_size = cfg.Dataset.buffer_size, 
                             shuffle = False)
val_dataset = val_loader.get_dataset()

print("Validataion Dataset")
for i, (x, y) in enumerate(val_dataset):
    print(x.shape, y.shape)
    if i == 10:
        break
    

test_loader = EuroSatLoader(csv_path = cfg.Path.test_df, 
                             image_folder = cfg.Path.image_folder,
                             batch_size = cfg.Dataset.batch_size,
                             img_size = cfg.Dataset.img_size,
                             buffer_size = cfg.Dataset.buffer_size, 
                             shuffle = False)
test_dataset = test_loader.get_dataset()

print("Test Dataset")
for i, (x, y) in enumerate(test_dataset):
    print(x.shape, y.shape)
    if i == 10:
        break