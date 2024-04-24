from model.data_loader import PneumothoraxDataset
from torch.utils.data import DataLoader
import torch

# used to generate dataset
# PneumothoraxDataset(
#     root="data/pneumothorax",
#     anotations_file="train-rle.csv",
#     img_dir="dicom-images-train-raw",
#     mask_dir="dicom-images-train-mask",
#     dcm_dir="dicom-images-train",
#     generate=True,
# )

# training_data = PneumothoraxDataset(
#     root="data/pneumothorax",
#     anotations_file="train-rle.csv",
#     img_dir="dicom-images-train-raw",
#     mask_dir="dicom-images-train-mask",
# )


# train_dataloader = DataLoader(training_data, batch_size=5, shuffle=True)
# print(training_data.__getitem__(0))
# print("========")
# print(training_data.__getitem__(1))
# train_features, train_labels = next(iter(train_dataloader))
# print(f"Feature batch shape: {train_features.size()}")
# print(f"Labels batch shape: {train_labels.size()}")

# showOverlay(
#     "data/pneumothorax/dicom-images-train-mask",
#     "data/pneumothorax/dicom-images-train-raw",
#     "1.2.276.0.7230010.3.1.4.8323329.4904.1517875185.355709",
# )

training_data = PneumothoraxDataset(
    root="data",
    anotations_file="test.csv",
    img_dir="",
    mask_dir="",
)

train_dataloader = DataLoader(training_data)
data = training_data.__getitem__(0)
mask = data[1]

# print all the  unique pixel values in the mask
print(data[0].shape)
print(mask.shape)
unique = torch.unique(mask)
print(unique)
