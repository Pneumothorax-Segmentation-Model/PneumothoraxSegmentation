import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
import segmentation_models_pytorch as smp
from segmentation_models_pytorch import utils
from segmentation_models_pytorch.utils.base import Loss
import wandb

resume = 1
number_of_epochs = 10

encoder_name = "se_resnext101_32x4d"

wandb.init(
    # Set the project where this run will be logged
    project="PneumothoraxSegmentation",
    # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
    name=f"experiment_se_resnext101_32x4d_lr_plateau_combo_loss",
    # Track hyperparameters and run metadata
    config={
        "learning_rate": 0.0001,
        "architecture": "se_resnext101_32x4d",
        "dataset": "SIIM-ACR",
        "epochs": number_of_epochs,
    },
)


torch.cuda.empty_cache()


root = "../../Data"
print(os.getcwd())
print(os.listdir(root))


images_dir = "png_files"
masks_dir = "mask_files"
train_csv = "csv/train_upsampled.csv"
val_csv = "csv/val_final.csv"


def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(" ".join(name.split("_")).title())
        plt.imshow(image)
    plt.show()


class Dataset(BaseDataset):

    def __init__(
        self,
        root,
        images_dir,
        masks_dir,
        csv,
        aug_fn=None,
        id_col="DICOM",
        aug_col="Augmentation",
        preprocessing_fn=None,
    ):
        images_dir = os.path.join(root, images_dir)
        masks_dir = os.path.join(root, masks_dir)
        df = pd.read_csv(os.path.join(root, csv))

        self.ids = [(r[id_col], r[aug_col]) for i, r in df.iterrows()]
        self.images = [os.path.join(images_dir, item[0] + ".png") for item in self.ids]
        self.masks = [
            os.path.join(masks_dir, item[0] + "_mask.png") for item in self.ids
        ]
        self.aug_fn = aug_fn
        self.preprocessing_fn = preprocessing_fn

    def __getitem__(self, i):

        image = cv2.imread(self.images[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = (cv2.imread(self.masks[i], 0) == 255).astype("float")
        mask = np.expand_dims(mask, axis=-1)

        aug = self.ids[i][1]
        # if aug:
        augmented = self.aug_fn(aug)(image=image, mask=mask)
        image, mask = augmented["image"], augmented["mask"]

        if self.preprocessing_fn:
            sample = self.preprocessing_fn(image=image, mask=mask)
            image, mask = sample["image"], sample["mask"]

        return image, mask

    def __len__(self):
        return len(self.ids)


from albumentations import (
    HorizontalFlip,
    RandomBrightnessContrast,
    RandomGamma,
    CLAHE,
    ElasticTransform,
    GridDistortion,
    OpticalDistortion,
    ShiftScaleRotate,
    Normalize,
    GaussNoise,
    Compose,
    Lambda,
    Resize,
)


def augmentation_fn(value, resize=21):
    augmentation_options = {
        0: [],
        1: [HorizontalFlip(p=1)],
        2: [RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1)],
        3: [RandomGamma(p=1)],
        4: [CLAHE(clip_limit=4.0, tile_grid_size=(4, 4), p=1)],
        5: [OpticalDistortion(p=1)],
        6: [ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=15, p=1)],
        7: [GaussNoise(p=1)],
        8: [
            HorizontalFlip(p=1),
            RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1),
            RandomGamma(p=1),
        ],
        9: [
            HorizontalFlip(p=1),
            RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1),
            CLAHE(clip_limit=4.0, tile_grid_size=(4, 4), p=1),
        ],
        10: [
            HorizontalFlip(p=1),
            RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1),
            OpticalDistortion(p=1),
        ],
        11: [
            HorizontalFlip(p=1),
            RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1),
            GaussNoise(p=1),
        ],
        12: [
            ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=15, p=1),
            GaussNoise(p=1),
        ],
        13: [CLAHE(clip_limit=4.0, tile_grid_size=(4, 4), p=1), GaussNoise(p=1)],
        14: [CLAHE(clip_limit=4.0, tile_grid_size=(4, 4), p=1), OpticalDistortion(p=1)],
        15: [CLAHE(clip_limit=4.0, tile_grid_size=(4, 4), p=1), RandomGamma(p=1)],
        16: [RandomGamma(p=1), OpticalDistortion(p=1)],
        17: [
            RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1),
            GaussNoise(p=1),
        ],
        18: [
            ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=15, p=1),
            RandomGamma(p=1),
        ],
        19: [
            ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=15, p=1),
            HorizontalFlip(p=1),
        ],
        20: [
            ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=15, p=1),
            GaussNoise(p=1),
            OpticalDistortion(p=1),
        ],
        21: [Resize(width=512, height=512, interpolation=cv2.INTER_AREA)],
    }

    return Compose(augmentation_options[resize] + augmentation_options[value])


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype("float32")


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform

    Args:
        preprocessing_fn (callbale): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose

    """

    _transform = [
        Lambda(image=preprocessing_fn),
        Lambda(image=to_tensor, mask=to_tensor),
    ]
    return Compose(_transform)


dataset = Dataset(
    root=root,
    images_dir=images_dir,
    masks_dir=masks_dir,
    csv=train_csv,
    aug_fn=augmentation_fn,
)
print(len(dataset))
image, mask = dataset[1]  # get some sample
print(image.shape)
print(mask.shape)
print(np.unique(mask.squeeze()))
print(set(mask.flatten()))
# visualize(
#     image=image,
#     cars_mask=mask.squeeze(),
# )


class WeightedSumOfLosses(Loss):
    def __init__(self, l1, l2, w1=1, w2=0.5):
        name = "{} + {}".format(l1.__name__, l2.__name__)
        super().__init__(name=name)
        self.l1 = l1
        self.l2 = l2
        self.w1 = w1
        self.w2 = w2

    def __call__(self, *inputs):
        return self.w1 * self.l1.forward(*inputs) + self.w2 * self.l2.forward(*inputs)


ENCODER = "se_resnext101_32x4d"
ENCODER_WEIGHTS = "imagenet"
ACTIVATION = (
    "sigmoid"  # could be None for logits or 'softmax2d' for multiclass segmentation
)
DEVICE = "cuda"

# create segmentation model with pretrained encoder
epoch = -1
max_score = 0
val_loss = 0
train_loss = 0
train_iou = 0
val_iou = 0

model = smp.Unet(
    encoder_name=ENCODER,
    encoder_weights=ENCODER_WEIGHTS,
    activation=ACTIVATION,
).to(DEVICE)


loss = WeightedSumOfLosses(utils.losses.DiceLoss(), utils.losses.BCELoss())
metrics = [utils.metrics.IoU(threshold=0.5)]

optimizer = torch.optim.Adam(
    [
        dict(params=model.parameters(), lr=0.0001),
    ]
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer=optimizer,
    mode="max",
    factor=0.8,
    patience=2,
    threshold=0.01,
    threshold_mode="abs",
)


if resume:
    checkpoint = torch.load("latest_model_cosine.pth")
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    epoch = checkpoint["epoch"]
    loss = checkpoint["loss_function"]
    max_score = checkpoint["max_score"]
    val_loss = checkpoint["val_loss"]
    train_loss = checkpoint["train_loss"]
    train_iou = checkpoint["train_iou"]
    val_iou = checkpoint["val_iou"]
    started_lr = checkpoint["started_lr"]
    scheduler.best = max_score


print("Scheduler State Dict Outside: ", scheduler.state_dict())
print("Epoch:", epoch)
print("Loss Function:", loss)
print("Max Val Score:", max_score)
print(
    f"Train Iou: {train_iou}\nValid Iou: {val_iou}\nTrain Loss: {train_loss}\nVal Loss: {val_loss}"
)
print("Optimizer LR:", optimizer.param_groups[0]["lr"])

preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)


train_dataset = Dataset(
    root=root,
    images_dir=images_dir,
    masks_dir=masks_dir,
    csv=train_csv,
    aug_fn=augmentation_fn,
    preprocessing_fn=get_preprocessing(preprocessing_fn),
)
print(len(train_dataset))

val_dataset = Dataset(
    root=root,
    images_dir=images_dir,
    masks_dir=masks_dir,
    csv=val_csv,
    aug_fn=augmentation_fn,
    preprocessing_fn=get_preprocessing(preprocessing_fn),
)

print(len(val_dataset))

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
valid_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)


train_epoch = utils.train.TrainEpoch(
    model,
    loss=loss,
    metrics=metrics,
    optimizer=optimizer,
    device=DEVICE,
    verbose=True,
)

valid_epoch = utils.train.ValidEpoch(
    model,
    loss=loss,
    metrics=metrics,
    device=DEVICE,
    verbose=True,
)


val_iou_scores = []
train_iou_scores = []
val_dice_loss = []
train_dice_loss = []
starting_lrs = []

for i in range(epoch + 1, epoch + number_of_epochs + 1):

    print("\nEpoch: {}".format(i))
    train_logs = train_epoch.run(train_loader)
    valid_logs = valid_epoch.run(valid_loader)

    train_iou_scores.append((i, train_logs["iou_score"]))
    val_iou_scores.append((i, valid_logs["iou_score"]))
    val_dice_loss.append((i, valid_logs["dice_loss + bce_loss"]))
    train_dice_loss.append((i, train_logs["dice_loss + bce_loss"]))

    started_lr = optimizer.param_groups[0]["lr"]
    print("Started with LR:", started_lr)
    scheduler.step(valid_logs["iou_score"])
    print("Changed LR to:", optimizer.param_groups[0]["lr"])
    starting_lrs.append((i, started_lr))

    if max_score < valid_logs["iou_score"]:
        max_score = valid_logs["iou_score"]
        print("best valid model!")

    checkpoint = {
        "epoch": i,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss_function": loss,
        "max_score": max_score,
        "val_loss": valid_logs["dice_loss + bce_loss"],
        "train_loss": train_logs["dice_loss + bce_loss"],
        "train_iou": train_logs["iou_score"],
        "val_iou": valid_logs["iou_score"],
        "scheduler_state_dict": scheduler.state_dict(),
        "started_lr": started_lr,
    }

    torch.save(checkpoint, f"model_epoch_{i}.pth")
    print("model saved!")

    if i == epoch + number_of_epochs:
        torch.save(checkpoint, "latest_model_cosine.pth")
        print("latest model saved!")

    np.savetxt(
        f"train_iou_scores_{epoch+number_of_epochs}.txt", train_iou_scores, fmt="%.5f"
    )
    np.savetxt(
        f"val_iou_scores_{epoch+number_of_epochs}.txt", val_iou_scores, fmt="%.5f"
    )
    np.savetxt(f"val_dice_loss_{epoch+number_of_epochs}.txt", val_dice_loss, fmt="%.5f")
    np.savetxt(
        f"train_dice_loss_{epoch+number_of_epochs}.txt", train_dice_loss, fmt="%.5f"
    )
    np.savetxt(
        f"started_learning_rates_{epoch+number_of_epochs}.txt", starting_lrs, fmt="%.5f"
    )

    wandb.log(
        {
            "epoch": i,
            "max_score": max_score,
            "val_loss": valid_logs["dice_loss + bce_loss"],
            "train_loss": train_logs["dice_loss + bce_loss"],
            "train_iou": train_logs["iou_score"],
            "val_iou": valid_logs["iou_score"],
            "started_lr": started_lr,
        }
    )

wandb.finish()


print("train iou:\n", train_iou_scores)
print()
print("train loss:\n", train_dice_loss)
print()
print("val iou:\n", val_iou_scores)
print()
print("val loss:\n", val_dice_loss)
