import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import cv2
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
import segmentation_models_pytorch as smp
from segmentation_models_pytorch import utils
from segmentation_models_pytorch.utils.base import Loss
import pydicom
from tqdm import tqdm
import multiprocessing


## General Definitions

test_dir = "../../test_data"
dest_dir = "../../test_results"
model_path = "../../artifacts"
low = 16
up = 48


def main():
    test_model(22)
    # pool_obj = multiprocessing.Pool(24)

    # pool_obj.map(test_model, range(low, up + 1))

    # pool_obj.close()

    # print("All Done!")


def test_model(idx):
    print(f"Model of Epoch {idx}:")
    pred_to_csv(test_dir, dest_dir, model_path, idx)


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


def pred_to_csv(test_dir, dest_dir, model_path, model_number):
    ENCODER = "efficientnet-b4"
    ENCODER_WEIGHTS = "imagenet"
    ACTIVATION = (
        "sigmoid"  # could be None for logits or 'softmax2d' for multiclass segmentation
    )
    DEVICE = "cuda"

    epoch = -1

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

    checkpoint = torch.load(model_path + f"/model_epoch_{model_number}.pth")
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

    meta_df = pd.DataFrame(columns=["ImageId", "EncodedPixels"])

    images = [f for f in os.listdir(test_dir) if not f.startswith(".")]
    for image in tqdm(images):
        ds = pydicom.dcmread(os.path.join(test_dir, image))
        image_array = cv2.cvtColor(ds.pixel_array, cv2.COLOR_GRAY2RGB)
        image_array_512 = resize(512)(image=image_array)["image"]

        image_array_512t = get_preprocessing(preprocessing_fn)(image=image_array_512)[
            "image"
        ]

        image_tensor = torch.from_numpy(image_array_512t).to(DEVICE).unsqueeze(0)

        predicted_mask = model.predict(image_tensor)
        pr_mask = predicted_mask.squeeze().cpu().numpy().round()

        pr_mask_1024 = resize_up(1024)(image=image_array_512, mask=pr_mask)["mask"]

        pr_rle = mask2rle(pr_mask_1024, 1024, 1024)
        new_row = {"ImageId": image.split(".")[0], "EncodedPixels": pr_rle}
        meta_df = meta_df._append(new_row, ignore_index=True)

    # tstamp = pd.Timestamp.now().strftime("%Y-%m-%d-%H-%M")
    meta_df.to_csv(f"{dest_dir}/inter_cubic_{model_number}_final-rle.csv", index=False)

    print("Success!")


def resize(num):
    return Compose([Resize(width=num, height=num, interpolation=cv2.INTER_AREA)])


def resize_up(num):
    return Compose([Resize(width=num, height=num, interpolation=cv2.INTER_CUBIC)])


def mask2rle(img, width, height):
    rle = []
    lastColor = 0
    currentPixel = 0
    runStart = -1
    runLength = 0

    for x in range(width):
        for y in range(height):
            currentColor = img[x][y]
            if currentColor != lastColor:
                if currentColor == 1.0:
                    runStart = currentPixel
                    runLength = 1
                else:
                    rle.append(str(runStart))
                    rle.append(str(runLength))
                    runStart = -1
                    runLength = 0
                    currentPixel = 0
            elif runStart > -1:
                runLength += 1
            lastColor = currentColor
            currentPixel += 1
    return " ".join(rle) if len(rle) else "-1"


if __name__ == "__main__":
    main()
