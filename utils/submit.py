from tqdm import tqdm
from utils.mask import rleFromMask
import pandas as pd


def compile_results(predections, output_path):

    rles = []
    for prediction in tqdm(predections):
        img = prediction.numpy()
        rles.append(rleFromMask(img, 1024, 1024))

    sample_sub = pd.read_csv("../data/sample_submission.csv")
    ids = sample_sub["ImageId"].to_list()

    sub = pd.DataFrame({"ImageId": ids, "EncodedPixels": rles})
    # if the prediction is null set the rle to -1
    sub.loc[sub.EncodedPixels == "", "EncodedPixels"] = "-1"
    print(sub.head())

    # saving the submission file to the output path with the current timestamp
    tstamp = pd.Timestamp.now().strftime("%Y-%m-%d-%H-%M")
    sub.to_csv(f"{output_path}/submission_{tstamp}.csv", index=False)
