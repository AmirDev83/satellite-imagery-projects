import time

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import jaccard_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from torchgeo.datasets import LandCoverAI


def extract_features(dataloader):
    x_all = []
    y_all = []

    for batch in tqdm(dataloader):
        images = np.rollaxis(batch["image"].numpy(), 1, 4).reshape(-1, 3)
        masks = batch["mask"].numpy().reshape(-1)
        x_all.append(images)
        y_all.append(masks)

    x_all = np.concatenate(x_all, axis=0)
    y_all = np.concatenate(y_all, axis=0)

    return x_all, y_all


def main():
    train_dataset = LandCoverAI(
        root="/home/calebrobinson/ssdprivate/torchgeo_dataset_integration_tests/data/LandCoverAI",
        split="train",
    )
    test_dataset = LandCoverAI(
        root="/home/calebrobinson/ssdprivate/torchgeo_dataset_integration_tests/data/LandCoverAI",
        split="test",
    )

    train_dataloader = DataLoader(
        train_dataset, batch_size=32, shuffle=True, num_workers=8
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=32, shuffle=False, num_workers=8
    )

    tic = time.time()
    x_train, y_train = extract_features(train_dataloader)
    x_test, y_test = extract_features(test_dataloader)
    print("Feature extraction time: {:02f} seconds".format(time.time() - tic))

    # shuffle train efficiently
    # tic = time.time()
    # n_train = x_train.shape[0]
    # idx = np.random.permutation(n_train)
    # x_train = x_train[idx]
    # y_train = y_train[idx]
    # print("Shuffling time: {:02f} seconds".format(time.time() - tic))

    n_sample = int(0.01 * x_train.shape[0])
    print("Training on {} samples".format(n_sample))
    tic = time.time()
    model = RandomForestClassifier(n_estimators=400, n_jobs=-1)
    model.fit(x_train[:n_sample], y_train[:n_sample])
    print("Training time: {:02f} seconds".format(time.time() - tic))

    tic = time.time()
    step_size = 10000000
    y_preds = []
    for i in tqdm(range(0, x_test.shape[0], step_size)):
        y_preds.append(model.predict(x_test[i : i + step_size]))
    y_pred = np.concatenate(y_preds, axis=0)
    print("Inference time: {:02f} seconds".format(time.time() - tic))

    tic = time.time()
    ious = jaccard_score(y_test, y_pred, average=None)
    print("Scoring time: {:02f} seconds".format(time.time() - tic))

    print("Results")
    print("-------")
    print(ious)
    print(np.mean(ious))

    # Compute a confusion matrix efficiently
    tic = time.time()
    classes = np.unique(y_test)
    confusion_matrix = np.zeros((len(classes), len(classes)))
    for i, c in enumerate(classes):
        mask1 = y_test == c
        for j, d in enumerate(classes):
            mask2 = y_pred == d
            confusion_matrix[i, j] = np.sum(mask1 & mask2)
    np.save("confusion_matrix.npy", confusion_matrix)
    print("Confusion matrix time: {:02f} seconds".format(time.time() - tic))

    # Compute mean IoU from confusion matrix
    mean_iou = np.diag(confusion_matrix) / (
        np.sum(confusion_matrix, axis=1)
        + np.sum(confusion_matrix, axis=0)
        - np.diag(confusion_matrix)
    )
    print(mean_iou)


if __name__ == "__main__":
    main()
