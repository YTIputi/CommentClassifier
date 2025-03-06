import os
import kaggle


def download_dataset(data_dir):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    kaggle.api.dataset_download_files("atifaliak/youtube-comments-dataset", path=data_dir, unzip=True)


if __name__ == "__main__":
    data_dir = "data/raw"
    download_dataset(data_dir)
