from torch.utils.data import Dataset
import torch


class CommentDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = self.texts[item]
        label = self.labels[item]

        return {
            'text': torch.tensor(text, dtype=torch.float32),
            'label': torch.tensor(label, dtype=torch.float32)
        }