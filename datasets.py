import numpy as np
from PIL import Image
from torch.utils import data
from pathlib import Path
from torchvision import transforms


def InfiniteSampler(n):
    # i = 0
    i = n - 1
    order = np.random.permutation(n)
    while True:
        yield order[i]
        i += 1
        if i >= n:
            np.random.seed(3407)
            order = np.random.permutation(n)
            i = 0


class InfiniteSamplerWrapper(data.sampler.Sampler):
    def __init__(self, data_source):
        super().__init__(data_source)
        self.num_samples = len(data_source)

    def __iter__(self):
        return iter(InfiniteSampler(self.num_samples))

    def __len__(self):
        return 2 ** 31


class FlatFolderDataset(data.Dataset):
    def __init__(self, root, transform, device):
        super(FlatFolderDataset, self).__init__()
        self.root = root
        self.paths = list(Path(self.root).glob('*'))
        self.transform = transform
        self.device = device

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(str(path)).convert('RGB')
        img = self.transform(img).to(self.device)
        return img

    def __len__(self):
        return len(self.paths)

    def name(self):
        return 'FlatFolderDataset'


def get_data_iter(device, data_dir, data_transform=None, batch_size=16, num_workers=16):
    if data_transform is None:
        data_transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.RandomCrop(256),
            transforms.ToTensor(),
        ])

        dataset = FlatFolderDataset(data_dir, data_transform, device)
        return iter(data.DataLoader(dataset,
                                    batch_size=batch_size,
                                    sampler=InfiniteSamplerWrapper(dataset),
                                    num_workers=num_workers))
