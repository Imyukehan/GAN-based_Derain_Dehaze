import os
import random
from PIL import Image
from torch.utils.data import Dataset


class PairedImageDataset(Dataset):
    def __init__(self, data_dir, transforms=None, crop_size=None):
        in_list_path = os.path.join(data_dir, 'in_list.txt')
        gt_list_path = os.path.join(data_dir, 'gt_list.txt')
        assert os.path.exists(in_list_path) and os.path.exists(gt_list_path), 'File list not found!'

        self.in_list = open(in_list_path).read().splitlines()
        self.gt_list = open(gt_list_path).read().splitlines()
        assert len(self.in_list) == len(self.gt_list), 'Lengths of the file list are inconsistent!'

        self.transforms = transforms

        if crop_size is not None:
            assert isinstance(crop_size, list) and (len(crop_size) == 1 or len(crop_size) == 2)
            if len(crop_size) == 1:
                self.crop_size = crop_size * 2
            else:
                self.crop_size = crop_size
        else:
            self.crop_size = None

    def __getitem__(self, index):
        img_in = Image.open(self.in_list[index])
        img_gt = Image.open(self.gt_list[index])

        if self.crop_size is not None:
            # apply random crop to images
            img_in, img_gt = self._random_crop([img_in, img_gt])

        # apply transforms to
        img_in = self.transforms(img_in)
        img_gt = self.transforms(img_gt)

        return {"in": img_in, "gt": img_gt}

    def __len__(self):
        return len(self.in_list)

    # randomly crop the images to specific size
    def _random_crop(self, images):
        left = random.randrange(images[0].width - self.crop_size[0])
        top = random.randrange(images[1].height - self.crop_size[1])
        right = left + self.crop_size[0]
        bottom = top + self.crop_size[1]
        return (img.crop((left, top, right, bottom)) for img in images)