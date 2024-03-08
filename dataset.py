from sample import load_samples
from torch.utils.data import Dataset, DataLoader

class ShakeFive2Dataset(Dataset):
    def __init__(self, image_paths, transform=False):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_filepath = self.image_paths[idx]
        image = cv2.imread(image_filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        label = image_filepath.split('/')[-2]
        label = self.class_to_idx[label]
        if self.transform is not None:
            image = self.transform(image=image)["image"]

        return image, label


#######################################################
#                  Create Dataset
#######################################################

train_dataset = LandmarkDataset(train_image_paths, train_transforms)
valid_dataset = LandmarkDataset(valid_image_paths, test_transforms)  # test transforms are applied
test_dataset = LandmarkDataset(test_image_paths, test_transforms)