from PIL import Image
import torch.utils.data as D
from tqdm import tqdm
from torchvision import transforms
import torch
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
trans = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor()
    ])


class MyDataset(D.Dataset):
    def __init__(self, img, mask, transform):
        self.img = img
        self.mask = mask
        self.transform = transform

    def __len__(self):
        return len(self.img)

    def __getitem__(self, idx):
        img = self.img[idx]
        mask = self.mask[idx]
        img = self.transform(img)
        mask = self.transform(mask)
        # mask = torch.cat([1 - mask, mask], dim=0)
        return img, mask

def get_dataloader(BATCH_SIZE):
    img = []
    mask = []
    for idx in tqdm(range(1, 1001)):
      img.append(Image.open(f'../shuiti/data/{idx}.jpg'))
      mask.append(Image.open(f'../shuiti/data/{idx}_gt.png'))

    train_loader = D.DataLoader(MyDataset(img[:900], mask[:900], trans), batch_size = BATCH_SIZE, shuffle = True, pin_memory = True, num_workers = 0)
    valid_loader = D.DataLoader(MyDataset(img[900:], mask[900:], trans), batch_size = BATCH_SIZE, pin_memory = True, num_workers = 0)
    return train_loader, valid_loader