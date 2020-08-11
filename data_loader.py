import numpy as np
import cv2
from PIL import Image
import torch.utils.data as D
from tqdm import tqdm
from torchvision import transforms
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
trans = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor()
    ])
def get_img(root):
    img_ = []
    mask_ = []
    for idx in tqdm(range(1, 1001)):
        img = Image.open(f'{root}/{idx}.jpg')
        i1 = img.crop((0,0,256,256))
        i2 = img.crop((235,235,491,491))
        i3 = img.crop((235,0,491,256))
        i4 = img.crop((0,235,256,491))
        mask = Image.open(f'{root}/{idx}_gt.png')
        m1 = mask.crop((0,0,256,256))
        m2 = mask.crop((235,235,491,491))
        m3 = mask.crop((235,0,491,256))
        m4 = mask.crop((0,235,256,491))
        # img_.append(img)
        img_.append(i1)
        img_.append(i2)
        img_.append(i3)
        img_.append(i4)
        # mask_.append(mask)
        mask_.append(m1)
        mask_.append(m2)
        mask_.append(m3)
        mask_.append(m4)
    img_ = img_[:3600]
    mask_ = mask_[:3600]
    return img_, mask_
def gamma_transform(img, gamma):
    gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
    return cv2.LUT(img, gamma_table)

def random_gamma_transform(img, gamma_vari):
    log_gamma_vari = np.log(gamma_vari)
    alpha = np.random.uniform(-log_gamma_vari, log_gamma_vari)
    gamma = np.exp(alpha)
    return gamma_transform(img, gamma)

def rotate(xb,yb,angle):
    img_w, img_h = xb.shape[:2]
    M_rotate = cv2.getRotationMatrix2D((img_w/2, img_h/2), angle, 1)
    xb = cv2.warpAffine(xb, M_rotate, (img_w, img_h))
    yb = cv2.warpAffine(yb, M_rotate, (img_w, img_h))
    return xb,yb

def blur(img):
    img = cv2.blur(img, (3, 3));
    return img

def add_noise(img):
    for i in range(200): #添加点噪声
        temp_x = np.random.randint(0,img.shape[0])
        temp_y = np.random.randint(0,img.shape[1])
        img[temp_x][temp_y] = 255
    return img


def data_augment(xb,yb):
    xb = np.array(xb)
    yb = np.array(yb)
    if np.random.random() < 0.25:
        xb,yb = rotate(xb,yb,90)
    if np.random.random() < 0.25:
        xb,yb = rotate(xb,yb,180)
    if np.random.random() < 0.25:
        xb,yb = rotate(xb,yb,270)
    if np.random.random() < 0.25:
        xb = cv2.flip(xb, 1)  # flipcode > 0：沿y轴翻转
        yb = cv2.flip(yb, 1)

    if np.random.random() < 0.25:
        xb = random_gamma_transform(xb,1.0)

    if np.random.random() < 0.25:
        xb = blur(xb)

    if np.random.random() < 0.2:
        xb = add_noise(xb)

    return Image.fromarray(xb), Image.fromarray(yb)

class MyDataset(D.Dataset):
    def __init__(self, img, mask, transform, train = True):
        self.img = img
        self.mask = mask
        self.transform = transform
        self.train = train

    def __len__(self):
        return len(self.img)

    def __getitem__(self, idx):
        img = self.img[idx]
        mask = self.mask[idx]
        if self.train:
            img, mask = data_augment(img, mask)
        img = self.transform(img)
        mask = self.transform(mask)
        return img, mask

def get_dataloader(BATCH_SIZE, root, start, end):
    img, mask = get_img(root)
    train_loader = D.DataLoader(MyDataset(img[:start]+img[end:], mask[:start]+mask[end:], trans), batch_size = BATCH_SIZE, shuffle = True, pin_memory = True, num_workers = 0)
    valid_loader = D.DataLoader(MyDataset(img[start:end], mask[start:end], trans, train = False), batch_size = BATCH_SIZE, pin_memory = True, num_workers = 0)
    return train_loader, valid_loader