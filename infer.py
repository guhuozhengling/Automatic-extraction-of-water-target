import torch
from torchvision import transforms
from PIL import Image

from model import UNet

model = UNet()
model.load_state_dict(torch.load('best0-1000.pkl'))
model.to('cuda')

img = Image.open('data/985.jpg')
trans = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor()
])
tensor = trans(img).unsqueeze(axis = 0).cuda()

model.eval()
with torch.no_grad():
    res = model(tensor)
    res = res.squeeze().cpu()
    res = transforms.ToPILImage()(res)
    res.save('res.jpg')