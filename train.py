from model import UNet
from data_loader import get_dataloader
from apex import amp
import torch
import torch.nn as nn
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--batch', type = int, default = 32)
parser.add_argument('--root', type = str, default = 'data')
parser.add_argument('--start', type = int, default = 0)
parser.add_argument('--end', type = int, default = 180)
args = parser.parse_args()
# from torchvision import models
# model = models.segmentation.deeplabv3_resnet101(pretrained=True)
# model.classifier[4] = nn.Conv2d(256, 1, 1)
model = UNet()
model.to('cuda')
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001, betas = (0.9, 0.999), eps = 1e-8, weight_decay = 1e-7)
model, optimizer = amp.initialize(model, optimizer, opt_level="O1", verbosity=0)
lr_schedular = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones= [10, 20, 30, 40, 50], gamma = 0.5)
train_loader, valid_loader = get_dataloader(args.batch, args.root, args.start, args.end)

best_loss = float('inf')
es = 0
for epoch in range(1000):
  model.train()
  t_loss = 0
  for img, mask in tqdm(train_loader):
    img = img.cuda()
    mask = mask.cuda()
    output = model(img)
    loss = criterion(output, mask)
    t_loss += loss.item()
    optimizer.zero_grad()
    with amp.scale_loss(loss, optimizer) as scaled_loss:
      scaled_loss.backward()
    # loss.backward()
    optimizer.step()
  lr_schedular.step()
  model.eval()
  v_loss = 0
  with torch.no_grad():
    for img, mask in tqdm(valid_loader):
      img = img.cuda()
      mask = mask.cuda()
      output = model(img)
      loss = criterion(output, mask)
      v_loss += loss.item()
  if v_loss<best_loss:
      es = 0
      best_loss = v_loss
      torch.save(model.state_dict(), f'best{args.start}-{args.end}.pkl')
  else:
    es += 1
    if es > 10:
      break
  print(f'epoch {epoch} | train loss:{t_loss/4:.4f} | valid loss:{v_loss:.4f}')
