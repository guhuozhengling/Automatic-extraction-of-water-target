from model import UNet
from data_loader import get_dataloader
from apex import amp
import torch
import torch.nn as nn
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--batch', type = int, default = 4)
args = parser.parse_args()

model = UNet()
model.to('cuda')
# criterion = DiceLoss()
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001, betas = (0.9, 0.999), eps = 1e-8, weight_decay = 1e-8)
model, optimizer = amp.initialize(model, optimizer, opt_level="O1", verbosity=0)
lr_schedular = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones= [50, 100, 150, 200, 250], gamma = 0.5)
train_loader, valid_loader = get_dataloader(args.batch)
from tqdm import tqdm
best_loss = 100
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
      best_loss = v_loss
      torch.save(model.state_dict(), 'best.pkl')
  print(f'epoch {epoch} | train loss:{t_loss/900:.4f} | valid loss:{v_loss/100:.4f}')
