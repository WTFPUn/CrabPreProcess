from VQVAE import VQVAE
import torch

batch_size = 16
num_training_updates = 1000

num_hiddens = 128
num_residual_hiddens = 32
num_residual_layers = 2

embedding_dim = 64
num_embeddings = 512

commitment_cost = 0.25

decay = 0.99

learning_rate = 1e-3

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = VQVAE(num_hiddens, num_residual_layers, num_residual_hiddens,
              num_embeddings, embedding_dim, 
              commitment_cost, decay).to(device)

from torch import optim
optimizer = optim.Adam(model.parameters(), lr=learning_rate, amsgrad=False)

from torchvision import transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    # flip the image
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

from CrabDataset import CrabDataset

train_dataset = CrabDataset('./dataset_big/',"original", "target", transform=transform)

from torch.utils.data import DataLoader
dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


import numpy as np

# model.load_state_dict(torch.load('./weight/vqvae_3200.pth'))

continue_epoch = 0

model.train()
train_res_recon_error = []
train_res_perplexity = []

from torch.nn import functional as F
import tqdm

save_every = 100
loop = tqdm.tqdm(range(num_training_updates))
for i in loop:
    (data, result) = next(iter(dataloader))
    data = data.to(device)
    result = result.to(device)
    optimizer.zero_grad()

    vq_loss, data_recon, perplexity = model(result)
    recon_error = F.mse_loss(data_recon, result)
    loss = recon_error + vq_loss
    loss.backward()

    optimizer.step()
    
    train_res_recon_error.append(recon_error.item())
    train_res_perplexity.append(perplexity.item())

    if (i+1) % 100 == 0:
        loop.set_description('recon_error: %.3f, perplexity: %.3f' % (np.mean(train_res_recon_error[-100:]), np.mean(train_res_perplexity[-100:])))

    if (i+1) % save_every == 0:
        torch.save(model.state_dict(), './weight/vqvae_%d.pth' % (i+1+continue_epoch))


# load the model


# try to input an image
from PIL import Image
import matplotlib.pyplot as plt

sample = Image.open('./gray_crab/1713595939.885463.jpg')
sample = transform(sample).to(device)
# reshape the image to 1, 3, 1200, 900
sample = sample.unsqueeze(0)


model.eval()

with torch.no_grad():
    _, data_recon, _ = model(sample)

plt.figure(figsize=(20, 20))
plt.subplot(1, 3, 1)
plt.imshow(sample[0].cpu().numpy().transpose(1, 2, 0))
plt.title('original image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(data_recon[0].cpu().numpy().transpose(1, 2, 0))
plt.title('reconstructed image')






plt.axis('off')

plt.show()

