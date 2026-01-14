import torch
import torchvision
import torch.nn.functional as F
from torchvision import transforms 
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
import math
import os # Added for file saving
from torch.optim import Adam # Moved import to top

# --- CONFIGURATION ---
IMG_SIZE = 64
BATCH_SIZE = 128
T = 300 # Timesteps
EPOCHS = 100 # Number of training loops
LR = 0.001 # Learning Rate

# Check for GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# --- DIFFUSION MATH ---

def linear_beta_schedule(timesteps, start=0.0001, end=0.02): 
    return torch.linspace(start, end, timesteps)

def get_index_from_list(vals, t, x_shape): 
    """ 
    Returns a specific index at t of a passed tensor of values vals
    while considering the batch dimension.
    """
    batch_size = t.shape[0]
    out = vals.gather(-1, t.cpu()) 
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

def forward_diffusion_sample(x_0, t, device="cpu"):
    """ 
    Given an image x_0 and a timestep t, we sample the noise and return the
    image with the noise applied to it.
    """
    noise = torch.randn_like(x_0)
    sqrt_alphas_cumprod_t = get_index_from_list(sqrt_alphas_cumprod, t, x_0.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t, x_0.shape
    )
    
    return sqrt_alphas_cumprod_t.to(device) * x_0.to(device) \
    + sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device), noise.to(device)

# Define beta schedule
betas = linear_beta_schedule(timesteps=T)

# Calculate variables to be used throughout the model
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

# --- DATASET LOADING ---

def load_transformed_dataset():
    data_transforms = [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), # Scales data to [0,1] 
        transforms.Lambda(lambda t: (t * 2) - 1) # Scales between [-1, 1]
    ]
    data_transform = transforms.Compose(data_transforms)

    # Note: This might require 'pip install scipy' in some environments
    train = torchvision.datasets.Flowers102(root=".", download=True, 
                                         transform=data_transform)
    test = torchvision.datasets.Flowers102(root=".", download=True, 
                                         transform=data_transform, split='test')
    return torch.utils.data.ConcatDataset([train, test])
    
def show_tensor_image(image):
    """
    Converts tensor back to image format for matplotlib (CHW -> HWC)
    """
    reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.Lambda(lambda t: t.permute(1, 2, 0)), 
        transforms.Lambda(lambda t: t * 255.),
        transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
        transforms.ToPILImage(),
    ])

    if len(image.shape) == 4:
        image = image[0, :, :, :] 
    plt.imshow(reverse_transforms(image))

# Initialize Data
print("Loading Dataset... (This may take a moment to download)")
try:
    data = load_transformed_dataset()
    dataloader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    print("Dataset Loaded Successfully.")
except ImportError:
    print("Error: Missing dependency. Try running: !pip install scipy")

# --- MODEL ARCHITECTURE (U-NET) ---

class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False):
        super().__init__()
        self.time_mlp =  nn.Linear(time_emb_dim, out_ch)
        if up:
            self.conv1 = nn.Conv2d(2*in_ch, out_ch, 3, padding=1)
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.bnorm2 = nn.BatchNorm2d(out_ch)
        self.relu  = nn.ReLU()
        
    def forward(self, x, t, ):
        h = self.bnorm1(self.relu(self.conv1(x)))
        time_emb = self.relu(self.time_mlp(t))
        time_emb = time_emb[(..., ) + (None, ) * 2]
        h = h + time_emb
        h = self.bnorm2(self.relu(self.conv2(h)))
        return self.transform(h)

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class SimpleUnet(nn.Module):
    """
    A simplified Unet architecture for image denoising
    """
    def __init__(self):
        super().__init__()
        img_channels = 3
        down_channels = (64, 128, 256, 512, 1024)
        up_channels = (1024, 512, 256, 128, 64)
        out_dim = 3 
        time_emb_dim = 32

        self.time_mlp = nn.Sequential(
                SinusoidalPositionEmbeddings(time_emb_dim),
                nn.Linear(time_emb_dim, time_emb_dim),
                nn.ReLU()
            )
        
        self.conv0 = nn.Conv2d(img_channels, down_channels[0], 3, padding=1)

        self.downs = nn.ModuleList([Block(down_channels[i], down_channels[i+1], \
                                    time_emb_dim) \
                    for i in range(len(down_channels)-1)])
        
        self.ups = nn.ModuleList([Block(up_channels[i], up_channels[i+1], \
                                        time_emb_dim, up=True) \
                    for i in range(len(up_channels)-1)])
        
        self.output = nn.Conv2d(up_channels[-1], out_dim, 1)

    def forward(self, x, timestep):
        t = self.time_mlp(timestep)
        x = self.conv0(x)
        residual_inputs = []
        for down in self.downs:
            x = down(x, t)
            residual_inputs.append(x)
        for up in self.ups:
            residual_x = residual_inputs.pop()
            x = torch.cat((x, residual_x), dim=1)           
            x = up(x, t)
        return self.output(x)

# Initialize Model
model = SimpleUnet()
model.to(device)
print("Model Parameters:", sum(p.numel() for p in model.parameters()))

# --- LOSS & SAMPLING ---

def get_loss(model, x_0, t):
    x_noisy, noise = forward_diffusion_sample(x_0, t, device)
    noise_pred = model(x_noisy, t)
    return F.l1_loss(noise, noise_pred)

@torch.no_grad()
def sample_timestep(x, t):
    betas_t = get_index_from_list(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = get_index_from_list(sqrt_recip_alphas, t, x.shape)
    
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
    )
    posterior_variance_t = get_index_from_list(posterior_variance, t, x.shape)
    
    if t == 0:
        return model_mean
    else:
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise 

@torch.no_grad()
def sample_plot_image(epoch_num=0):
    img_size = IMG_SIZE
    img = torch.randn((1, 3, img_size, img_size), device=device)
    plt.figure(figsize=(15,15))
    plt.axis('off')
    num_images = 10
    stepsize = int(T/num_images)

    for i in range(0,T)[::-1]:
        t = torch.full((1,), i, device=device, dtype=torch.long)
        img = sample_timestep(img, t)
        img = torch.clamp(img, -1.0, 1.0)
        if i % stepsize == 0:
            plt.subplot(1, num_images, int(i/stepsize)+1)
            show_tensor_image(img.detach().cpu())
    
    plt.show()
    # Save the plot for reference
    if not os.path.exists('./generated_images'):
        os.makedirs('./generated_images')
    plt.savefig(f'./generated_images/plot_epoch_{epoch_num}.png')

# --- TRAINING LOOP (FIXED) ---

optimizer = Adam(model.parameters(), lr=LR)

# Create weights folder if it doesn't exist
if not os.path.exists('./weights'):
    os.makedirs('./weights')

print(f"Starting training for {EPOCHS} epochs...")

for epoch in range(EPOCHS):
    for step, batch in enumerate(dataloader):
      optimizer.zero_grad()
      
      # Handle batch loading
      t = torch.randint(0, T, (BATCH_SIZE,), device=device).long()
      
      # Batch[0] contains the images
      loss = get_loss(model, batch[0], t)
      loss.backward()
      optimizer.step()

      if step % 50 == 0:
        print(f"Epoch {epoch} | Step {step:03d} | Loss: {loss.item()} ")

    # Save weights & Sample every 5 epochs
    if epoch % 5 == 0:
        print(f"Saving Checkpoint at Epoch {epoch}...")
        torch.save(model.state_dict(), f'./weights/flower_model_epoch_{epoch}.pt')
        sample_plot_image(epoch)

# Final Save
torch.save(model.state_dict(), 'flower_diffusion_final.pt')
print("Training Complete! Download 'flower_diffusion_final.pt' from the files tab.")