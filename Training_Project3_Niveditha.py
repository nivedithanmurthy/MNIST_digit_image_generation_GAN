import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt


# Hyperparameters
batch_size = 128
learning_rate = 0.0002
num_epochs = 200
latent_dim = 100
image_size = 28 * 28  # MNIST images are 28x28 pixels
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# MNIST Dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# Generator
class Generator(nn.Module):
    def __init__(self, latent_dim, image_dim):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, image_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)

# Discriminator
class Discriminator(nn.Module):
    def __init__(self, image_dim):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(image_dim, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

# Initialize the Generator and Discriminator
generator = Generator(latent_dim, image_size).to(device)
discriminator = Discriminator(image_size).to(device)

# Optimizers
optimizer_G = optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))

# Loss function
criterion = nn.BCELoss()

d_losses = []
g_losses = []

# Training loop
for epoch in range(num_epochs):
    for i, (images, _) in enumerate(train_loader):
        # Get the current batch size (it may be different from the defined batch size for the last batch)
        current_batch_size = images.size(0)

        # Prepare real images
        real_images = images.view(current_batch_size, -1).to(device)
        real_labels = torch.ones(current_batch_size, 1).to(device)

        # Train Discriminator with real images
        optimizer_D.zero_grad()
        outputs = discriminator(real_images)
        d_loss_real = criterion(outputs, real_labels)
        d_loss_real.backward()

        # Generate fake images
        z = torch.randn(current_batch_size, latent_dim).to(device)
        fake_images = generator(z)
        fake_labels = torch.zeros(current_batch_size, 1).to(device)

        # Train Discriminator with fake images
        outputs = discriminator(fake_images.detach())
        d_loss_fake = criterion(outputs, fake_labels)
        d_loss_fake.backward()
        optimizer_D.step()

        d_loss = d_loss_real + d_loss_fake

        # Train Generator
        optimizer_G.zero_grad()
        outputs = discriminator(fake_images)
        g_loss = criterion(outputs, real_labels)
        g_loss.backward()
        optimizer_G.step()

        # Store losses
        d_losses.append(d_loss.item())
        g_losses.append(g_loss.item())

        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}')

# Plotting the comparison of G and D losses
plt.figure(figsize=(10, 5))
plt.title("Comparison of Generator and Discriminator Loss During Training")
plt.plot(d_losses, label="Discriminator")
plt.plot(g_losses, label="Generator")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()

# Plotting the discriminator losses
plt.figure(figsize=(10, 5))
plt.title("Discriminator Loss During Training")
plt.plot(d_losses, label="Discriminator Loss")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()

# Plotting the generator losses
plt.figure(figsize=(10, 5))
plt.title("Generator Loss During Training")
plt.plot(g_losses, label="Generator Loss")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()
# Save the models
torch.save(generator, 'generator.pt')
torch.save(generator.state_dict(), 'generator_weights.pt')

# Generate and plot images
np.random.seed(504)
num_generated = 100
z = np.random.normal(size=[num_generated, latent_dim])
z = torch.from_numpy(z).float().to(device)
generated_images = generator(z)
n = int(np.sqrt(num_generated))
fig, axes = plt.subplots(n, n, figsize=(8, 8))
for i in range(n):
    for j in range(n):
        axes[i, j].imshow(generated_images[i * n + j].cpu().detach().numpy().reshape(28, 28), cmap='gray')
        axes[i, j].axis('off')
plt.show()
