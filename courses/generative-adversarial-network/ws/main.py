import torch
from tqdm import tqdm
from torchinfo import summary
from torch.utils.data import DataLoader
from torchvision import datasets, transforms as T

from generator import Generator
from discriminator import Discriminator
from functions import fake_loss, real_loss, weights_init


# The variables in this section can be adjusted to obtain different results
device = 'cuda'
batch_size = 128 # used in trailoader and training loop
noise_vector_dim = 64 # random noise vector size, used when creating the generator model

lr = 0.0002
beta_1 = 0.5
beta_2 = 0.99

epochs = 20

torch.manual_seed(42)


# Create training augmentation
train_augs = T.Compose([
    T.RandomRotation((-20, 20)),
    T.ToTensor(), # (h, w, c) => (c, h, w) to use channel-height-width convention
])

# Load MNIST Dataset
trainset = datasets.MNIST(
    'MNIST/',
    download=True,
    train=True,
    transform=train_augs
)

print('Total images in trainset:', len(trainset))

# Load Dataset Into Batches
trainloader = DataLoader(
    trainset,
    batch_size=batch_size,
    shuffle=True,
)

print('Total # of batches in trainloader:', len(trainloader))

dataiter = iter(trainloader)
images, label = next(dataiter)
print(images.shape)

# Instantiate Discriminator Network
D = Discriminator()
D.to(device)
D = D.apply(weights_init)
# summary(D, input_size=(1, 28, 28))

# Instantiate Generator Network
G = Generator(noise_vector_dim)
G.to(device)
G = G.apply(weights_init)
# summary(G, input_size=(1, noise_vector_dim))


D_optimizer = torch.optim.Adam(
    D.parameters(),  # D.paramaters = initial weights and biases
    lr=lr,
    betas=(beta_1, beta_2),
)

G_optimizer = torch.optim.Adam(
    G.parameters(),  # G.paramaters = initial weights and biases
    lr=lr,
    betas=(beta_1, beta_2),
)

# Training Loop
for i in range(epochs):
  total_discriminator_loss = 0.0
  total_generator_loss = 0.0

  for real_img, _ in tqdm(trainloader):
    real_img = real_img.to(device)
    noise = torch.randn(batch_size, noise_vector_dim, device=device)

    # find loss and update weights for D
    D_optimizer.zero_grad()
    fake_img = G(noise)
    D_prediction = D(fake_img)
    D_fake_loss = fake_loss(D_prediction)

    D_prediction = D(real_img)
    D_real_loss = real_loss(D_prediction)

    D_loss = (D_fake_loss + D_real_loss) / 2

    total_discriminator_loss += D_loss.item()

    D_loss.backward()
    D_optimizer.step()

    # find loss and update weights for Generator Network
    G_optimizer.zero_grad()
    noise = torch.randn(batch_size, noise_vector_dim, device=device)
    fake_img = G(noise)
    D_prediction = D(fake_img) # We want this prediction to be close to the real target that is 1
    G_loss = real_loss(D_prediction)

    total_generator_loss += G_loss.item()

    G_loss.backward() # find the gradients
    G_optimizer.step()

  # average the loss with total number of batches
  avg_d_loss = total_discriminator_loss / len(trainloader)
  avg_g_loss = total_generator_loss / len(trainloader)

  print('Epoch: {epoch}, Discriminator Loss: {d_loss}, Generator Loss: {g_loss}'.format(
    epoch=(i + 1),
    d_loss=avg_d_loss,
    g_loss=avg_g_loss
  ))

  # optional: show_tensor_images(fake_img)


"""
Notes
-----


Show image from dataset
***********************
image, label = trainset[9000]
plt.imshow(image.squeeze(), cmap = 'gray')


Run after training is completed
**********************************
Now you can use Generator Network to generate handwritten images

noise = torch.randn(batch_size, noise_vector_dim, device = device)
generated_image = G(noise)

show_tensor_images(generated_image)

"""