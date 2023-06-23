from torch import nn

"""
Network : Generator

z_dim = 64
input : (batch_size, z_dim): input only includes batch_size and noise_vector_dim, and we need height a width so we will reshape it

      |
      | Reshape
      V

input : (batch_size, channel, height, width) -> (batch_size, z_dim , 1 , 1)
      |                                                                                               ---- SUMMARY ----
      V
ConvTranspose2d( in_channels = z_dim, out_channels = 256, kernel_size = (3,3), stride = 2)             #(batch_size, 256, 3, 3)
BatchNorm2d()                                                                                          #(batch_size, 256, 3, 3)
ReLU()                                                                                                 #(batch_size, 256, 3, 3)
      |
      V
ConvTranspose2d( in_channels = 256, out_channels = 128, kernel_size = (4,4), stride = 1)               #(batch_size, 128, 6, 6)
BatchNorm2d()                                                                                          #(batch_size, 128, 6, 6)
ReLU()                                                                                                 #(batch_size, 128, 6, 6)
      |
      V
ConvTranspose2d( in_channels = 128, out_channels = 64, kernel_size = (3,3), stride = 2)                #(batch_size, 64, 13, 13)
BatchNorm2d()                                                                                          #(batch_size, 64, 13, 13)
ReLU()                                                                                                 #(batch_size, 64, 13, 13)
      |
      V
ConvTranspose2d( in_channels = 64, out_channels = 1, kernel_size = (4,4), stride = 2)                  #(batch_size, 1, 28, 28)
Tanh()                                                                                                 #(batch_size, 1, 28, 28)
"""
noise_vector_dim = 64 # random noise vector size, used when creating the generator model

def get_generator_block(in_channels, out_channels, kernel_size, stride, final_block=False):
  if final_block == True:
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride),
        nn.Tanh()
    )

  return nn.Sequential(
      nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride),
      nn.BatchNorm2d(out_channels),
      nn.ReLU(0.2)
  )

class Generator(nn.Module):
  def __init__(self, noise_dim):
    super(Generator, self).__init__()

    self.noise_dim = noise_dim
    self.block_1 = get_generator_block(noise_vector_dim, 256, (3, 3), 2)
    self.block_2 = get_generator_block(256, 128, (4, 4), 1)
    self.block_3 = get_generator_block(128, 64, (3, 3), 2)
    self.block_4 = get_generator_block(64, 1, (4, 4), 2, final_block=True)


  def forward(self, random_noise_vector):
    # First change vector shape
    # (batch_size, noise_dim) -> (batch_size, noise_dim, 1, 1)
    x = random_noise_vector.view(-1, self.noise_dim, 1, 1)

    x1 = self.block_1(x)
    x2 = self.block_2(x1)
    x3 = self.block_3(x2)
    x4 = self.block_4(x3)

    return x4
