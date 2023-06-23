from torch import nn

"""
Network : Discriminator

input : (batch_size, 1, 28, 28) # (batch_size, number_of_channels, height, width) in this case number_of_channels = 1 because images are gray scale
      |                                                                                               ---- SUMMARY ----
      V
Conv2d( in_channels = 1, out_channels = 16, kernel_size = (3,3), stride = 2)                           #(batch_size, 16, 13, 13)
BatchNorm2d()                                                                                          #(batch_size, 16, 13, 13)
LeakyReLU()                                                                                            #(batch_size, 16, 13, 13)
      |
      V
Conv2d( in_channels = 16, out_channels = 32, kernel_size = (5,5), stride = 2)                          #(batch_size, 32, 5, 5)
BatchNorm2d()                                                                                          #(batch_size, 32, 5, 5)
LeakyReLU()                                                                                            #(batch_size, 32, 5, 5)
      |
      V
Conv2d( in_channels = 32, out_channels = 64, kernel_size = (5,5), stride = 2)                          #(batch_size, 64, 1, 1)
BatchNorm2d()                                                                                          #(batch_size, 64, 1, 1)
LeakyReLU()                                                                                            #(batch_size, 64, 1, 1)
      |
      V
Flatten()                                                                                              #(batch_size, 64)
Linear(in_features = 64, out_features = 1)                                                             #(batch_size, 1)
"""

def get_discriminator_block(in_channels, out_channels, kernel_size, stride):
  return nn.Sequential(
      nn.Conv2d(in_channels, out_channels, kernel_size, stride),
      nn.BatchNorm2d(out_channels),
      nn.LeakyReLU(0.2)
  )

class Discriminator(nn.Module):
  def __init__(self):
    super(Discriminator, self).__init__()

    self.block_1 = get_discriminator_block(1, 16, (3, 3), 2)
    self.block_2 = get_discriminator_block(16, 32, (5, 5), 2)
    self.block_3 = get_discriminator_block(32, 64, (5, 5), 2)

    self.flatten_layer = nn.Flatten()
    self.linear_layer = nn.Linear(in_features=64, out_features=1)

  def forward(self, images):
    x1 = self.block_1(images)
    x2 = self.block_2(x1)
    x3 = self.block_3(x2)

    x4 = self.flatten_layer(x3)
    x5 = self.linear_layer(x4)
    # Note: we're not using Sigmoid, we're gonna use binary cross entropy
    # with logic loss which takes raw outputs

    return x5
