from torch import nn, ones_like, zeros_like
from matplotlib import pyplot as plt
from torchvision.utils import make_grid

# Used to plot some of images from the batch
def show_tensor_images(tensor_img, num_images = 16, size=(1, 28, 28)):
    unflat_img = tensor_img.detach().cpu()
    img_grid = make_grid(unflat_img[:num_images], nrow=4)
    plt.imshow(img_grid.permute(1, 2, 0).squeeze())
    plt.show()


# Replace Random initialized weights with Normal weights for more robust training
def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight, 0.0, 0.02)
        nn.init.constant_(m.bias, 0)


def real_loss(discriminator_prediction):
  criterion = nn.BCEWithLogitsLoss()
  ground_truth = ones_like(discriminator_prediction)
  loss = criterion(discriminator_prediction, ground_truth)

  return loss


def fake_loss(discriminator_prediction):
  criterion = nn.BCEWithLogitsLoss()
  ground_truth = zeros_like(discriminator_prediction)
  loss = criterion(discriminator_prediction, ground_truth)

  return loss
