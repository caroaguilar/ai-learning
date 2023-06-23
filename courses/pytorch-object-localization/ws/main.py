from albumentations import (
  Compose,
  Resize,
  HorizontalFlip,
  VerticalFlip,
  Rotate,
  BboxParams,
)

from torch import (
  rand,
  no_grad,
  save,
  load,
  Size,
)

from torch.utils.data import DataLoader
from torch.optim import Adam

from numpy import Inf
from pandas import read_csv
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from dataset import ObjLocDataset
from model import ObjLocModel
from utils import compare_plots


# The variables in this section can be adjusted to obtain different results
BATCH_SIZE = 16
IMG_SIZE = 140
LR = 0.001
EPOCHS = 40
DEVICE = 'cuda'

# Read CSV into DataFrame
df = read_csv('object-localization-dataset/train.csv')

# Define training and validation subset
train_df, valid_df = train_test_split(df, test_size = 0.20, random_state = 42)

# Define Augmentations
train_augs = Compose([
    Resize(IMG_SIZE, IMG_SIZE),
    HorizontalFlip(p=0.5),
    VerticalFlip(p=0.5),
    Rotate()
  ],
  bbox_params=BboxParams(format = 'pascal_voc', label_fields = ['class_labels'])
)

valid_augs = Compose([
    Resize(IMG_SIZE, IMG_SIZE),
  ],
  bbox_params=BboxParams(
    format='pascal_voc',
    label_fields=['class_labels'],
  )
)

# Create custom dataset
trainset = ObjLocDataset(train_df, train_augs)
validset = ObjLocDataset(valid_df, valid_augs)

print('Total examples in trainset: {}'.format(len(trainset)))
print('Total examples in validset: {}'.format(len(validset)))

# Load dataset into batches
trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
validloader = DataLoader(validset, batch_size=BATCH_SIZE, shuffle=False)

print("Total no. batches in trainloader : {}".format(len(trainloader)))
print("Total no. batches in validloader : {}".format(len(validloader)))


# Create custom model
model = ObjLocModel()
# Transfer model to GPU
model.to(DEVICE)

# rand(batch_size, num_of_channels, height, width)
random_img = rand(1, 3, 140, 140).to(DEVICE)
model(random_img).shape

###################################
# Define Train and Eval Functions #
###################################
def train_fn(model, dataloader, optimizer):
  total_loss = 0.0
  # specify model is in training mode, so Dropout layer is ON
  model.train()

  for data in tqdm(dataloader):
    images, gt_bboxes = data
    images, gt_bboxes = images.to(DEVICE), gt_bboxes.to(DEVICE)

    # get predicted bounding boxes and loss
    bboxes, loss = model(images, gt_bboxes)

    # gradient computation steps
    optimizer.zero_grad()
    loss.backward() # find the gradients
    optimizer.step() # update weights and the biases parameters of the model

    total_loss += loss.item()

  # return average loss
  return total_loss / len(dataloader)

def eval_fn(model, dataloader):
  total_loss = 0.0
  model.eval() # Dropout layer is OFF

  with no_grad(): # to make sure there's no gradient computation inside the eval function
    for data in tqdm(dataloader):
      images, gt_bboxes = data
      images, gt_bboxes = images.to(DEVICE), gt_bboxes.to(DEVICE)

      # get predicted bounding boxes and loss
      bboxes, loss = model(images, gt_bboxes)

      total_loss += loss.item()

    # return average loss
    return total_loss / len(dataloader)


#################
# Training Loop #
#################

optimizer = Adam(model.parameters(), lr=LR) # parameters = weights and biases
best_valid_loss = Inf

model_name = 'best_model.pt'

for i in range(EPOCHS):
  train_loss = train_fn(model, trainloader, optimizer)
  valid_loss = eval_fn(model, validloader)

  if valid_loss < best_valid_loss:
    save(model.state_dict(), model_name)
    print('Weigths are saved...')
    best_valid_loss = valid_loss

  print('Epoch: {epoch}, Train Loss: {tloss}, Valid Loss: {vloss}'.format(
    epoch=(i + 1),
    tloss=train_loss,
    vloss=valid_loss),
  )

#############
# Inference #
#############


model.load_state_dict(load(model_name))
model.eval()

with no_grad():
  image, gt_bbox = validset[12]
  # image is in this format (c, h, w), we need to unsqueeze to
  # include the batch_size in axis 0 (batch_size, c, h , w)
  image = image.unsqueeze(0).to(DEVICE)
  predicted_bbox = model(image)

  compare_plots(image, gt_bbox, predicted_bbox)
  # when plotting:
  #   - the green rectangle is the ground truth bounding box
  #   - the red rectangle is the predicted bounding box

"""
Notes
=====

Albumentations
--------------
  Reference https://albumentations.ai/
  Image augmentation library. Provides augmentation for various tasks
  like classification, segmentation, object detection, keypoints

Augmentation
-------------
  albumentations: We are going to use it for localization or detection task
  Augmentation for localization or detection is a bit different.
  If we use it in classification like the example above, if the image is rotated to a 20 degree, its label is going to be seen, it's label is going
  to be eggplant only, there will be no effect on the label if the image is rotated
  For detection/localization if the image it's rotated to 20 degrees, its bounding box will also be rotated, if the image is flipped vertically
  its bounding box will also be flipped vertically


Plot Image from trainset
------------------------
  from matplotlib import pyplot

  img, bbox = trainset[120]

  xmin, ymin, xmax, ymax = bbox

  pt1 = (int(xmin), int(ymin))
  pt2 = (int(xmax), int(ymax))

  bnd_img = cv2.rectangle(img.permute(1, 2, 0).numpy(),pt1, pt2,(255,0,0),2)
  pyplot.imshow(bnd_img)

Inspect shape of batches
------------------------
  for images, bboxes in trainloader:
    print("Shape of one batch images : {}".format(images.shape))
    print("Shape of one batch bboxes : {}".format(bboxes.shape))


  Shape of one batch images : Size([16, 3, 140, 140])
  16 = batch size
  3 = number of channels
  140 = height
  140 = width

  Shape of one batch bboxes : Size([16, 4])
  16 = batch size
  4 = values (xmin, ymin, xmax, ymax)


Understand the dataset
------------------------

Rendering an image from the Dataset
***********************************
- Mushroom
row = df.iloc[2] # select row
img = cv2.imread(DATA_DIR + row.img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

pt1 = (row.xmin, row.ymin)
pt2 = (row.xmax, row.ymax)

bound_box_img = cv2.rectangle(img, pt1, pt2, (255, 0 ,0), 2)
plt.imshow(bound_box_img)
