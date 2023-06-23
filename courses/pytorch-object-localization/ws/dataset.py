from pathlib import Path

from torch import from_numpy, Tensor
from torch.utils.data import Dataset
from cv2 import cvtColor, imread, COLOR_BGR2RGB

class ObjLocDataset(Dataset):

  def __init__(self, df, augmentations = None):
    self.df = df
    self.augmentations = augmentations

  def __len__(self):
    return len(self.df)

  def __getitem__(self, index):
    row = self.df.iloc[index]

    xmin = row.xmin
    ymin = row.ymin
    xmax = row.xmax
    ymax = row.ymax

    # Albumentation requires the bounding box with this format
    bbox = [[xmin, ymin, xmax, ymax]]

    img_path = 'object-localization-dataset/{}'.format(row.img_path)
    img = cvtColor(
     imread(img_path),
      COLOR_BGR2RGB
    )

    if self.augmentations:
      data = self.augmentations(
        image=img,
        bboxes=bbox,
        class_labels=[None]
      )
      img = data['image']
      bbox = data['bboxes'][0]

    # Convert to Torch tensor
    # PyTorch uses channel, height, weight, convention,
    # so we need to shift our channel axis to the 0 axis and then scale the image
    # permute: (h, w, c) -> (c, h, w)
    img = from_numpy(img).permute(2, 0, 1) / 255

    # Convert bounding box to torch tensor
    bbox = Tensor(bbox)

    return img, bbox