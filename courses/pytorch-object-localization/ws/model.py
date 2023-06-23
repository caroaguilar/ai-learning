from timm import create_model
from torch.nn import Module, MSELoss

class ObjLocModel(Module):

  def __init__(self):
    super(ObjLocModel, self).__init__()

    self.backbone = create_model(
      'efficientnet_b0',
      pretrained=True,
      num_classes=4 #num_classes = 4 = values (xmin, ymin, xmax, ymax)
    )

  def forward(self, images, gt_bboxes = None):
    bboxes = self.backbone(images) # bboxes = predicted bounding boxes

    if gt_bboxes != None:
      loss = MSELoss()(bboxes, gt_bboxes)
      return bboxes, loss

    return bboxes