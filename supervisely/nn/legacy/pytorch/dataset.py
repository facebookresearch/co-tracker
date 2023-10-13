from supervisely.nn.dataset import SlyDataset

from torchvision.transforms.functional import to_tensor


class PytorchSegmentationSlyDataset(SlyDataset):
    """Thin wrapper around the base Supervisely dataset IO logic to handle PyTorch specific conversions.

    The base class handles locating and reading images and annotation data from disk, conversions between named classes
    and integer class ids to feed to the models, and rendering annotations as images.
    """

    def _get_sample_impl(self, img_fpath, ann_fpath):
        # Read the image, read the annotation, render the annotation as a bitmap of per-pixel class ids, resize to
        # requested model input size.
        img, gt = super()._get_sample_impl(img_fpath=img_fpath, ann_fpath=ann_fpath)

        # PyTorch specific logic:
        # - convert from (H x W x Channels) to (Channels x H x W);
        # - convert from uint8 to float32 data type;
        # - move from 0...255 to 0...1 intensity range.
        img_tensor = to_tensor(img)

        return img_tensor, gt