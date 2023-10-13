import inspect
from collections import OrderedDict
from supervisely.sly_logger import logger
from supervisely.annotation.annotation import Annotation
from supervisely.project.project_meta import ProjectMeta
try:
    from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
    from imgaug.augmentables.segmaps import SegmentationMapsOnImage
except:
    pass
import numpy as np


if not hasattr(np, 'bool'): np.bool = np.bool_

def create_aug_info(category_name, aug_name, params, sometimes: float = None):
    clean_params = params
    #clean_params = remove_unexpected_arguments(category_name, aug_name, params)
    res = {
        "category": category_name,
        "name": aug_name,
        "params": clean_params,
    }
    if sometimes is not None:
        if type(sometimes) is not float or not (0.0 <= sometimes <= 1.0):
            raise ValueError(f"sometimes={sometimes}, type != {type(sometimes)}")
        res["sometimes"] = sometimes
    res["python"] = aug_to_python(res)
    return res


def aug_to_python(aug_info):
    pstr = ""
    for name, value in aug_info["params"].items():
        v = value
        if isinstance(v, list) and len(v) == 2:  #name != 'nb_iterations' and
            v = tuple(v)
        elif isinstance(v, dict) and "x" in v.keys() and "y" in v.keys():
            x = v["x"]
            y = v["y"]
            if isinstance(x, list) and len(x) == 2:
                x = tuple(x)
            if isinstance(y, list) and len(y) == 2:
                y = tuple(y)
            v = {"x": x, "y": y}
            
        if type(value) is str:
            pstr += f"{name}='{v}', "
        else:
            pstr += f"{name}={v}, "
    method_py = f"iaa.{aug_info['category']}.{aug_info['name']}({pstr[:-2]})"

    res = method_py
    if "sometimes" in aug_info:
        res = f"iaa.Sometimes({aug_info['sometimes']}, {method_py})"
    return res


def pipeline_to_python(aug_infos, random_order=False):
    template = \
"""import imgaug.augmenters as iaa

seq = iaa.Sequential([
{}
], random_order={})
"""
    py_lines = []
    for info in aug_infos:
        line = aug_to_python(info)
        _validate = info["python"]
        if line != _validate:
            raise ValueError("Generated python line differs from the one from config: \n\n{!r}\n\n{!r}"
                             .format(line, _validate))
        py_lines.append(line)
    res = template.format('\t' + ',\n\t'.join(py_lines), random_order)
    return res


def get_default_params_by_function(f):
    params = []
    method_info = inspect.signature(f)
    for param in method_info.parameters.values():
        formatted = str(param)
        if 'deprecated' in formatted or 'seed=None' in formatted or 'name=None' in formatted:
            continue
        if param.default == inspect._empty:
            continue
        params.append({
            "pname": param.name,
            "default": param.default
        })
    return params


def get_default_params_by_name(category_name, aug_name):
    func = get_function(category_name, aug_name)
    defaults = get_default_params_by_function(func)
    return defaults


def get_function(category_name, aug_name):
    try:
        import imgaug.augmenters as iaa
    except ModuleNotFoundError as e:
        logger.error(f'{e}. Try to install extra dependencies. Run "pip install supervisely[aug]"')
        raise e
    try:
        submodule = getattr(iaa, category_name)
        aug_f = getattr(submodule, aug_name)
        return aug_f
    except Exception as e:
        logger.error(repr(e))
        # raise e
        return None


def build_pipeline(aug_infos, random_order=False):
    try:
        import imgaug.augmenters as iaa
    except ModuleNotFoundError as e:
        logger.error(f'{e}. Try to install extra dependencies. Run "pip install supervisely[aug]"')
        raise e
    pipeline = []
    for aug_info in aug_infos:
        category_name = aug_info["category"]
        aug_name = aug_info["name"]
        params = aug_info["params"]

        aug_func = get_function(category_name, aug_name)

        for param_name, param_val in params.items():
            if isinstance(param_val, dict):
                if "x" in param_val.keys() and "y" in param_val.keys():
                    if isinstance(param_val["x"], list) and len(param_val["x"]) == 2:
                        param_val["x"] = tuple(param_val["x"])
                    if isinstance(param_val["y"], list) and len(param_val["y"]) == 2:
                        param_val["y"] = tuple(param_val["y"])
            elif isinstance(param_val, list) and len(param_val) == 2:
                # all {'par': [n1, n2]} to {'par': (n1, n2)}
                params[param_name] = tuple(param_val)

        aug = aug_func(**params)

        sometimes = aug_info.get("sometimes", None)
        if sometimes is not None:
            aug = iaa.meta.Sometimes(sometimes, aug)
        pipeline.append(aug)
    augs = iaa.Sequential(pipeline, random_order=random_order)
    return augs


def build(aug_info):
    return build_pipeline([aug_info])


def remove_unexpected_arguments(category_name, aug_name, params):
    # to avoid this:
    # TypeError: f() got an unexpected keyword argument 'b'
    defaults = get_default_params_by_name(category_name, aug_name)
    allowed_names = [d["pname"] for d in defaults]

    res = OrderedDict()
    for name, value in params.items():
        if name in allowed_names:
            res[name] = value
    return res


def _apply(augs, img, boxes=None, masks=None):
    try:
        import imgaug.augmenters as iaa
    except ModuleNotFoundError as e:
        logger.error(f'{e}. Try to install extra dependencies. Run "pip install supervisely[aug]"')
        raise e
    augs: iaa.Sequential
    res = augs(images=[img], bounding_boxes=boxes, segmentation_maps=masks)
    #return image, boxes, masks
    return res[0][0], res[1], res[2]


def apply(augs, meta: ProjectMeta, img, ann: Annotation, segmentation_type='semantic'):
    # @TODO: save object tags

    # works for rectangles
    det_meta, det_mapping = meta.to_detection_task(convert_classes=False)
    det_ann = ann.to_detection_task(det_mapping)
    ia_boxes = det_ann.bboxes_to_imgaug()

    # works for polygons and bitmaps
    seg_meta, seg_mapping = meta.to_segmentation_task()
    seg_ann = ann.to_nonoverlapping_masks(seg_mapping)

    if segmentation_type == 'semantic':
        seg_ann = seg_ann.to_segmentation_task()
        class_to_index = {obj_class.name: idx for idx, obj_class in enumerate(seg_meta.obj_classes, start=1)}
        index_to_class = {v: k for k, v in class_to_index.items()}
    elif segmentation_type == 'instance':
        class_to_index = None
        index_to_class = {idx: label.obj_class.name for idx, label in enumerate(seg_ann.labels, start=1)}
    elif segmentation_type == 'panoptic':
        raise NotImplementedError

    ia_masks = seg_ann.masks_to_imgaug(class_to_index)

    res_meta = det_meta.merge(seg_meta)  # TagMetas should be preserved

    res_img, res_ia_boxes, res_ia_masks = _apply(augs, img, ia_boxes, ia_masks)
    res_ann = Annotation.from_imgaug(res_img,
                                     ia_boxes=res_ia_boxes, ia_masks=res_ia_masks,
                                     index_to_class=index_to_class, meta=res_meta)
    # add image tags
    res_ann = res_ann.clone(img_tags=ann.img_tags)
    return res_meta, res_img, res_ann


def apply_to_image(augs, img):
    res_img, _, _ = _apply(augs, img, None, None)
    return res_img


def _instances_to_nonoverlapping_mask(instance_masks):
    """Convert instance segmentation masks to nonoverlapping objects on
    semantic-like segmentation mask.

    Parameters
    ----------
    instance_masks : (H,W,C) ndarray, UINT
        Instance segmentation masks.

    Returns
    -------
    (H,W) ndarray
        Segmentation map with all instances.

    """
    common_img = np.zeros(instance_masks.shape[:2], np.int32)
    n_instances = instance_masks.shape[2]

    for idx in range(n_instances):
        common_img[instance_masks[:,:,idx].astype(np.bool)] = idx + 1
    
    return common_img

def _nonoverlapping_mask_to_instances(common_mask, n_instances):
    """Convert nonoverlapping objects on semantic-like segmentation mask to 
    instance segmentation masks.

    Parameters
    ----------
    common_mask : (H,W) ndarray, UINT
        Segmentation map with all instances.

    n_instances : int
        Number of objects on segmentation map. After applying augmentations some of instances 
        may disappear from image. In this case result array may contain empty channels.

    Returns
    -------
    (H,W,C) ndarray
        Instance segmentation masks.

    """
    bitmap_masks = []
    for idx in range(n_instances):
        instance_mask = common_mask == idx + 1
        bitmap_masks.append(instance_mask.astype(np.uint8))
    return np.stack(bitmap_masks, axis=-1)


def apply_to_image_and_bbox(augs, img, bboxes):
    """Apply augmentations to image and bounding boxes. 

    Parameters
    ----------
    augs : iaa.Sequential
        Defined imgaug augmentation sequence.
    
    img : (H, W) or (H, W, C) ndarray
        Input image.

    bboxes : List[bbox], where bbox: list or tuple of bbox coords in XYXY format
        Bounding boxes.

    Returns
    -------
    (H,W,C) ndarray
        Augmented image.

    List[bbox], where bbox: list of bbox coords in XYXY format
        Augmented bounding boxes.
        
    """
    assert isinstance(bboxes, list)

    ia_boxes = [BoundingBox(box[0], box[1], box[2], box[3]) for box in bboxes]
    ia_boxes = BoundingBoxesOnImage(ia_boxes, shape=img.shape[:2])
    res_img, res_boxes, _ = _apply(augs, img, boxes=ia_boxes)
    res_boxes = [[res_box.x1, res_box.y1, res_box.x2, res_box.y2] for res_box in res_boxes]

    return res_img, res_boxes
    

def apply_to_image_and_mask(augs, img, mask, segmentation_type="semantic"):
    """Apply augmentations to image and segmentation mask (instance or semantic). 

    Parameters
    ----------
    augs : iaa.Sequential
        Defined imgaug augmentation sequence.
    
    img : (H, W) or (H, W, C) ndarray
        Input image.

    mask : (H, W) ndarray if semantic or (H, W, C) ndarray if instance
        Instance or semantic segmentation mask.
        If segmentation_type=='semantic', shape must be (H, W).
        If segmentation_type=='instance', shape must be (H, W, C), where
        C is num_objects, C > 0.

    segmentation_type : str, one of ('semantic', 'instance')
        Define how to process segmentation masks.

    Returns
    -------
    (H,W,C) ndarray
        Augmented image.

    (H,W) ndarray if semantic or (H,W,C) ndarray if instance
        Augmented segmentation mask.

    """
    assert isinstance(segmentation_type, str) and segmentation_type in ["semantic", "instance"]
    assert isinstance(mask, np.ndarray)
    assert mask.shape[:2] == img.shape[:2]

    if segmentation_type == "instance":
        N_instances = mask.shape[2]
        mask = _instances_to_nonoverlapping_mask(mask) # [H,W,C] -> [H,W]
    
    segmaps = SegmentationMapsOnImage(mask, shape=img.shape[:2])
    
    res_img, _, res_segmaps = _apply(augs, img, masks=segmaps)

    res_mask = res_segmaps.get_arr()
    if segmentation_type == "instance":
        res_mask = _nonoverlapping_mask_to_instances(res_mask, N_instances) # [H,W] -> [H,W,C]
    
    if res_img.shape[:2] != res_mask.shape[:2]:
        raise ValueError(f"Image and mask have different shapes "
                         f"({res_img.shape[:2]} != {res_mask.shape[:2]}) after augmentations. "
                         f"Please, contact tech support")
    return res_img, res_mask


def apply_to_image_bbox_and_mask(augs, img, bboxes, mask, segmentation_type="semantic"):
    """Apply augmentations to image, bounding boxes and segmentation mask 
    (instance or semantic). 

    Parameters
    ----------
    augs : iaa.Sequential
        Defined imgaug augmentation sequence.
    
    img : (H, W) or (H, W, C) ndarray
        Input image.

    bboxes : List[bbox], where bbox: list or tuple of bbox coords in XYXY format
        Bounding boxes.

    mask : (H, W) ndarray if semantic or (H, W, C) ndarray if instance
        Instance or semantic segmentation mask.
        If segmentation_type=='semantic', shape must be (H, W).
        If segmentation_type=='instance', shape must be (H, W, C), where
        C is num_objects, C > 0.

    segmentation_type : str, one of ('semantic', 'instance')
        Define how to process segmentation masks.

    Returns
    -------
    (H,W,C) ndarray
        Augmented image.

    List[bbox], where bbox: list of bbox coords in XYXY format
        Augmented bounding boxes.

    (H,W) ndarray if semantic or (H,W,C) ndarray if instance
        Augmented segmentation mask.

    """
    assert isinstance(bboxes, list)
    assert isinstance(segmentation_type, str) and segmentation_type in ["semantic", "instance"]
    assert isinstance(mask, np.ndarray)
    assert img.shape[:2] == mask.shape[:2]

    if segmentation_type == "semantic":
        assert mask.ndim == 2
    elif segmentation_type == "instance":
        assert mask.ndim == 3

    boxes = [BoundingBox(box[0], box[1], box[2], box[3]) for box in bboxes]
    boxes = BoundingBoxesOnImage(boxes, shape=img.shape[:2])
    
    if segmentation_type == "instance":
        N_instances = mask.shape[2]
        mask = _instances_to_nonoverlapping_mask(mask) # [H,W,C] -> [H,W]
    segmaps = SegmentationMapsOnImage(mask, shape=mask.shape)

    res_img, res_boxes, res_segmaps = _apply(augs, img, boxes=boxes, masks=segmaps)

    res_mask = res_segmaps.get_arr()
    if segmentation_type == "instance":
        res_mask = _nonoverlapping_mask_to_instances(res_mask, N_instances) # [H,W] -> [H,W,C]
    res_boxes = [[res_box.x1, res_box.y1, res_box.x2, res_box.y2] for res_box in res_boxes]

    return res_img, res_boxes, res_mask


def apply_to_image_bbox_and_both_types_masks(augs, img, bboxes, semantic_mask, instance_masks):
    """Apply augmentations to image, bounding boxes and both types of segmentation masks 
    (instance and semantic together). 

    Parameters
    ----------
    augs : iaa.Sequential
        Defined imgaug augmentation sequence.
    
    img : (H,W) or (H,W,C) ndarray
        Input image.

    bboxes : List[bbox], where bbox: list or tuple of bbox coords in XYXY format
        Bounding boxes.

    semantic_mask : (H,W) ndarray
        Semantic segmentation mask.

    instance_masks : (H,W,C) ndarray
        Instance segmentation masks. 
        C is num_objects, C > 0.

    Returns
    -------
    (H,W) or (H,W,C) ndarray, UINT
        Augmented image.

    List[bbox], where bbox: list of bbox coords in XYXY format
        Augmented bounding boxes.

    (H,W) ndarray, UINT
        Augmented semantic segmentation mask.
    
    (H,W,C) ndarray, UINT
        Augmented instance segmentation masks.

    """
    assert isinstance(bboxes, list)
    assert isinstance(semantic_mask, np.ndarray) and semantic_mask.ndim == 2
    assert isinstance(instance_masks, np.ndarray) and instance_masks.ndim == 3
    assert img.shape[:2] == semantic_mask.shape[:2] == instance_masks.shape[:2]

    boxes = [BoundingBox(box[0], box[1], box[2], box[3]) for box in bboxes]
    boxes = BoundingBoxesOnImage(boxes, shape=img.shape[:2])

    N_instances = instance_masks.shape[2]
    merged_instance_masks = _instances_to_nonoverlapping_mask(instance_masks) # [H,W,C] -> [H,W]
    # merge instance and segmentation masks as different channels of one mask
    all_masks = np.stack((merged_instance_masks, semantic_mask, semantic_mask), axis=-1)
    segmaps = SegmentationMapsOnImage(all_masks, shape=all_masks.shape)

    res_img, res_boxes, res_segmaps = _apply(augs, img, boxes=boxes, masks=segmaps)

    res_masks = res_segmaps.get_arr()
    res_semantic = res_masks[:,:,2]
    res_instances = _nonoverlapping_mask_to_instances(res_masks[:,:,0], N_instances) # [H,W] -> [H,W,C]
    res_boxes = [[res_box.x1, res_box.y1, res_box.x2, res_box.y2] for res_box in res_boxes]

    return res_img, res_boxes, res_semantic, res_instances