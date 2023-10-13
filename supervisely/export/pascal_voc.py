# coding: utf-8

import os
from collections import defaultdict

from supervisely.io import fs as fs_utils
from supervisely.imaging import image as image_utils
from supervisely.project.project import Project
from supervisely.annotation.annotation import Annotation
from supervisely.geometry.rectangle import Rectangle

OUT_IMG_EXT = '.jpg'
XML_EXT = '.xml'
TXT_EXT = '.txt'


def save_project_as_pascal_voc_segmentation(save_path, project: Project):
    raise NotImplementedError


def save_images_lists(path, tags_to_lists):
    for tag_name, samples_desc_list in tags_to_lists.items():
        with open(os.path.join(path, tag_name + TXT_EXT), 'w') as fout:
            for record in samples_desc_list:
                fout.write('{}  {}\n'.format(record[0], record[1]))  # 0 - sample name, 1 - objects count


def save_project_as_pascal_voc_detection(save_path, project: Project):
    import pascal_voc_writer
    
    # Create root pascal 'datasets' folders
    for dataset in project.datasets:
        pascal_dataset_path = os.path.join(save_path, dataset.name)

        images_dir = os.path.join(pascal_dataset_path, 'JPEGImages')
        anns_dir = os.path.join(pascal_dataset_path, 'Annotations')
        lists_dir = os.path.join(pascal_dataset_path, 'ImageSets/Layout')

        fs_utils.mkdir(pascal_dataset_path)
        for subdir in ['ImageSets',  # Train list, Val list, etc.
                       'ImageSets/Layout',
                       'Annotations',
                       'JPEGImages']:
            fs_utils.mkdir(os.path.join(pascal_dataset_path, subdir))

        samples_by_tags = defaultdict(list)  # TRAIN: [img_1, img2, ..]

        for item_name in dataset:
            img_path, ann_path = dataset.get_item_paths(item_name)
            no_ext_name = fs_utils.get_file_name(item_name)
            pascal_img_path = os.path.join(images_dir, no_ext_name + OUT_IMG_EXT)
            pascal_ann_path = os.path.join(anns_dir, no_ext_name + XML_EXT)


            if item_name.endswith(OUT_IMG_EXT):
                fs_utils.copy_file(img_path, pascal_img_path)
            else:
                img = image_utils.read(img_path)
                image_utils.write(pascal_img_path, img)

            ann = Annotation.load_json_file(ann_path, project_meta=project.meta)

            # Read tags for images lists generation
            for tag in ann.img_tags:
                samples_by_tags[tag.name].append((no_ext_name ,len(ann.labels)))

            writer = pascal_voc_writer.Writer(path=pascal_img_path,
                                              width=ann.img_size[1],
                                              height=ann.img_size[0])

            for label in ann.labels:
                obj_class = label.obj_class
                rect: Rectangle = label.geometry.to_bbox()
                writer.addObject(name=obj_class.name,
                                 xmin = rect.left,
                                 ymin = rect.top,
                                 xmax = rect.right,
                                 ymax = rect.bottom)
            writer.save(pascal_ann_path)

        save_images_lists(lists_dir, samples_by_tags)
