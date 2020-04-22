import os.path as osp
import os
import xml.etree.ElementTree as ET
import cv2
import scipy.misc as smisc
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from config import cfg
import random

idx_palette = np.reshape(np.asarray(cfg.class_color), (-1))
output_lbl_path=os.path.join(cfg.base_output_path, cfg.lbl_path)
kernel = np.ones((3,3), np.uint8)

def rotate_image(image, rotangle=45):
    if len(image.shape)==3: h, w, _ = image.shape
    else: h, w = image.shape
    M = cv2.getRotationMatrix2D((h/2, w/2), rotangle, 1)
    rotated_image = cv2.warpAffine(image, M, (w, h))
    return rotated_image

def readOneImage(fname, input_ann_path, input_rgb_path):
    rgbI = cv2.imread(osp.join(input_rgb_path, fname+'.jpg'))
    # annI = smisc.imread(osp.join(input_ann_path, fname+'.png'))
    # annI = np.array(Image.open(osp.join(input_ann_path, fname+'.png')))
    annI = cv2.cvtColor(cv2.imread(osp.join(input_ann_path, fname+'.png')), cv2.COLOR_BGR2RGB)
    return rgbI, annI

def read_files(ann_path):
    images=set()
    image_list=os.listdir(ann_path)
    print(image_list)

    for im in image_list:
        if im.strip().split('.')[-1]=='xml':
            im = '.'.join(im.strip().split('.')[:-1])
            images.add(im)
    return list(images)

def parse_xml_files(dirname, images):
    classes = {}
    
    imgtree = ET.parse(os.path.join(dirname, im+'.xml'))
    root = imgtree.getroot()

    for item in root.findall('classStack'):
        for child in item:
            classes[child[4].text.lower()] = convert_hex_to_rgb(child[3].text)
    return classes

def convert_hex_to_rgb(hex):
    hlen = len(hex)
    return tuple(int(hex[i:i+hlen//3], 16) for i in range(0, hlen, hlen//3))

def k_nearest_neighbors(classes, annI):
    ht, wd, _ = annI.shape
    mask = np.zeros([ht, wd])
    # colorI = np.zeros([ht, wd, 3])
    
    # check for every pixel in the image
    for r1 in range(ht):
        for c1 in range(wd):
            best = float('inf')
            chosencls = 0 
            for cname in classes:
                clabel = cfg.class_label[cname]
                dist = eucledian_dis(annI[r1][c1], np.array(classes[cname])) # check the nearest class of the pixel
                if dist < best: 
                    best=dist
                    chosencls=clabel
            mask[r1, c1]=chosencls
    
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask, create_color_image(mask, classes)

def create_color_image(mask, classes):
    ht, wd = mask.shape
    colorI = np.zeros([ht, wd, 3])
    # create the colored mask
    for cname in classes:
        clabel = cfg.class_label[cname]
        colorI[mask==clabel]=cfg.class_color[clabel]
    return colorI


def eucledian_dis(pt1, pt2):
    return np.linalg.norm(pt1-pt2)

def manhattan_dis(pt1, pt2):
    return np.sum(np.abs(pt1-pt2))

def create_output_dirs():
    if(not osp.exists(cfg.base_output_path)):
        os.makedirs(cfg.base_output_path, exist_ok=True)
    if(not osp.exists(output_lbl_path)):
        os.makedirs(output_lbl_path, exist_ok=True)
    if(not osp.exists(output_clr_path)):
        os.makedirs(output_clr_path, exist_ok=True)


if __name__ == '__main__':

    create_output_dirs()

    images = read_files(cfg.dirname) # read images from the input directory

    with open(cfg.output_list_name, 'w') as f:
        for im in images:
            f.write(im+"\n")
            print("Image: ", im)
            classes = parse_xml_files(cfg.dirname, im)
            classes['background']=(0, 0, 0)

            rgbI, annI = readOneImage(im, cfg.dirname, cfg.dirname)

            mask, colorI = k_nearest_neighbors(classes, annI)
            mask = mask.astype(np.uint8)

            save_labelI = Image.new('P', (annI.shape[1], annI.shape[0]))
            save_labelI.putpalette(list(idx_palette))
            save_labelI.paste(Image.fromarray(mask), (0,0))
            save_labelI.save(os.path.join(output_lbl_path, im+'.png'))

           