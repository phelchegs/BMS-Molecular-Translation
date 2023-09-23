import pandas as pd
import random
from tqdm.auto import tqdm
tqdm.pandas()
import re
from tqdm import tqdm
import numpy as np
import cv2
from albumentations import (
    Compose, OneOf, Normalize, Resize, HorizontalFlip, VerticalFlip, Rotate, RandomRotate90, CenterCrop
    )
from albumentations.pytorch import ToTensorV2
from InChI_extra_image_gen import add_noise


def split_form(text):
    PATTERN = re.compile('\d+|[A-Z][a-z]?|[^A-Za-z\d\/]|\/[a-z]')
    return ' '.join(re.findall(PATTERN, text))

def get_atom_counts(df):
    TARGETS = [
    'B', 'Br', 'C', 'Cl',
    'F', 'H', 'I', 'N',
    'O', 'P', 'S', 'Si']
    formula_regex = re.compile(r'[A-Z][a-z]?[0-9]*')
    element_regex = re.compile(r'[A-Z][a-z]?')
    number_regex = re.compile(r'[0-9]*')
    
    atom_dict_list = []
    for i in tqdm(df['Formula'].values):
        atom_dict = dict()
        for j in formula_regex.findall(i):
            atom = number_regex.sub("", j)
            dgts = element_regex.sub("", j)
            atom_cnt = int(dgts) if len(dgts) > 0 else 1
            atom_dict[atom] = atom_cnt
        atom_dict_list.append(atom_dict)

    atom_df = pd.DataFrame(atom_dict_list).fillna(0).astype(int)
    atom_df = atom_df.sort_index(axis = 1)
    for atom in TARGETS:
        df[atom] = atom_df[atom]
    return df

def train_file_path(image_id):
    #pay attention to the directory before /train, need to change accordingly.
    return "./bms-molecular-translation/train/{}/{}/{}/{}.png".format(
        image_id[0], image_id[1], image_id[2], image_id 
    )

#Two ways to treat the input images. 1.crop and pad to fit the images' size to be constant. 2.resize images to certain w and h. Here is the crop function.
def crop_image(img, 
               contour_min_pixel = 2, 
               small_stuff_size  = 2, 
               small_stuff_dist  = 5, 
               pad_pixels       = 5):
    
    # idea: pad with contour_min_pixels just in case we cut off
    #       a small part of the structure that is separated by a missing pixel
    
    #findContours only find white obj in black background color.
    img = 255 - img
    
    #Make all pixels except pure background, i.e. pure black, white and distinguish them using method BINARY and OTSU in order to not missing any obj. OTSU plus BINARY basically make the obj more distinguishable compared with just BINARY.
    _, thresh   = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    #RETR_LIST lists all the contours without hierarchy of nested contours. CHAIN_APPROX_SIMPLE returns only the key pixels that form the contour, e.g., 4 points for a rectangle contour.
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    
    #Store the small contours.
    small_stuff = []
    
    x_min0, y_min0, x_max0, y_max0 = np.inf, np.inf, 0, 0
    for i in contours:
        if len(i) < contour_min_pixel:  # if NO. of pixels is too small, ignore contours under contour_min_size pixels
            continue
        #x,y are the top-left coordinate of the rectangle and w, h are contour's width and heigh
        x, y, w, h = cv2.boundingRect(i)
        if w <= small_stuff_size and h <= small_stuff_size:  # collect position of contours which are smaller than small_stuff_size.
            small_stuff.append([x, y, x+w, y+h])
            continue
            
        #find the largest bounding rectangle.
        x_min0 = min(x_min0, x)
        y_min0 = min(y_min0, y)
        x_max0 = max(x_max0, x + w)
        y_max0 = max(y_max0, y + h)
        
    x_min, y_min, x_max, y_max = x_min0, y_min0, x_max0, y_max0
    
    # enlarge the found crop box if it cuts out small stuff that is very close by
    for i in range(len(small_stuff)):
        
        #if the small stuff overlap with the big obj, count the small stuff into the obj, update the xmin max ymin max with the small stuff's.
        if small_stuff[i][0] < x_min0 and small_stuff[i][0] + small_stuff_dist >= x_min0:
             x_min = small_stuff[i][0]
        if small_stuff[i][1] < y_min0 and small_stuff[i][1] + small_stuff_dist >= y_min0:
             y_min = small_stuff[i][1]
        if small_stuff[i][2] > x_max0 and small_stuff[i][2] - small_stuff_dist <= x_max0:
             x_max = small_stuff[i][2]
        if small_stuff[i][3] > y_max0 and small_stuff[i][3] - small_stuff_dist <= y_max0:
             y_max = small_stuff[i][3]
                             
    if pad_pixels > 0:  # make sure we get the crop within a valid range, pad_pixels is the range to ensure the crop is larger than the obj but not exceeding the canvas.
        y_min = max(0, y_min-pad_pixels)
        y_max = min(img.shape[0], y_max+pad_pixels)
        x_min = max(0, x_min-pad_pixels)
        x_max = min(img.shape[1], x_max+pad_pixels)
        
    img_cropped = img[y_min:y_max, x_min:x_max]
    
    #flip the black/white colors.
    # img_cropped = 255 - img_cropped
    
    return img_cropped

def pad_image(image, desired_size):
    h, w = image.shape[0], image.shape[1]
    delta_h = desired_size - h
    delta_w = desired_size - w
    top, bottom = delta_h//2, delta_h - (delta_h//2)
    left,right = delta_w//2, delta_w - (delta_w//2)
    img_padded = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT,
    value = [255, 255, 255])
    return img_padded

def preprocess_train_images(train_df, transform, CFG):
    #Goal of this func is to make all images the same size to fit the transformer model (crop and pad),
    #create a new column 'image' to record the original image data and the transformed image data if the trans flag is 'rotate90 or verticalflip'.
    #Here only one transformation is prepared because of the preliminary feeling that the scale of dataset is enough.
    assert set(['InChI_text', 'file_path', 'text_length']).issubset(train_df.columns), 'make sure the df has been preprocessed and certain columns are created.'
    
    trans_img = []
    ori_img = []
    transform_type = ['rotate90', 'verticalflip']
    df = train_df.copy()
    resize = Compose([Resize(CFG.image_size, CFG.image_size)])
    
    for i in tqdm(range(len(train_df))):
        img_path = train_df.loc[i, 'file_path']
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if CFG.crop == True:
            image = crop_image(image, 
               contour_min_pixel = 2, 
               small_stuff_size  = 2, 
               small_stuff_dist  = 5, 
               pad_pixels       = 10)
        image = resize(image = image)['image']
        image = add_noise(image)
        #np.expand_dims is used here because the input images needs to have 3 dimensions with the last one as 1.
        #But imread(cv2.IMREAD_GRAYSCALE) can only give a 2D image.
        image = np.expand_dims(image, axis = -1)
        ori_img.append(image)
        if CFG.trans_type == 'rotate90 or verticalflip':
            trans_image = transform(transform_type[random.randint(0, 1)])(image = image)['image']
            trans_img.append(trans_image)
    df.insert(3, 'image', ori_img)
    if CFG.trans_type == 'rotate90 or verticalflip':
        train_df['image'] = trans_img
        temp = pd.concat([df, train_df]).sample(frac = 1).reset_index(drop = True)
        return temp
    else:
        return df
    
def get_transform(trans_type):
    #transform images, need to annotate trans flag.
    if trans_type == 'rotate90':
        return Compose([
            OneOf([
                Rotate([90, 90], p = 0.5),
                Rotate([-90, -90], p = 0.5),
            ], p = 1.0),
        ])
    elif trans_type == 'verticalflip':
        return Compose([
            OneOf([
                VerticalFlip()
            ], p = 1.0),
        ])
    
def get_aug(CFG):
    #the goal is to normalize the image data and convert np array to torch tensor before sending to the model
    return Compose([Normalize(mean = CFG.pixels_mean, std = CFG.pixels_std), ToTensorV2()])
