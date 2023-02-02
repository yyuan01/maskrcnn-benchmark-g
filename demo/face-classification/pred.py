from __future__ import print_function
import argparse
import os
import face_recognition
import numpy as np
import sklearn
import pickle
from face_recognition import face_locations
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import cv2
import pandas as pd
import csv
import matplotlib.pyplot as plt
import PIL.Image
# 源代码 https://github.com/wondonghyeon/face-classification/blob/master/pred.py
# we are only going to use 4 attributes
# CSV_FILE = 'image_features_race_images_priority.csv'
# INPUT_DIR = "/shared/yuanyuan/images_priority"
# Sample Command:
# CUDA_VISIBLE_DEVICES=7 python pred.py --model face_model.pkl --input_dir /shared/yuanyuan/parallel/images/dir_001 --csv_file 'image_features_race_dir_001.csv'
parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str,
                    default='test/', required = True,
                    help='input image directory (default: test/)')
parser.add_argument('--output_dir', type=str,
                    default='results/',
                    help='output directory to save the results (default: results/')
parser.add_argument('--model', type=str,
                    default='face_model.pkl', required = True,
                    help='path to trained model (default: face_model.pkl)')
parser.add_argument('--csv_file', type=str,
                    default='image_features_race_testing.csv',
                    help='output csv name')

args = parser.parse_args()
INPUT_DIR = args.input_dir
output_dir = args.output_dir
model_path = args.model
CSV_FILE = args.csv_file

COLS = ['Male', 'Asian', 'White', 'Black']
CATEGORIES = ['HASH', 'Asian_count', 'White_count', 'Black_count', 'Male', 'Asian', 'White', 'Black', 'Baby', 
'Child', 'Youth', 'Middle Aged', 'Senior', 'Black Hair', 'Blond Hair', 'Brown Hair', 'Bald', 'No Eyewear', 'Eyeglasses', 
'Sunglasses', 'Mustache', 'Smiling', 'Frowning', 'Chubby', 'Blurry', 'Harsh Lighting', 'Flash', 'Soft Lighting', 
'Outdoor', 'Curly Hair', 'Wavy Hair', 'Straight Hair', 'Receding Hairline', 'Bangs', 'Sideburns', 'Fully Visible Forehead', 
'Partially Visible Forehead', 'Obstructed Forehead', 'Bushy Eyebrows', 'Arched Eyebrows', 'Narrow Eyes', 'Eyes Open', 
'Big Nose', 'Pointy Nose', 'Big Lips', 'Mouth Closed', 'Mouth Slightly Open', 'Mouth Wide Open', 'Teeth Not Visible', 
'No Beard', 'Goatee', 'Round Jaw', 'Double Chin', 'Wearing Hat', 'Oval Face', 'Square Face', 'Round Face', 'Color Photo', 
'Posed Photo', 'Attractive Man', 'Attractive Woman', 'Indian', 'Gray Hair', 'Bags Under Eyes', 'Heavy Makeup', 'Rosy Cheeks',
 'Shiny Skin', 'Pale Skin', '5 o Clock Shadow', 'Strong Nose-Mouth Lines', 'Wearing Lipstick', 'Flushed Face', 
 'High Cheekbones', 'Brown Eyes', 'Wearing Earrings', 'Wearing Necktie', 'Wearing Necklace', 'top', 'right', 'bottom', 'left'] 
 # len = 81 = 73 (raw size of raw = clf.predict_proba(face_encodings)) + 8 front 4 and back 4 
N_UPSCLAE = 1
count = 0
print("len CATEGORIES", len(CATEGORIES))
def extract_features(img_path):
    """Exctract 128 dimensional features
    """
    X_img = face_recognition.load_image_file(img_path)
    img_name = os.path.basename(img_path)
    # print("X_img: ", X_img, X_img.shape)
    plt.imsave('./gan_imgs/' + str(img_name)[:-4] + '.png', X_img)
    locs = face_locations(X_img, number_of_times_to_upsample = N_UPSCLAE)
    # print("locs: ", locs)
    if len(locs) == 0:
        return None, None
    face_encodings = face_recognition.face_encodings(X_img, known_face_locations=locs)
    # print("face_encodings: ", np.array(face_encodings), np.array(face_encodings).shape)
    # plt.imsave('face2_' + str(count) + '.png', face_encodings)

    face_image = Image.fromarray(X_img)
    face_count = 0 # For future use
    face = face_image.crop((locs[0][3]-15, locs[0][0]-25, locs[0][1]+15, locs[0][2]+15))
    # face.save("./gan_faces/" + str(img_name)[:-4] + "_face.png", "PNG")
    return face_encodings, locs

def predict_one_image(img_path, clf, labels):
    """Predict face attributes for all detected faces in one image
    """
    face_encodings, locs = extract_features(img_path) 
    # face_encodings vactorize face to 128 dimension vector
    if not face_encodings:
        return None, None
    print("face_encodings locs", face_encodings, locs, face_encodings[0].shape)
    raw = clf.predict_proba(face_encodings)    
    print("raw ", raw, len(raw[0]), sum(raw[0]))
    print("labels ", labels)
    pred = pd.DataFrame(clf.predict_proba(face_encodings),
                        columns = labels) # labels are CATEGORIES - 8
    # print("pred:")
    # print(pred)
    pred.to_csv("./testing.csv")
    # pred = pred.loc[:, COLS]
    return pred, locs
def draw_attributes(img_path, df):
    csv_file = CSV_FILE
    """Write bounding boxes and predicted face attributes on the image
    """

    img = cv2.imread(img_path)
    # img  = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    # print(img_path)
    # print(df)

    race_feature_dict = dict.fromkeys(CATEGORIES, 0)
    for row in df.iterrows():
        # print("row: ", row[1], len(row[1]), row[1][73:])
        top, right, bottom, left = row[1][73:].astype(int)
        if row[1]['Male'] >= 0.5:
            gender = 'Male'
        else:
            gender = 'Female'
        race = np.argmax(row[1][1:4])
        text_showed = "{} {}".format(race, gender)

        race_feature_dict[race + "_count"] += 1
        race_feature_dict['HASH'] = os.path.basename(img_path)[:-4]

        # print(race)
        # print("row:")
        # print(row[1][1:4].to_dict())
        # print(race_feature_dict)
        cv2.rectangle(img, (left, top), (right, bottom), (0, 0, 255), 2)
        font = cv2.FONT_HERSHEY_DUPLEX
        img_width = img.shape[1]
        cv2.putText(img, text_showed, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

    for row in df.iterrows():
        # score_dict = row[1][1:4].to_dict()
        score_dict = row[1].to_dict()
        # print("score_dict: ", score_dict)
        # add scores:
        # race_feature_dict['asian_score'] = score_dict['Asian']
        # race_feature_dict['white_score'] = score_dict['White']
        # race_feature_dict['black_score'] = score_dict['Black']
        race_feature_dict = {**race_feature_dict, **score_dict}
        # print("race_feature_dict", race_feature_dict)
        with open(csv_file, "a") as f:
                writer = csv.DictWriter(f, fieldnames=CATEGORIES)
                writer.writerow(race_feature_dict)

    return img



def main():
    csv_file = CSV_FILE
    input_dir = INPUT_DIR
    # input_dir = '/home/yuanyuan/maskrcnn-benchmark/demo/sample_png/'
    # input_dir = "/home/yuanyuan/kiva/data/images"
    # input_dir = args.img_dir

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    # load the model
    with open(model_path, 'rb') as f:
        # clf, labels = pickle.load(f)
        clf, labels = pickle.load(f, encoding='latin1')
        print("clf, labels", clf, labels)
    # Add central csv file:  
    
    categories = CATEGORIES
    with open(csv_file, 'w') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=categories)
        writer.writeheader()

    print("classifying images in {}".format(input_dir))
    for fname in tqdm(sorted(os.listdir(input_dir))):
        img_path = os.path.join(input_dir, fname)
        # try:
        pred, locs = predict_one_image(img_path, clf, labels)
        # except:
        #     print("Skipping {}".format(img_path))
        if not locs:
            continue
        locs = \
            pd.DataFrame(locs, columns = ['top', 'right', 'bottom', 'left'])
        df = pd.concat([pred, locs], axis=1)
        img = draw_attributes(img_path, df)
        # cv2.imwrite(os.path.join(output_dir, fname), img)
        os.path.splitext(fname)[0]
        output_csvpath = os.path.join(output_dir,
                                      os.path.splitext(fname)[0] + '.csv')
    
        # df.to_csv(output_csvpath, index = False)

if __name__ == "__main__":
    main()
