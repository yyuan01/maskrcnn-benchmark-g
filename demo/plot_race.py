import argparse
import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from scipy import stats
from itertools import permutations
from itertools import combinations
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns; sns.set()
import pycountry_convert as pc
from string import ascii_letters

# IMAGE_FEATURES = "/Users/yuanyuan/athey/kiva_namazu_code/demo/feature_csv/image_features.csv"
# IMAGE_FEATURES_RACE = "/Users/yuanyuan/athey/kiva_namazu_code/demo/feature_csv/image_features_race.csv"
# LOAN = "/Users/yuanyuan/athey/kiva_namazu/data/kiva_berkeley/loans.csv"
# MAP = "/Users/yuanyuan/athey/kiva_namazu/data/kiva_berkeley/imagehash_to_id_expired_default.csv"

CATEGORIES = [
        "__background",
        "person",
        "bicycle",
        "car",
        "motorcycle",
        "airplane",
        "bus",
        "train",
        "truck",
        "boat",
        "traffic light",
        "fire hydrant",
        "stop sign",
        "parking meter",
        "bench",
        "bird",
        "cat",
        "dog",
        "horse",
        "sheep",
        "cow",
        "elephant",
        "bear",
        "zebra",
        "giraffe",
        "backpack",
        "umbrella",
        "handbag",
        "tie",
        "suitcase",
        "frisbee",
        "skis",
        "snowboard",
        "sports ball",
        "kite",
        "baseball bat",
        "baseball glove",
        "skateboard",
        "surfboard",
        "tennis racket",
        "bottle",
        "wine glass",
        "cup",
        "fork",
        "knife",
        "spoon",
        "bowl",
        "banana",
        "apple",
        "sandwich",
        "orange",
        "broccoli",
        "carrot",
        "hot dog",
        "pizza",
        "donut",
        "cake",
        "chair",
        "couch",
        "potted plant",
        "bed",
        "dining table",
        "toilet",
        "tv",
        "laptop",
        "mouse",
        "remote",
        "keyboard",
        "cell phone",
        "microwave",
        "oven",
        "toaster",
        "sink",
        "refrigerator",
        "book",
        "clock",
        "vase",
        "scissors",
        "teddy bear",
        "hair drier",
        "toothbrush",
    ]

CATEGORIES_transportation = [
        "person",
        "bicycle",
        "car",
        "motorcycle",
        "airplane",
        "bus",
        "train",
        "truck",
        "boat",
        "traffic light",
        "fire hydrant",
        "stop sign",
        "parking meter",
        "bench",   
    ]

CATEGORIES_animal = [
        "person",
        "bird",
        "cat",
        "dog",
        "horse",
        "sheep",
        "cow",
        "elephant",
        "bear",
        "zebra",
        "giraffe",
    ]

CATEGORIES_travel_and_sport = [
        "person",
        "backpack",
        "umbrella",
        "handbag",
        "tie",
        "suitcase",
        "frisbee",
        "skis",
        "snowboard",
        "sports ball",
        "kite",
        "baseball bat",
        "baseball glove",
        "skateboard",
        "surfboard",
        "tennis racket",
    ]

CATEGORIES_food_and_dining = [
        "person",
        "bottle",
        "wine glass",
        "cup",
        "fork",
        "knife",
        "spoon",
        "bowl",
        "banana",
        "apple",
        "sandwich",
        "orange",
        "broccoli",
        "carrot",
        "hot dog",
        "pizza",
        "donut",
        "cake",
        "chair",
    ]


CATEGORIES_tech_and_household = [
        "person",
        "chair",
        "couch",
        "potted plant",
        "bed",
        "dining table",
        "toilet",
        "tv",
        "laptop",
        "mouse",
        "remote",
        "keyboard",
        "cell phone",
        "microwave",
        "oven",
        "toaster",
        "sink",
        "refrigerator",
        "book",
        "clock",
        "vase",
        "scissors",
        "teddy bear",
        "hair drier",
        "toothbrush",
    ]

IMAGE_FEATURES = "/home/yuanyuan/maskrcnn-benchmark/demo/feature_csv/image_features_full.csv"
# IMAGE_FEATURES_RACE = "/home/yuanyuan/maskrcnn-benchmark/demo/feature_csv/image_features_race_full.csv"
IMAGE_FEATURES_RACE = "/home/yuanyuan/maskrcnn-benchmark/demo/face-classification/race_features_augmentation/image_features_race_dir_001.csv"

LOAN = "/home/yuanyuan/kiva/data/kiva_berkeley/loans.csv"
MAP = "/home/yuanyuan/kiva/data/kiva_berkeley/imagehash_to_id_expired_default.csv"

if __name__ == "__main__":
    print("loading data: ")
    image_features = pd.read_csv(IMAGE_FEATURES)
    race_features = pd.read_csv(IMAGE_FEATURES_RACE)
    loan = pd.read_csv(LOAN)
    hash_id_map = pd.read_csv(MAP)[['HASH', 'image_id', 'GENDER']]

    print("image_featuers:")
    print(image_features)
    print("race_featuers:")
    print(race_features)
    print("loan:")
    print(loan)
    print("hash_id_map:")
    print(hash_id_map)

    # Join with hash_id map:
    joint_general = pd.merge(left=image_features, right=hash_id_map, left_on='HASH', right_on="HASH", how="inner")
    joint_race = pd.merge(left=race_features, right=hash_id_map, left_on='HASH', right_on="HASH", how="inner")
    joint_race["gender_code"] = np.where(joint_race["GENDER"]=="Female", 0, 1)

    print("joint_general:")
    print(joint_general)
    print("joint_race:")
    print(joint_race)


    # General Features:
    joint_general_and_loan = pd.merge(left=joint_general, right=loan, left_on="image_id", right_on="IMAGE_ID", how='inner') 
    joint_general_and_loan["gender_code"] = np.where(joint_general_and_loan["GENDER"]=="female", 0, 1)
    joint_general_and_loan = joint_general_and_loan.ix[joint_general_and_loan["STATUS"] != "fundRaising"]
    # joint_general_and_loan.to_csv("joint_general_and_loan_corr.csv")

    joint_general_and_loan_grouped_status = joint_general_and_loan.groupby(['STATUS']).mean() # general and loan group by status mean
    joint_general_and_loan_grouped_status.to_csv("joint_general_and_loan_grouped_status.csv")
    joint_general_and_loan_grouped_status = joint_general_and_loan.groupby(['STATUS']).sum() # general and loan group by status sum
    joint_general_and_loan_grouped_status.to_csv("joint_general_and_loan_grouped_status_sum.csv")

    print("joint_general_and_loan")
    print(joint_general_and_loan)

    # Race Features - Groupped by hash id
    joint_race = joint_race.groupby(['HASH']).mean()
    joint_race_and_loan = pd.merge(left=joint_race, right=loan, left_on="image_id", right_on="IMAGE_ID", how='inner')
    joint_race_and_loan = joint_race_and_loan.ix[joint_race_and_loan["COUNTRY_CODE"] != "TL"]
    joint_race_and_loan = joint_race_and_loan.ix[joint_race_and_loan["STATUS"] != "fundRaising"]
    joint_race_and_loan["CONTINENT_CODE"] = joint_race_and_loan["COUNTRY_CODE"].apply(pc.country_alpha2_to_continent_code)
    joint_race_and_loan_grouped_status = joint_race_and_loan.groupby(['STATUS']).mean() # race and loan group by status mean
    joint_race_and_loan_grouped_status.to_csv("joint_race_and_loan_grouped_status.csv")
    joint_race_and_loan['count'] = 1
    joint_race_and_loan_sum = joint_race_and_loan.groupby(['STATUS']).sum()

    # joint_race_and_loan["gender_code"] = np.where(joint_race_and_loan["GENDER"]=="Female", 0, 1)
    print("joint_race_and_loan")
    print(joint_race_and_loan)

    

    # Plot race vs status:
    fig, (ax, ax1, ax2) = plt.subplots(1,3,figsize=(25,8))
    ax = sns.scatterplot(x="Asian", y="White", hue="STATUS", data=joint_race_and_loan, ax=ax, alpha=0.8)
    ax1 = sns.scatterplot(x="Black", y="Asian", hue="STATUS", data=joint_race_and_loan, ax=ax1, alpha=0.8)  
    ax2 = sns.scatterplot(x="Black", y="White", hue="STATUS", data=joint_race_and_loan, ax=ax2, alpha=0.8)
    plt.savefig("temp_status.png")

    # Plot race vs continent:
    fig, (ax, ax1, ax2) = plt.subplots(1,3,figsize=(25,8))
    ax = sns.scatterplot(x="Asian", y="White", hue="CONTINENT_CODE", data=joint_race_and_loan, ax=ax, alpha=0.8)
    ax1 = sns.scatterplot(x="Black", y="Asian", hue="CONTINENT_CODE", data=joint_race_and_loan, ax=ax1, alpha=0.8)
    ax2 = sns.scatterplot(x="Black", y="White", hue="CONTINENT_CODE", data=joint_race_and_loan, ax=ax2, alpha=0.8)
    plt.savefig("temp_continent.png")


    # Compute Race Correlation Matrix:
    # df_race = joint_race_and_loan[["asian_score", "white_score", "black_score"]]
    df_race = joint_race.drop(columns=["Asian_count","White_count","Black_count","top","right","bottom","left", "image_id", 
        'Sunglasses', "Mustache", "Blurry", "Harsh Lighting", "Flash", "Soft Lighting", "Sideburns", "Mouth Closed", "Mouth Slightly Open", 
        "Mouth Wide Open", "Teeth Not Visible", "No Beard", "Goatee", "Double Chin", "Color Photo", "Posed Photo",  "Attractive Man", "Attractive Woman", "5 o Clock Shadow", "Strong Nose-Mouth Lines", "gender_code"])

    # "Attrative Man",

    print("df_race: ", df_race)
    # Compute the correlation matrix
    sns.set(font_scale=0.8)
    corr = df_race.corr()
    # corr = sns.clustermap(df_race, metric="correlation")
    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=np.bool))
    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(30, 30))
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    # Draw the heatmap with the mask and correct aspect ratio
    # sns.clustermap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
    #             square=True, linewidths=.5, cbar_kws={"shrink": .5})
    sns.clustermap(corr, center=0, cmap="vlag",
               linewidths=.75, figsize=(13, 13))


    plt.savefig("temp_corr_race.png")

'''
    joint_general_and_loan = image_features
    df_general = joint_general_and_loan[CATEGORIES]
    df_general = df_general.drop(columns=["__background"])
    sns.set(font_scale=3.3)
    # Compute the correlation matrix
    corr = df_general.corr()
    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=np.bool))
    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(30, 30))
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.savefig("temp_corr_general.png")

    df_general = joint_general_and_loan[CATEGORIES_tech_and_household]
    sns.set(font_scale=3.3)
    # Compute the correlation matrix
    corr = df_general.corr()
    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=np.bool))
    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(30, 30))
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.savefig("corr_tech_and_household.png")

    df_general = joint_general_and_loan[CATEGORIES_animal]
    sns.set(font_scale=3.3)
    # Compute the correlation matrix
    corr = df_general.corr()
    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=np.bool))
    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(30, 30))
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.savefig("corr_animal.png")

    df_general = joint_general_and_loan[CATEGORIES_transportation]
    sns.set(font_scale=3.3)
    # Compute the correlation matrix
    corr = df_general.corr()
    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=np.bool))
    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(30, 30))
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.savefig("corr_transportation.png")

    df_general = joint_general_and_loan[CATEGORIES_food_and_dining]
    sns.set(font_scale=3.3)
    # Compute the correlation matrix
    corr = df_general.corr()
    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=np.bool))
    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(30, 30))
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.savefig("corr_food_and_dining.png")

    df_general = joint_general_and_loan[CATEGORIES_travel_and_sport]
    sns.set(font_scale=3.3)
    # Compute the correlation matrix
    corr = df_general.corr()
    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=np.bool))
    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(30, 30))
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.savefig("corr_travel_and_sport.png")




    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(joint_race_and_loan['asian_score'], joint_race_and_loan['white_score'], joint_race_and_loan['black_score'], c=joint_race_and_loan["CONTINENT_CODE"], colormap='viridis')
    # ax.view_init(30, 185)

    # fig0 = ax.get_figure()
    # fig0.savefig("temp0.png")

    # fig1 = ax.get_figure()
    # fig1.savefig("temp1.png")
    
    #
    # fig2 = ax2.get_figure()
    # fig2.savefig("temp2.png")


'''










