# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import argparse
import cv2
from glob import glob
from maskrcnn_benchmark.config import cfg
from predictor import COCODemo
import os
import time
import csv

print("start")
parser = argparse.ArgumentParser(description="PyTorch Object Detection Webcam Demo")
parser.add_argument('--input_dir', type=str,
                    default='test/', required = True,
                    help='input image directory (default: test/)')

parser.add_argument('--output_dir', type=str,
                    default='test/', required = True,
                    help='output image directory (default: test/)')

parser.add_argument('--csv_file', type=str,
                    default='image_features_general_testing.csv',
                    help='output csv name')

# CSV_FILE = 'image_features_general_images_priority.csv'
# IMG_DIR = "/shared/yuanyuan/images_priority"
parser.add_argument(
        "--config-file",
        default="../configs/caffe2/e2e_mask_rcnn_R_50_FPN_1x_caffe2.yaml",
        metavar="FILE",
        help="path to config file",
    )
parser.add_argument(
    "--confidence-threshold",
    type=float,
    default=0.7,
    help="Minimum score for the prediction to be shown",
)
parser.add_argument(
    "--min-image-size",
    type=int,
    default=224,
    help="Smallest size of the image to feed to the model. "
        "Model was trained with 800, which gives best results",
)
parser.add_argument(
    "--show-mask-heatmaps",
    dest="show_mask_heatmaps",
    help="Show a heatmap probability for the top masks-per-dim masks",
    action="store_true",
)
parser.add_argument(
    "--masks-per-dim",
    type=int,
    default=2,
    help="Number of heatmaps per dimension to show",
)
parser.add_argument(
    "opts",
    help="Modify model config options using the command-line",
    default=None,
    nargs=argparse.REMAINDER,
)

# '/home/yuanyuan/maskrcnn-benchmark/demo/output_png/'
args = parser.parse_args()
csv_file = args.csv_file
img_dir = args.input_dir
output_dir = args.output_dir

print("csv_file: ", csv_file)
print("img_dir: ", img_dir)
print("output_dir: ", output_dir)

def main(csv_file, img_dir):
    # load config from file and command-line arguments
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    # prepare object that handles inference plus adds predictions on top of image.
    # coco_demo = COCODemo(
    #     cfg,
    #     csv_file,
    #     confidence_threshold=args.confidence_threshold,
    #     show_mask_heatmaps=args.show_mask_heatmaps,
    #     masks_per_dim=args.masks_per_dim,
    #     min_image_size=args.min_image_size,
    # )
    
    # Changed - original above
    coco_demo = COCODemo(
        cfg,
        confidence_threshold=args.confidence_threshold,
        show_mask_heatmaps=args.show_mask_heatmaps,
        masks_per_dim=args.masks_per_dim,
        min_image_size=args.min_image_size,
    )

    cam = cv2.VideoCapture(-1)
    # while True:
    start_time = time.time()
    ret_val, img = cam.read()
    #img_dir = "/home/yuanyuan/sshfs/stone2_pdf/pdf_subset/sample_png"
    # img_dir = "/home/yuanyuan/maskrcnn-benchmark/demo/output_png"
    # img_dir = "/home/yuanyuan/kiva/data/images"
    images = sorted(glob(os.path.join(img_dir, '*.jpg')))
    total_size = len(images)
    print("Total images size: ", total_size)
    count = 0
    coco_categories = sorted(coco_demo.CATEGORIES)
    coco_categories.append("HASH")
    print(coco_categories)
    
    with open(csv_file, 'w') as csv_f:
        # writer = csv.DictWriter(csv_file, fieldnames=coco_categories)
        writer = csv.DictWriter(csv_f, fieldnames=["ratio", "hash"])
        writer.writeheader()
    

    print("Done with header writing. ")
    for img in images:
        print("Current count: {} / {}".format(count, total_size))
        basename = os.path.splitext(os.path.basename(img))[0]
        feature_dict = dict.fromkeys(coco_categories, 0)
        count += 1
        image = cv2.imread(img)
        # composite = coco_demo.run_on_opencv_image(image, feature_dict, img)
        # Changed - original above
        composite = coco_demo.run_on_opencv_image(image, count, basename, output_dir, csv_file)
        # cv2.imwrite(output_dir+str(count)+'.png', composite)
        # print(composite)
        print("Time: {:.2f} s / img".format(time.time() - start_time))
    print("Done")

    # cv2.imwrite('/home/yuanyuan/maskrcnn-benchmark/demo/output_png/'+str(count)+'.png', composite)
        #cv2.imshow("COCO detections", composite)
    # if cv2.waitKey(1) == 27:
        # break  # esc to quit
    # cv2.destroyAllWindows()

if __name__ == "__main__":
    main(csv_file, img_dir)