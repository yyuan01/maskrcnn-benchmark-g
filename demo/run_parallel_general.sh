CUDA_VISIBLE_DEVICES=7 python webcam.py --input_dir /shared/yuanyuan/parallel/images/dir_001 --csv_file 'image_features_general_dir_001.csv' &
CUDA_VISIBLE_DEVICES=7 python webcam.py --input_dir /shared/yuanyuan/parallel/images/dir_002 --csv_file 'image_features_general_dir_002.csv' &
CUDA_VISIBLE_DEVICES=7 python webcam.py --input_dir /shared/yuanyuan/parallel/images/dir_003 --csv_file 'image_features_general_dir_003.csv' &
CUDA_VISIBLE_DEVICES=7 python webcam.py --input_dir /shared/yuanyuan/parallel/images/dir_004 --csv_file 'image_features_general_dir_004.csv' &
CUDA_VISIBLE_DEVICES=7 python webcam.py --input_dir /shared/yuanyuan/parallel/images/dir_005 --csv_file 'image_features_general_dir_005.csv' &

# CUDA_VISIBLE_DEVICES=7 python pred.py --model face_model.pkl --input_dir /shared/yuanyuan/parallel/images/dir_011 --csv_file 'image_features_race_dir_011.csv'  





