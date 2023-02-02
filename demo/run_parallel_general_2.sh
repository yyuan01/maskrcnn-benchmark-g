CUDA_VISIBLE_DEVICES=6 python webcam.py --input_dir /shared/yuanyuan/parallel/images/dir_006 --csv_file 'image_features_general_dir_006.csv' &
CUDA_VISIBLE_DEVICES=6 python webcam.py --input_dir /shared/yuanyuan/parallel/images/dir_007 --csv_file 'image_features_general_dir_007.csv' &
CUDA_VISIBLE_DEVICES=6 python webcam.py --input_dir /shared/yuanyuan/parallel/images/dir_008 --csv_file 'image_features_general_dir_008.csv' &
CUDA_VISIBLE_DEVICES=6 python webcam.py --input_dir /shared/yuanyuan/parallel/images/dir_009 --csv_file 'image_features_general_dir_009.csv' &
CUDA_VISIBLE_DEVICES=6 python webcam.py --input_dir /shared/yuanyuan/parallel/images/dir_010 --csv_file 'image_features_general_dir_010.csv' &
CUDA_VISIBLE_DEVICES=6 python webcam.py --input_dir /shared/yuanyuan/parallel/images/dir_011 --csv_file 'image_features_general_dir_011.csv' 

# CUDA_VISIBLE_DEVICES=7 python pred.py --model face_model.pkl --input_dir /shared/yuanyuan/parallel/images/dir_011 --csv_file 'image_features_race_dir_011.csv'  





