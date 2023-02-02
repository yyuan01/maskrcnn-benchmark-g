import os
import pickle
import PIL.Image
import numpy as np
import tensorflow as tf
import dnnlib
import dnnlib.tflib as tflib
import config
from encoder.generator_model import Generator
import matplotlib.pyplot as plt
import glob

PATH = "/home/yuanyuan/maskrcnn-benchmark/gan/stylegan-encoder"

black_path_list = glob.glob(PATH + "/latent_black/*.npy")
white_path_list = glob.glob(PATH + "/latent_white/*.npy")
asian_path_list = glob.glob(PATH + "/latent_asian/*.npy")

race_d = np.load("/home/yuanyuan/maskrcnn-benchmark/gan/stylegan-encoder/race_d.npy")
def collect_latents(path_list):
    latents = []
    for path in path_list:
        v = np.load(path)
        print(v, v.shape)
        latents.append(v)
    return latents

black_list = collect_latents(black_path_list)
white_list = collect_latents(white_path_list)
asian_list = collect_latents(asian_path_list)

print(len(black_list))
print(black_list[0].shape, type(black_list))
black_center = sum(black_list)/len(black_list)
white_center = sum(white_list)/len(white_list)
asian_center = sum(asian_list)/len(asian_list)

b2w = white_center - black_center
a2w = white_center - asian_center
b2a = asian_center - black_center

np.save("b2w.npy", b2w)
np.save("a2w.npy", a2w)
np.save("b2a.npy", b2a)

print("done")


    



