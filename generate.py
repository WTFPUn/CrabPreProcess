from datagenerator import DataGenerator, GenerateParams
import numpy as np
import cv2
import os
import random
import uuid
import tqdm
import sys
params = GenerateParams(rotate_angle=[30, 45 , 90], scale=[1.3, 1], crab_flip=True, object_flip_h=True, object_flip_v=True)
crab = cv2.imread('./crab/crab.png')
rem_mesh_cover = cv2.imread("./Box/cover_ero.png")
cover_fullmesh = cv2.imread("./Box/box_fullmesh.png")

gen = DataGenerator(crab, rem_mesh_cover, cover_fullmesh)

gen.generate(params)

root = "./dataset_same_bg"
if not os.path.exists(root):
  os.makedirs(root)
  os.makedirs(f"{root}/original")
  os.makedirs(f"{root}/target")


color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
for i in tqdm.tqdm(range(30)):

  img_size = (1200, 900)
  obj_per_image = random.randint(1, 20)
  n = 20

  out = gen.insert_in_image(img_size, color, obj_per_image, n)
  for target, original  in out:
    id = uuid.uuid4()
    cv2.imwrite(f"{root}/original/{str(id)}.png", original)
    cv2.imwrite(f"{root}/target/{str(id)}.png", target)


# # run python playground.py after running this script
# os.system("python playground.py")


