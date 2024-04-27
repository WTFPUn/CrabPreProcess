from __future__ import annotations

import cv2
import numpy as np

import random
import itertools
from typing import List, Tuple, TypedDict, Literal
import tqdm

stop_loop = 10

class DataGenerator:
		def __init__(
				self, crab: np.ndarray, cover_nomesh: np.ndarray, cover_fullmesh: np.ndarray
		):
				self.crab = self.__crab_shift(cv2.resize(crab, (300, 400)))
				self.cover_nomesh = cv2.resize(cover_nomesh, (300, 400))
				self.cover_fullmesh = cv2.resize(cover_fullmesh, (300, 400))
				self.obj: List[Tuple[np.ndarray, np.ndarray]] = []

		def __get_min_obj_height(self):
				return min(self.obj, key=lambda x: x[0].shape[0])

		def __get_min_obj_width(self):
				return min(self.obj, key=lambda x: x[0].shape[1])

		def __get_max_obj_height(self):
				return max(self.obj, key=lambda x: x[0].shape[0])

		def __get_max_obj_width(self):
				return max(self.obj, key=lambda x: x[0].shape[1])

		def __crab_shift(self, crab: np.ndarray):
				return np.roll(crab, 25, axis=0)

		def __validate_scale(self, scale: List[float], img_size: Tuple[int, int]) -> bool:
				# check scale of object is not bigger than image size
				img_w = img_size[0]
				img_h = img_size[1]

				for s in scale:
						if s * self.__get_max_obj_height()[0].shape[0] > img_h:
								return False
						if s * self.__get_max_obj_width()[0].shape[1] > img_w:
								return False
				return True

		def generate(self, params: GenerateParams):

				p_flip_crab: List[np.ndarray] = []
				p_flip_crab.append(self.crab)
				if params["crab_flip"]:
						p_flip_crab.append(np.fliplr(self.crab))
						p_flip_crab.append(np.flipud(self.crab))
						p_flip_crab.append(np.flipud(np.fliplr(self.crab)))

				# put cover on crab
				obj_no_mesh: List[np.ndarray] = []
				for crab in p_flip_crab:
						temp = crab.copy()
						for i in range(self.cover_nomesh.shape[0]):
								for j in range(self.cover_nomesh.shape[1]):
										if not all(self.cover_nomesh[i, j] == 0):
												temp[i, j] = self.cover_nomesh[i, j]
						obj_no_mesh.append(temp)

				obj_full_mesh: List[np.ndarray] = []
				for crab in p_flip_crab:
						temp = crab.copy()
						for i in range(self.cover_fullmesh.shape[0]):
								for j in range(self.cover_fullmesh.shape[1]):
										if not all(self.cover_fullmesh[i, j] == 0):
												temp[i, j] = self.cover_fullmesh[i, j]
						obj_full_mesh.append(temp)

						# pair mesh and no mesh
				pairs: List[Tuple[np.ndarray, np.ndarray]] = []
				for i in range(len(obj_no_mesh)):
						pairs.append((obj_no_mesh[i], obj_full_mesh[i]))

				# rotate and scale
				rotated: List[Tuple[np.ndarray, np.ndarray]] = []
				if len(params["rotate_angle"]) > 0:
						for pair in pairs:
								for angle in params["rotate_angle"]:
										M = cv2.getRotationMatrix2D(
												(pair[0].shape[1] // 2, pair[0].shape[0] // 2), angle, 1
										)
										rotated.append(
												(
														cv2.warpAffine(
																pair[0], M, (pair[0].shape[1], pair[0].shape[0])
														),
														cv2.warpAffine(
																pair[1], M, (pair[1].shape[1], pair[1].shape[0])
														),
												)
										)

				# scale
				scaled: List[Tuple[np.ndarray, np.ndarray]] = []
				if len(params["scale"]) > 0:
						for pair in pairs:
								for scale in params["scale"]:
										scaled.append(
												(
														cv2.resize(
																pair[0],
																(
																		int(pair[0].shape[1] * scale),
																		int(pair[0].shape[0] * scale),
																),
														),
														cv2.resize(
																pair[1],
																(
																		int(pair[1].shape[1] * scale),
																		int(pair[1].shape[0] * scale),
																),
														),
												)
										)

				self.obj = rotated + scaled
				return self.obj

		def insert_in_image(
				self,
				img_size: Tuple[int, int],
				img_color: Tuple[int, int, int],
				object_per_image: int,
				n: int,
				obj_coords: bool = False,
		) -> List[Tuple[np.ndarray, np.ndarray]]:
				"""
				Inserts objects into images and generates versions with and without mesh overlays.

				Args:
								img_size (Tuple[int, int]): The desired image size (width, height).
								img_color (Tuple[int, int, int]): The background color in RGB format.
								object_per_image (int): The number of objects to insert into each image.
								n (int): The number of images to generate.

				Returns:
								List[Tuple[np.ndarray, np.ndarray]]: A list of tuples. Each tuple contains:
								* The generated image without a mesh overlay.
								* The generated image with a full mesh overlay.
				"""

				img_template = np.ones((img_size[0], img_size[1], 3), np.uint8) * img_color

				# mark points in the image that can place the object(not to close to the border)

				chunk_height = self.__get_max_obj_height()
				chunk_width = self.__get_max_obj_width()

				n_chunk_height = img_size[0] // chunk_height[0].shape[0]
				n_chunk_width = img_size[1] // chunk_width[0].shape[1]

				object_per_image = min(object_per_image, n_chunk_height * n_chunk_width)

				result = []
				coords = []
				


				for _ in tqdm.tqdm(range(n), desc=f"Generating images in {img_color} color... "):
						copied_img_no_mesh = img_template.copy()
						copied_img_full_mesh = img_template.copy()
						# sample obj respect by object_per_image and not overlap
						sample = random.sample(self.obj, object_per_image)
						# sample point by chunk 
						placed = 0
						for row in range(0, n_chunk_height):
								for col in range(0, n_chunk_width):
										# chance is the probability of placing an object in the current chunk 
										chance = (object_per_image - placed) / (n_chunk_height * n_chunk_width - row * n_chunk_width - col)
										# if the chance is 0, then we can't place any more objects
										if random.random() < chance:
												x, y = (row * chunk_height[0].shape[0], col * chunk_width[0].shape[1])
												obj = random.choice(sample)

												# check obj won't go out of the image. if it does, skip chunk
												if x + obj[0].shape[0] > img_size[0] or y + obj[0].shape[1] > img_size[1]:
														continue

												for i in range(obj[0].shape[0]):
													for j in range(obj[0].shape[1]):
															# insert pixel if it is not black
															if not all(obj[0][i, j] == 0):
																	copied_img_no_mesh[x + i, y + j] = obj[0][i, j]
															if not all(obj[1][i, j] == 0):
																	copied_img_full_mesh[x + i, y + j] = obj[1][i, j]



												placed += 1

												if placed == object_per_image:
														break
						result.append((copied_img_no_mesh, copied_img_full_mesh))
						if obj_coords:
								coords.append((x, y))

				return result


class GenerateParams(TypedDict):
		"""
		GenerateParams is a class that contains the parameters to generate a new image

		Attributes:
		------------
		rotate_angle: List[float]
		scale: List[float]
		crab_flip: bool
		object_flip_h: bool
		object_flip_v: bool
		"""

		rotate_angle: List[float]
		scale: List[float]
		crab_flip: bool
		object_flip_h: bool
		object_flip_v: bool

class CoordParams(TypedDict):
		format: Literal
