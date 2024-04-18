I have a many plan for this project. Split it into 3 part
1. Image Preprocessing
   1. Erosion, Dilation
   2. Image Synthesis
   3. Image Augmentation
2. Model Architecture Selection
3. Parameter Tuning

**Will be updated soon**
# Image processing

we need a useful data to detect the object. So, we need to preprocess the image. We will use the following techniques to preprocess the image.

## dataset generator
We split component of object into 3 part
1. box
2. cover(with mesh, no mesh)
3. crab

and use 3 component to generate the dataset.

### Box
WIP...

### Cover
For the cover of box for Image inpainting(Autoencoder) we need a cover with mesh to train the model and cover remove mesh for output of the model. And this is how to make cover without mesh

1. remove original image with canny(100-250)
2. erosion [1.]  with 5x5 kernel. I've try 2 method
   1. with square 5x5 kernel for 5 iteration
   2. with cross 5x5 kernel for 4 iteration(better)

### Crab
For crab I just removed background and shift the image to the center of the image.


### Object Mix
In this part we will use crab is a base image and use cover(2 type) lap on top of the crab. then add background to the image.

