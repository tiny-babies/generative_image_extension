# CHANGELOG

The original source code is from https://github.com/JiahuiYu/generative_inpainting.  Under this changelog, I will review all the work we did that made changes to this model.

## test.py

- Modified the parser arguments to be able to consider for image extension.
- doMirroring flag is for testing our model on an image with or without mirroring the image as input to the model.  If the doMirror flag is true, we added code that will preprocess the image so that it is mirrored, and the model does inpainting on the extended region between the original and the mirrored image.

## eval_metrics.py

- We created this script to evaluate results for certain images using PSNR and SSIM.

## generate_flist.py

- We created this script to process the dataset.  DeepFill v2 requires flist files to work, so we created this script to generate those flist files from our dataset.

## inpaint_model.py

- Changes to the code starting on lines 149 and 230 comments out the irregular mask to be used on the images. DeepFill v2 creates random, irregular masks to put on the images when training the model.  Our model does not want irregular masks, so we comment this out, and use only the mask created by bbox2mask.

## inpaint_ops.py

- We changes the function for bbox2mask on line 126.  The original model generates a mask on a random area of the image.  We had to change this so that the mask that is generated will always be in the same position, which is right where the mirrored input starts (the extended region). 

## preprocessing: createImage.py and preprocess_data.py

- preprocess_data.py we created to preprocess our dataset as described in the paper, where it transforms the dataset to a dataset of mirrored inputs.  It appends the mirrored version of the image to each image, then centers the image window and crops out everything else so the image remains 256 x 256 pixels.  The mask is added during training. 
- createImage.py script was used for testing around with how to process the images, and it was used to create the mask that is used as input for the test.py script.