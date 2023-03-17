# Generative Image Extention

![pytorch](https://img.shields.io/badge/tensorflow-v2-green.svg?style=plastic)
![license](https://img.shields.io/badge/license-CC_BY--NC-green.svg?style=plastic)

From the open source framework for generative image inpainting task, with the support of [Contextual Attention](https://arxiv.org/abs/1801.07892) (CVPR 2018) and [Gated Convolution](https://arxiv.org/abs/1806.03589) (ICCV 2019 Oral).

## Using DeepFill v2 code and pre-trained weights https://github.com/JiahuiYu/generative_inpainting.


[CVPR 2018 Paper](https://arxiv.org/abs/1801.07892) | [ICCV 2019 Oral Paper](https://arxiv.org/abs/1806.03589) | [Project](http://jiahuiyu.com/deepfill) | [Demo](http://jiahuiyu.com/deepfill) | [YouTube v1](https://youtu.be/xz1ZvcdhgQ0) | [YouTube v2](https://youtu.be/uZkEi9Y2dj4) | [BibTex](#citing)

<img src="/Examples/ex1/originalImg.jpg" width="30%"/> <img src="/Examples/ex1/inputimg.png" width="30%"/> <img src="/Examples/ex1/output_Mirroring_FineTuned.png" width="36%"/>
<img src="/Examples/ex7/originalImg.jpg" width="30%"/> <img src="/Examples/ex7/input_img_Mirroring.png" width="30%"/> <img src="/Examples/ex7/output_Mirroring_FineTuned.png" width="36%"/>


### Using updated DeepFill v2 code that is now using TensorFlow 2.  Also updated Neuralgym to TensorFlow 2: github.com/tiny-babies/neuralgym.

## Run

0. Requirements:
    * Install python3.
    * Install [tensorflow](https://www.tensorflow.org/install/) (tested on Release 2.4 - 2.7).
    * Install tensorflow toolkit [neuralgym](https://github.com/tiny-babies/neuralgym) (run `pip install git+https://github.com/tiny-babies/neuralgym`).
1. Training:
    * Prepare training images filelist and shuffle it ([example](https://github.com/JiahuiYu/generative_inpainting/issues/15)).
    * Modify [inpaint.yml](/inpaint.yml) to set DATA_FLIST, LOG_DIR, IMG_SHAPES and other parameters.
    * Run `python train.py`.
2. Resume training:
    * Modify MODEL_RESTORE flag in [inpaint.yml](/inpaint.yml). E.g., MODEL_RESTORE: 20180115220926508503_places2_model.
    * Run `python train.py`.
3. Testing:
    * Run `python test.py --image examples/input.png --mask examples/mask.png --output examples/output.png --checkpoint model_logs/your_model_dir --doMirroring True/False`.


## Datasets and pretrained weights for Image extention: 
https://drive.google.com/drive/u/1/folders/1utRC6zJA2BHl85Di4pO-fmu67KOe0O8c

## Pretrained model for DeepFill v2:

[Places2](https://drive.google.com/drive/folders/1y7Irxm3HSHGvp546hZdAZwuNmhLUVcjO?usp=sharing) | [CelebA-HQ](https://drive.google.com/drive/folders/1uvcDgMer-4hgWlm6_G9xjvEQGP8neW15?usp=sharing)


## TensorBoard

Visualization on [TensorBoard](https://www.tensorflow.org/programmers_guide/summaries_and_tensorboard) for training and validation is supported. Run `tensorboard --logdir model_logs --port 6006` to view training progress.



## Citing
```
@article{yu2018generative,
  title={Generative Image Inpainting with Contextual Attention},
  author={Yu, Jiahui and Lin, Zhe and Yang, Jimei and Shen, Xiaohui and Lu, Xin and Huang, Thomas S},
  journal={arXiv preprint arXiv:1801.07892},
  year={2018}
}

@article{yu2018free,
  title={Free-Form Image Inpainting with Gated Convolution},
  author={Yu, Jiahui and Lin, Zhe and Yang, Jimei and Shen, Xiaohui and Lu, Xin and Huang, Thomas S},
  journal={arXiv preprint arXiv:1806.03589},
  year={2018}
}
```
