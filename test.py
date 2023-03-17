import argparse

import cv2
import numpy as np
import tensorflow as tf
import neuralgym as ng

from inpaint_model import InpaintCAModel


parser = argparse.ArgumentParser()
# parser.add_argument('--image', default='', type=str,
#                     help='The filename of image to be completed.')
parser.add_argument('--mask', default='maskImg.png', type=str,
                    help='The filename of mask, value 255 indicates mask.')
parser.add_argument('--output', default='output.png', type=str,
                    help='Where to write output.')
parser.add_argument('--checkpoint_dir', default='', type=str,
                    help='The directory of tensorflow checkpoint.')
parser.add_argument('--original', default='originalImg.jpg', type=str,
                    help='The filename of original image, value 255 indicates mask.')
parser.add_argument('--doMirroring', default=False, type=bool,
                    help='True if using mirroring.')


if __name__ == "__main__":
    FLAGS = ng.Config('inpaint.yml')
    # ng.get_gpus(1)
    args, unknown = parser.parse_known_args()
    # print(args.doMirroring)

    if(args.doMirroring):

        model = InpaintCAModel()
        originalImg = cv2.imread(args.original)
        width = originalImg.shape[1]

        flippedimg = cv2.flip(originalImg, 1)
        finalImg = cv2.hconcat([originalImg, flippedimg])
        # crop:
        image = finalImg[:, width - int(width / 2):width + int(width/2)]
        # Add mask
        image[:, int(width/2):int(width/2)+int(width/4)] = 255

        cv2.imwrite('input_img.png', image)

        # image = cv2.imread(args.image)
        mask = cv2.imread(args.mask)
        # mask = cv2.resize(mask, (0,0), fx=0.5, fy=0.5)

        assert image.shape == mask.shape

        h, w, _ = image.shape
        grid = 8
        image = image[:h//grid*grid, :w//grid*grid, :]
        mask = mask[:h//grid*grid, :w//grid*grid, :]
        print('Shape of image: {}'.format(image.shape))

        image = np.expand_dims(image, 0)
        mask = np.expand_dims(mask, 0)
        input_image = np.concatenate([image, mask], axis=2)

        sess_config = tf.compat.v1.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        with tf.compat.v1.Session(config=sess_config) as sess:
            input_image = tf.constant(input_image, dtype=tf.float32)
            output = model.build_server_graph(FLAGS, input_image)
            output = (output + 1.) * 127.5
            output = tf.reverse(output, [-1])
            output = tf.saturate_cast(output, tf.uint8)
            # load pretrained model
            vars_list = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES)
            assign_ops = []
            for var in vars_list:
                vname = var.name
                from_name = vname
                var_value = tf.train.load_variable(args.checkpoint_dir, from_name)
                assign_ops.append(tf.compat.v1.assign(var, var_value))
            sess.run(assign_ops)
            print('Model loaded.')

            result = sess.run(output)
            model_results_img = result[0][:, :, ::-1]

            cropped_results = model_results_img[:, int(width/2):int(width/2)+int(width/4)]


            cv2.imwrite(args.output, cv2.hconcat([originalImg, cropped_results]))

    else:
        model = InpaintCAModel()



        originalImage = cv2.imread(args.original)

        mask = cv2.imread(args.mask)

        width = mask.shape[1]
        height = mask.shape[0]

        if originalImage.shape[1] < mask.shape[1]:
            tmpMask = np.empty(
                (height, width - originalImage.shape[1], 3), dtype="uint8")
            tmpMask.fill(255)
            image = cv2.hconcat([originalImage, tmpMask])
        else:
            # assume mask is already on the image for now.  We are not testing this case.
            image = originalImage

        cv2.imwrite('input_img.png', image)
        h, w, _ = image.shape
        grid = 8
        image = image[:h//grid*grid, :w//grid*grid, :]
        mask = mask[:h//grid*grid, :w//grid*grid, :]
        print('Shape of image: {}'.format(image.shape))

        image = np.expand_dims(image, 0)
        mask = np.expand_dims(mask, 0)
        input_image = np.concatenate([image, mask], axis=2)

        sess_config = tf.compat.v1.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        with tf.compat.v1.Session(config=sess_config) as sess:
            input_image = tf.constant(input_image, dtype=tf.float32)
            output = model.build_server_graph(FLAGS, input_image)
            output = (output + 1.) * 127.5
            output = tf.reverse(output, [-1])
            output = tf.saturate_cast(output, tf.uint8)
            # load pretrained model
            vars_list = tf.compat.v1.get_collection(
                tf.compat.v1.GraphKeys.GLOBAL_VARIABLES)
            assign_ops = []
            for var in vars_list:
                vname = var.name
                from_name = vname
                var_value = tf.train.load_variable(
                    args.checkpoint_dir, from_name)
                assign_ops.append(tf.compat.v1.assign(var, var_value))
            sess.run(assign_ops)
            print('Model loaded.')

            result = sess.run(output)
            model_results_img = result[0][:, :, ::-1]

            cropped_results = model_results_img[:, originalImage.shape[1]:width]

            cv2.imwrite(args.output, cv2.hconcat(
                [originalImage, cropped_results]))

