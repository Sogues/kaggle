#!/usr/bin/env python
# encoding: utf-8

import tensorflow as tf
import matplotlib.image as mpimg
import numpy as np
import math

def tf_resize_images(X_img_file_paths, image_size, channel=3):
    """
    param:
        X_img_file_paths: list of all files
        image_size: new image_size
        channel: origin image channel
    return:
        np.array
    """
    X_data = []
    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, (None, None, channel))
    tf_img = tf.image.resize_images(X, (image_size, image_size),
            tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for index, file_path in enumerate(X_img_file_paths):
            img = mpimg.imread(file_path)[:, :, :3]
            resized_img = sess.run(tf_img, feed_dict={X: img})
            X_data.append(resized_img)

    return np.array(X_data, dtype=np.float32)

def central_scale_images(X_imgs, scales, image_size, channel=3):
    """
    param:
        X_imgs: (N, H, W, C)
        scales: list of scale
    return:
        np.array
    """
    boxes = np.zeros((len(scales), 4), dtype=np.float32)
    for index, scale in enumerate(scales):
        x1 = y1 = 0.5 * (1-scale)
        x2 = y2 = 0.5 * (1+scale)
        boxes[index] = np.array([y1, x1, y2, x2], dtype=np.float32)

    box_ind = np.zeros((len(scales)), dtype=np.int32)
    crop_size = np.array([image_size, image_size], dtype=np.int32)

    X_scale_data = []
    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, shape=(1, image_size, image_size, channel))
    tf_img = tf.image.crop_and_resize(X, boxes, box_ind, crop_size)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for img_data in X_imgs:
            batch_img = np.expand_dims(img_data, axis=0)
            scaled_imgs = sess.run(tf_img, feed_dict={X:batch_img})
            X_scale_data.extend(scaled_imgs)

    return np.array(X_scale_data, dtype=np.float32)

def get_translate_parameters(index, image_size):
    """
    param:
        index:
            0 : left translate
            1 : right translate
            2 : top translate
            other : bottom translate
    """
    if index == 0:
        offset = np.array([0.0, 0.2], dtype=np.float32)
        size = np.array([image_size, math.ceil(0.8 * image_size)], dtype=np.int32)
        w_start = 0
        w_end = int(math.ceil(0.8*image_size))
        h_start = 0
        h_end = image_size
    elif index == 1:
        offset = np.array([0.0, -0.2], dtype=np.float32)
        size = np.array([image_size, math.ceil(0.8 * image_size)], dtype=np.int32)
        w_start = int(math.floor(0.2*image_size))
        w_end = image_size
        h_start = 0
        h_end = image_size
    elif index == 2:
        offset = np.array([0.2, 0.0], dtype=np.float32)
        size = np.array([math.ceil(0.8 * image_size), image_size], dtype=np.int32)
        w_start = 0
        w_end = image_size
        h_start = 0
        h_end = int(math.ceil(0.8 * image_size))
    else:
        offset = np.array([-0.2, 0.0], dtype=np.float32)
        size = np.array([math.ceil(0.8 * image_size), image_size], dtype=np.int32)
        w_start = 0
        w_end = image_size
        h_start =  int(np.floor(0.2 * image_size))
        h_end = image_size

    return offset, size, w_start, w_end, h_start, h_end

def translate_images(X_imgs):
    offsets = np.zeros((len(X_imgs), 2), dtype=np.float32)
    image_size = X_imgs.shape[1]
    n_translation = 4
    X_translated_arr = []

    tf.reset_default_graph()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(n_translation):
            X_translated = np.ones((len(X_imgs), image_size, image_size, 3),
                    dtype=np.float32)
            base_offset, size, w_start, w_end, h_start, h_end = get_translate_parameters(i, image_size)
            offsets[:, :] = base_offset
            glimpses = tf.image.extract_glimpse(X_imgs, size, offsets)
            glimpses = sess.run(glimpses)
            X_translated[:, h_start:h_end, w_start:w_end, :] = glimpses
            X_translated_arr.extend(X_translated)

    return np.array(X_translated_arr, dtype=np.float32)

def rotate_images(X_imgs, K):
    """
    param:
        X_imgs: (N, H, W, C)
        k: list [1, 2, 3]
            1: 90degree, 2: 180degree, 3:270degree
    """
    X_rotate = []
    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, shape=X_imgs.shape[1:])
    k = tf.placeholder(tf.int32)
    tf_img = tf.image.rot90(X, k=k)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for img in X_imgs:
            for i in K:
                rotated_img = sess.run(tf_img, feed_dict={X:img, k:i})
                X_rotate.append(rotated_img)

    return np.array(X_rotate, dtype=np.float32)

def rotate_images_by_angle(X_imgs, start_angle, end_angle, n_images):
    X_rotate = []
    iterate_at = (end_angle - start_angle) / (n_images - 1)

    tf.reset_default_graph()
    X = tf.placeholder(tf.float32,
            shape=(None, X_imgs.shape[1], X_imgs.shape[2], X_imgs.shape[3]))
    radian = tf.placeholder(tf.float32, shape=(len(X_imgs)))
    tf_img = tf.contrib.image.rotate(X, radian)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for idx in range(n_images):
            degrees_angle = start_angle + idx * iterate_at
            radian_value = degrees_angle * math.pi / 180
            radian_arr = [radian_value] * len(X_imgs)
            rotated_imgs = sess.run(tf_img, feed_dict={X:X_imgs, radian:radian_arr})
            X_rotate.extend(rotated_imgs)

    return np.array(X_rotate, dtype=np.float32)

def flip_images(X_imgs, k):
    X_flip = []
    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, shape=X_imgs.shape[1:])
    variables = []
    for item in k:
        if k == 1:
            variables.append(tf.image.flip_left_right(X))
        elif k == 2:
            tf_img2 = tf.image.flip_up_down(X)
        else:
            tf_img3 = tf.image.transpose_image(X)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for img in X_imgs



