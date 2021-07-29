from dataset_definitions import get_dataset
from model import generator, discriminator
from parallel_threading import parallel_map_as_tf_dataset
from losses import init_perception_model, get_pose_loss, init_pose_model
import tensorflow as tf
from utils import initialize_uninitialized, make_pretrained_weight_loader, ssim
from io import BytesIO
import matplotlib.pyplot as plt
import time
from parameters import params
import numpy as np
import tensorflow_gan as tfgan
import os

backend = tf.keras.backend

if __name__ == '__main__':
    print('Hyperparams:')
    for name, val in params.items():
        print('{}:\t{}'.format(name, val))

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    backend.set_session(sess)
    init_perception_model()

    # TESTING GRAPH
    print('build testing graph')

    test_count = params['test_count']

    test_dataset = get_dataset(params['dataset'], deterministic=False, with_to_masks=True)
    test_data = []

    for test_sample in test_dataset.next_test_sample():
        test_data.append(test_sample)
        if len(test_data) == test_count:
            break

    def test_gen():
        while True:
            for sample in test_data:
                yield sample


    test_dataset = parallel_map_as_tf_dataset(None, test_gen(), n_workers=5, deterministic=False)
    test_dataset = test_dataset.batch(1, drop_remainder=True)
    test_iterator = test_dataset.make_one_shot_iterator()
    (test_img_from, test_img_to, test_mask_from, test_mask_to, test_transform_params, test_3d_input_pose,
      test_3d_target_pose) = test_iterator.get_next()

    print('- GAN')
    with tf.variable_scope('GAN', reuse=False):
        pose_gan = tfgan.gan_model(
            generator,
            discriminator,
            real_data= test_img_to,
            generator_inputs=[test_img_from, test_mask_from, test_transform_params, test_3d_input_pose,
                              test_3d_target_pose],
            check_shapes=False
        )

    # 2D mask for target pose to compute foreground SSIM
    test_fg_mask = tf.reduce_max(test_mask_to, axis=3)
    test_fg_mask = test_fg_mask[:, :-1, :-1]
    test_fg_mask = tf.image.resize_images(test_fg_mask, (params['image_size'], params['image_size']),
                                           method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    test_fg_mask = tf.reduce_max(test_fg_mask, axis=3)

    with tf.variable_scope('GAN/Generator', reuse=True):
        test_model = pose_gan.generator_fn([test_img_from, test_mask_from, test_transform_params, test_3d_input_pose, test_3d_target_pose])

    test_pose_loss = get_pose_loss(test_img_to, test_model[0])

    init_pose_model(sess, 'pose3d_minimal/checkpoint/model.ckpt-160684')

    if not os.path.exists(params['tb_dir']):
        os.makedirs(params['tb_dir'])

    start = time.time()
    checkpoint = tf.train.latest_checkpoint(params['check_dir'])
    summary_writer = tf.summary.FileWriter(params['tb_dir'])
    if checkpoint is not None:
        start_step = int(checkpoint.split('-')[-1])
        init_fn = make_pretrained_weight_loader(checkpoint, 'GAN', 'GAN', ['Adam', 'Momentum'], [])
        init_fn(sess)
        global_step = tf.Variable(start_step, trainable=False, name='global_step')
        initialize_uninitialized(sess)
        print(f'Loaded checkpoint from step {start_step}:', time.time() - start)

        print('Performing test')
        test_start = time.time()
        v_inp = []
        v_tar = []
        v_gen = []
        v_pl = []
        v_bg = []
        v_bg_mask = []
        v_fg = []
        v_fg_m = []
        test_generated = tf.clip_by_value(test_model[0], -1, 1)
        print('- generating images')
        for _ in range(test_count):
            inp, tar, gen, pl, bg, bg_mask, fg, fg_m = sess.run(
                [test_img_from, test_img_to, test_generated, test_pose_loss, test_model[1]['background'],
                  test_model[1]['foreground_mask'], test_model[1]['foreground'], test_fg_mask])
            v_inp.append(inp[0, :256, :256] / 2 + .5)
            v_tar.append(tar[0, :256, :256] / 2 + .5)
            v_gen.append(gen[0, :256, :256] / 2 + .5)
            v_pl += [pl]
            v_bg.append(bg[0, :256, :256] / 2 + .5)
            v_bg_mask.append(np.tile(bg_mask[0, :256, :256], [1, 1, 3]))
            v_fg.append(fg[0, :256, :256] / 2 + .5)
            v_fg_m.append(fg_m[0, ..., np.newaxis])

        prefix = 'test'
        print('- computing SSIM scores')
        ssim_score, ssim_fg, ssim_bg = ssim(v_tar, v_gen, masks=v_fg_m)
        summary = tf.Summary(value=[tf.Summary.Value(tag=f'{prefix}_metrics/ssim', simple_value=ssim_score)])
        summary_writer.add_summary(summary, 0)
        summary = tf.Summary(value=[tf.Summary.Value(tag=f'{prefix}_metrics/ssim_fg', simple_value=ssim_fg)])
        summary_writer.add_summary(summary, 0)
        summary = tf.Summary(value=[tf.Summary.Value(tag=f'{prefix}_metrics/ssim_bg', simple_value=ssim_bg)])
        summary_writer.add_summary(summary, 0)

        print('- computing pose score')
        pl = np.mean(v_pl)
        summary = tf.Summary(value=[tf.Summary.Value(tag=f'{prefix}_metrics/pose_loss', simple_value=pl)])
        summary_writer.add_summary(summary, 0)
        
        if not os.path.exists('output'):
            os.makedirs('output')

        print('- creating images for tensorboard')
        v_inp = np.concatenate(v_inp[:test_count], axis=0)
        v_tar = np.concatenate(v_tar[:test_count], axis=0)
        v_gen = np.concatenate(v_gen[:test_count], axis=0)
        v_bg = np.concatenate(v_bg[:test_count], axis=0)
        v_bg_mask = np.concatenate(v_bg_mask[:test_count], axis=0)
        v_fg = np.concatenate(v_fg[:test_count], axis=0)
        res = np.concatenate([v_inp, v_tar, v_gen, v_bg, v_bg_mask, v_fg], axis=1)
        plt.imsave('output/res_with_mask.png', res, format='png')
        s = BytesIO()
        plt.imsave(s, res, format='png')
        res = tf.Summary.Image(encoded_image_string=s.getvalue(), height=res.shape[0], width=res.shape[1])
        summary = tf.Summary(value=[tf.Summary.Value(tag=f'{prefix}_img', image=res)])
        summary_writer.add_summary(summary, 0)
        summary_writer.flush()
        print('Performed Test:', time.time() - test_start)

        res2 = np.concatenate([v_inp, v_tar, v_gen], axis=1)
        plt.imsave('output/res.png', res2, format='png')
        
    else:
        print("No Model Found!!")

