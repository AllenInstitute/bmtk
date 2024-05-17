import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
import json
from PIL import Image
import argparse
import logging

# from create_configs import create_lgnmodel_config
from allensdk.core.brain_observatory_cache import BrainObservatoryCache

# logger = logging.Logger(__name__)

# def get_natural_scenes(output_dir='bob_images'):
#     """Fetches the 118 Brain Obs natural scene images from the data, saves them in npy format"""
#     if not os.path.exists(output_dir):
#         os.mkdir(output_dir)

#     boc = BrainObservatoryCache(manifest_file='boc/manifest.json')
#     data_set = boc.get_ophys_experiment_data(501498760)
#     scenes = data_set.get_stimulus_template('natural_scenes')

#     for i in range(0, len(scenes)):
#         scene = scenes[i]
#         base_name = os.path.join(output_dir, 'scene.{:03d}.gray_{}x{}'.format(i, scene.shape[0], scene.shape[1]))
#         plt.imsave('{}.png'.format(base_name), scene, cmap='gray')
#         np.save('{}.npy'.format(base_name), scene)


# def create_gray_screen(output_dir='bob_images', res_y=918, res_x=1174):
#     """Creates grey-screen for BOb natural scene images"""
#     if not os.path.exists(output_dir):
#         os.mkdir(output_dir)
#     base_name = os.path.join(output_dir, 'gray_screen.{}x{}'.format(res_y, res_x))

#     gray_screen = np.full((res_y, res_x), fill_value=201.0)
#     plt.imsave(base_name + '.png', gray_screen, cmap='gray', vmin=0, vmax=256)
#     np.save(base_name + '.npy', gray_screen)


# """
# def create_movies(images_per_movie=10, res_y=918, res_x=1174, movie_set=1):
#     movies_dir = 'movies_set{:02d}'.format(movie_set)
#     if not os.path.exists(movies_dir):
#         os.mkdir(movies_dir)

#     gray_screen = np.array(Image.open('gray_screen/gray_screen.{}x{}.png'.format(res_y, res_x)).convert('L'))
#     gray_block = np.array([gray_screen for _ in range(45)])

#     movie_files = []
#     image_files = []
#     image_nums = []
#     images_perm = np.random.permutation(range(118))
#     for set_n, start_idx in enumerate(range(0, len(images_perm), images_per_movie)):
#         end_idx = np.min((start_idx + images_per_movie, len(images_perm)))
#         n_images = end_idx - start_idx

#         file_name = 'movie_{}.{}_images.{}x{}.60fps.npy'.format(set_n, n_images, res_y, res_x)
#         movie_arr = np.zeros((60 * n_images, res_y, res_x), dtype=np.int)
#         for movie_offset, img_idx in enumerate(range(start_idx, end_idx)):
#             # imag_off
#             image_file = 'images/scene.{:03d}.gray_{}x{}.png'.format(img_idx, res_y, res_x)
#             im = np.array(Image.open(image_file).convert('L'))
#             img_block = np.array([im for _ in range(15)])

#             offset = movie_offset * 60
#             movie_arr[offset:offset + 45, :, :] = gray_block
#             movie_arr[(offset + 45):(offset + 60), :, :] = img_block

#             movie_files.append(file_name)
#             print(img_idx)
#             image_nums.append(img_idx)
#             image_files.append(image_file)

#         np.save(os.path.join(movies_dir, file_name), movie_arr)

#     pd.DataFrame({
#         'movie_file': movie_files,
#         'image_number': image_nums,
#         'image_file': image_files
#     }).to_csv(os.path.join(movies_dir, 'movies_metadata.csv'), sep=' ', index=False)
# """

# """
# def create_movie_allns(res_y=918, res_x=1174):
#     movies_dir = 'movie_all_ns'
#     if not os.path.exists(movies_dir):
#         os.mkdir(movies_dir)

#     gray_screen = np.array(Image.open('gray_screen/gray_screen.{}x{}.png'.format(res_y, res_x)).convert('L'))
#     gray_block = np.array([gray_screen for _ in range(30)])

#     full_movie = np.zeros((30 + 118 * 15, res_y, res_x), dtype=np.int)
#     full_movie[0:30, :, :] = gray_block

#     image_files = []
#     image_nums = []
#     np.random.seed(100)
#     images_perm = np.random.permutation(range(118))
#     for image_num, image_id in enumerate(images_perm):
#         image_file = 'images/scene.{:03d}.gray_{}x{}.png'.format(image_id, res_y, res_x)
#         im = np.array(Image.open(image_file).convert('L'))
#         im_block = np.array([im for _ in range(15)])

#         offset = 30 + image_num * 15
#         full_movie[offset:(offset + 15), :, :] = im_block

#         image_files.append(image_file)
#         image_nums.append(image_num)

#     pd.DataFrame({
#         'movie_file': 'all_natural_scenes.trial00.60fp.npy',
#         'image_number': image_nums,
#         'image_file': image_files
#     }).to_csv(os.path.join(movies_dir, 'movies_metadata.csv'), sep=' ', index=False)

#     np.save(os.path.join(movies_dir, 'all_natural_scenes.trial00.60fp.npy'), full_movie)
# """


# def check_movie(movie_file):
#     # movie_file = 'movie_all_ns/all_natural_scenes.trial00.60fp.npy'
#     movie_array = np.load(movie_file)
#     plt.figure()
#     plt.imshow(movie_array[30, :, :], cmap='gray')

#     plt.figure()
#     plt.imshow(movie_array[29, :, :], cmap='gray')

#     plt.figure()
#     plt.imshow(movie_array[59, :, :], cmap='gray')

#     plt.figure()
#     plt.imshow(movie_array[60, :, :], cmap='gray')

#     plt.figure()
#     plt.imshow(movie_array[-1, :, :], cmap='gray')

#     plt.show()


# def create_ns_movie(n_images=118, ips=10, fps=1000.0, res_row=120, res_col=240, gs_pre=500, gs_post=100, rng_seed=None,
#                     images_dir='bob_images', output_dir='movies', overwrite=True, create_config=True, normalize=True,
#                     config_dir='lgnmodel_configs'):
#     frames_per_image = int(float(fps) / ips)  # num of frames that will be given to each image
#     frames_per_gs_pre = int(gs_pre / 1000.0 * fps)  # num frames for initial gray screen
#     frames_per_gs_post = int(gs_post / 1000.0 * fps)  # num frames for ending gray screen
#     n_frames = frames_per_gs_pre + frames_per_image*n_images + frames_per_gs_post
#     set_num = '{:02d}'.format(0)
#     total_time = n_frames/float(fps)*1000.0

#     # Find the movie path, make sure not to overwrite existing movie files
#     if not os.path.exists(output_dir):
#         os.mkdir(output_dir)

#     movie_path = os.path.join(output_dir, 'ns_{}images.set{}.{}ms.{}fps.{}ips.normalized.npy'.format(
#         n_images, set_num, int(total_time), int(fps), int(ips)
#     ))

#     if not overwrite and os.path.exists(movie_path):
#         for i in range(1, 100):
#             set_num = '{:02d}'.format(i)
#             movie_path = 'movies/ns_{}images.set{}.{}ms.{}fps.{}ips{}.npy'.format(
#                 n_images, set_num, int(total_time), int(fps), int(ips), '.normalized' if normalize else ''
#             )
#             if not os.path.exists(movie_path):
#                 break
#         else:
#             raise ValueError('Could not create valid movie path.')

#     if create_config:
#         if not os.path.exists(config_dir):
#             os.makedirs(config_dir)

#         base_name = 'ns_{}images_set{}_{}fps_{}ips{}'.format(n_images, set_num, int(fps), int(ips),
#                                                              '_normalized' if normalize else '')

#         config_file = 'ns_{}images.{}fps.{}ips.set{}{}.json'.format(n_images, int(fps), int(ips), set_num,
#                                                                     '.normalized' if normalize else '')
#         config_path = os.path.join(config_dir, config_file)
#         json.dump(
#             {
#                 'movie_file': movie_path,
#                 'fps': fps,
#                 'duration': total_time / 1000.0,
#                 'gray_screen': gs_pre / 1000.0,
#                 'output_dir': os.path.join('lgnmodel_results', 'output_{}'.format(base_name)),
#                 'force_overwrite': True,
#                 'normalized': normalize,
#                 'res_row': res_row,
#                 'res_col': res_col,
#                 'base_name': 'ns_{}images.{}ms.{}fps.{}ips.set{}{}'.format(
#                     n_images, int(total_time), int(fps), int(ips), set_num, '.normalized' if normalize else ''
#                 ),
#                 'n_trials': 5
#             },
#             open(config_path, 'w'),
#             indent=2
#         )
#         # base_name = 'ns_{}images_set{}_{}fps_{}ips{}'.format(n_images, set_num, int(fps), int(ips),
#         #                                                      '_normalized' if normalize else '')
#         # create_lgnmodel_config(
#         #     config_path=os.path.join('lgnmodel_configs', 'config.filternet.{}.json'.format(base_name)),
#         #     output_dir=os.path.join('lgnmodel_results', 'output_{}'.format(base_name)),
#         #     duration=float(total_time),
#         #     movie_path=os.path.basename(movie_path),
#         #     fps=fps,
#         #     res_row=res_row,
#         #     res_col=res_col
#         # )
#         # create_pointnet_config(
#         #     config_path=os.path.join('pointnet_configs', 'config.pointnet.{}.json'.format(base_name)),
#         #     output_dir=os.path.join('pointnet_results', 'output_{}'.format(base_name)),
#         #     duration=float(total_time),
#         #     lgn_spikes_path=os.path.join('lgnmodel_results', 'output_{}'.format(base_name), 'spikes.h5')
#         # )


#     # Keep track of the order and placement of the images, metadata will be written to csv
#     csv_path = movie_path[:-4] + '.metadata.csv'
#     img_names = []
#     frames_ids = []
#     image_times = []

#     if frames_per_gs_pre > 0:
#         img_names.append('grayscreen')
#         frames_ids.append((0, frames_per_gs_pre))
#         image_times.append((0, frames_per_gs_pre/float(fps)))

#     output_mat = np.zeros((n_frames, res_row, res_col), dtype=float)
#     mat_beg = frames_per_gs_pre
#     if rng_seed:
#         np.random.seed(rng_seed)
#     images_perm = np.random.permutation(range(n_images))
#     for image_num, image_id in enumerate(images_perm):
#         # Find the image and convert to a 120x240 matrix
#         # img_path = movie_jpegs[i]
#         img_path = os.path.join(images_dir, 'scene.{:03d}.gray_918x1174.png'.format(image_id))
#         pic = Image.open(img_path).convert('L')
#         pic = pic.resize((res_col, res_row))  # PIL the row/col order is swapped
#         pic_data = np.asarray(pic)
#         if normalize:
#             pic_data = pic_data.astype(dtype=float) * 2.0 / 255.0 - 1.0

#         mat_end = mat_beg + frames_per_image
#         output_mat[mat_beg:mat_end, :, :] = pic_data

#         # keep track of metadata
#         img_names.append(img_path[len(images_dir)+1:])
#         frames_ids.append((mat_beg, mat_end))
#         image_times.append((mat_beg/float(fps), mat_end/float(fps)))

#         mat_beg = mat_end

#     np.save(movie_path, output_mat)

#     if frames_per_gs_post > 0:
#         img_names.append('grayscreen')
#         f_beg = n_frames - frames_per_gs_post
#         f_end = n_frames
#         frames_ids.append((f_beg, f_end))
#         image_times.append((f_beg/float(fps), f_end/float(fps)))

#     pd.DataFrame({
#         'image_id': img_names,
#         'begin_frame': [f[0] for f in frames_ids],
#         'end_frame': [f[1] for f in frames_ids],
#         'begin_time': [t[0] for t in image_times],
#         'end_time': [t[1] for t in image_times]
#     }).to_csv(csv_path, sep=' ', index=False)


# '''
# def create_ns_movie(n_images=118, fps=1000, res_row=120, res_col=240, images_dur=.25, gs_pre=.5, gs_post=.1, 
#                     rng_seed=None, output_dir='movies', images_dir='bob_images', create_config=True, 
#                     config_dir='lgnmodel_configs', n_trials=1, normalize=False):
#     """Creates a movie based on the natural-scene images taken from Brain Observatory sessions, randomized. Also
#     creates a stimulus table

#     :param n_images: number of images taken from bob to show
#     :param fps: frame-per-second
#     :param res_y: y resolution
#     :param res_x: x resultion
#     :param images_dur: time (seconds) each image is shown.
#     :param gray_screen_dur: time (seconds) of initial grey-screen
#     :param rng_seed: seed for randomization
#     :param output_dir: directory where movie (and stimulus table) will
#     :param images_dir: directory of the BOb images
#     :param create_config: creates a config for running lgnmodel
#     :param config_dir: directory where lgnmodel config will be saved too.
#     :param n_trials: used for lgnmodel config.
#     :param normalize: normalize movie from [0, 255] --> [-1, 1]
#     :return:
#     """

#     ips = int(1.0/images_dur)
#     total_time = gs_pre + images_dur*n_images + gs_post

#     base_name = 'ns_{}images_{}fps_{}ips'.format(n_images, int(fps), int(ips))
#     trial_num = 0
#     for trial_num in range(100):
#         tmp_dir = os.path.join(output_dir, '{}_set{:02d}'.format(base_name, trial_num))
#         if not os.path.exists(tmp_dir):
#             output_dir = tmp_dir
#             break

#     movie_file = 'ns_{}images.{}fps.{}ips.{}x{}.set{:02d}{}.npy'.format(n_images, fps, int(ips), res_y, res_x,
#                                                                         trial_num, '.normalized' if normalize else '')

#     if rng_seed is not None:
#         np.random.seed(rng_seed)

#     n_frames_pre = int(gs_pre * fps)
#     n_frames_images = int(images_dur * fps)
#     n_frames_post = int(gs_post * fps)
#     full_movie = np.zeros((n_frames_pre + n_images*n_frames_images + n_frames_post, res_row, res_col), 
#                           dtype=np.int if normalize else np.float)

#     stimulus_table = {
#         'image_file': [],
#         'image_id': [],
#         'start_time': [],
#         'stop_time': [],
#         'n_frames': [],
#         'movie_file': movie_file,
#         'block_number': np.arange(n_images + int(n_frames_pre != 0) + int(n_frames_post != 0))
#     }

#     curr_frame = 0
#     if n_frames_pre > 0:
#         if normalize:
#             gray_screen = np.full((res_y, res_x), 0.0, dtype=np.float)
#         else:
#             gray_screen = np.full((res_y, res_x), 127, dtype=np.integer)

#         gray_block = np.array([gray_screen for _ in range(n_frames_gray)])
#         full_movie[curr_frame:(curr_frame + n_frames_gray), :, :] = gray_block

#         stimulus_table['image_file'].append('gray_screen')
#         stimulus_table['image_id'].append(-1)
#         stimulus_table['n_frames'].append(n_frames_gray)
#         stimulus_table['start_time'].append(float(curr_frame) / fps)
#         stimulus_table['stop_time'].append(float(curr_frame + n_frames_gray) / fps)
#         curr_frame += n_frames_gray

#     images_perm = np.random.permutation(range(n_images))
#     for image_num, image_id in enumerate(images_perm):
#         # image_file = os.path.join(images_dir, 'scene.{:03d}.gray_{}x{}.png'.format(image_id, res_y, res_x))
#         image_file = 'scene.{:03d}.gray_918x1174.png'.format(image_id)
#         img_path = os.path.join(images_dir, image_file)
#         pic = Image.open(img_path).convert('L')
#         pic = pic.resize((res_col, res_row))  # PIL the row/col order is swapped
#         pic_data = np.asarray(pic)
#         ## im = np.array(Image.open(image_file).convert('L'))
#         if normalize:
#             pic_data = pic_data.astype(dtype=np.float) * 2.0 / 255.0 - 1.0

#         # im_block = np.array([im for _ in range(n_frames_images)])
#         full_movie[curr_frame:(curr_frame + n_frames_images), :, :] = pic_data

#         stimulus_table['image_file'].append(image_file)
#         stimulus_table['image_id'].append(image_id)
#         stimulus_table['n_frames'].append(n_frames_images)
#         stimulus_table['start_time'].append(float(curr_frame) / fps)
#         stimulus_table['stop_time'].append(float(curr_frame + n_frames_images) / fps)

#         curr_frame += n_frames_images

#     os.makedirs(output_dir)
#     np.save(os.path.join(output_dir, movie_file), full_movie)
#     pd.DataFrame(data=stimulus_table).to_csv(os.path.join(output_dir, 'stimulus_table.csv'), sep=' ', index=False)

#     if create_config:
#         if not os.path.exists(config_dir):
#             os.makedirs(config_dir)

#         config_file = 'ns_{}images.{}fps.{}ips.set{:02d}{}.json'.format(n_images, int(fps), int(ips), trial_num, 
#                                                                         '.normalized' if normalize else '')
#         config_path = os.path.join(config_dir, config_file)
#         json.dump(
#             {
#                 'movie_file': os.path.join(output_dir, movie_file),
#                 'fps': fps,
#                 'duration': total_time,
#                 'gray_screen': gray_screen_dur,
#                 'output_dir': os.path.join('lgnmodel_results', 'output_{}'.format(base_name)),
#                 'force_overwrite': True,
#                 'normalized': normalize,
#                 'res_row': res_y,
#                 'res_col': res_x,
#                 'base_name': 'ns_{}images.{}ms.{}fps.{}ips.set{:02d}{}'.format(
#                     n_images, int(total_time), int(fps), int(ips), trial_num, '.normalized' if normalize else ''
#                 ),
#                 'n_trials': n_trials
#             },
#             open(config_path, 'w'),
#             indent=2
#         )
# '''

# """
# def get_natural_movies(movie_name='natural_movie_one', output_dir='movies'):
#     if not os.path.exists(output_dir):
#         os.mkdir(output_dir)
#     # base_name = os.path.join(output_dir, movie_name)

#     # print('HERE')
#     # exit()

#     boc = BrainObservatoryCache(manifest_file='boc/manifest.json')
#     data_set = boc.get_ophys_experiment_data(506248008)
#     for movie_name in ['natural_movie_one', 'natural_movie_two']:
#         # print(movie_name)
#         movie = data_set.get_stimulus_template(movie_name)
#         t, y, x = movie.shape
#         time = int(t / 30)
#         print(t, y, x)
#         exit()

#         fname = os.path.join(output_dir, '{}.{}s.30fps.{}x{}.npy'.format(movie_name, time, y, x))
#         # np.save(fname, movie)

#     boc = BrainObservatoryCache(manifest_file='boc/manifest.json')
#     data_set = boc.get_ophys_experiment_data(649409874)
#     for movie_name in ['natural_movie_three']:
#         # print(movie_name)
#         movie = data_set.get_stimulus_template(movie_name)
#         t, y, x = movie.shape
#         print(t, y, x)
#         exit()

#         time = int(t / 30)
#         fname = os.path.join(output_dir, '{}.{}s.30fps.{}x{}.npy'.format(movie_name, time, y, x))
#         np.save(fname, movie)
#         # print(movie.shape)


# def natural_movie_1000Hz_gs(output_dir='.', fps=1000):
#     boc = BrainObservatoryCache(manifest_file='boc/manifest.json')
#     data_set = boc.get_ophys_experiment_data(649409874)
#     # data_set = boc.get_ophys_experiment_data(506248008)

#     nframes_gs = int(fps * .5)
#     nframes_screen = int(fps * 3.0)

#     # movie_3sec_gs = np.zeros((int(1000*3.5), 304, 608), dtype=np.uint8)
#     movie_3sec_gs = np.full((int(1000 * 3.5), 304, 608), fill_value=127.5, dtype=np.float)

#     for movie_name in ['natural_movie_three']:
#         movie = data_set.get_stimulus_template(movie_name)
#         t, y, x = movie.shape

#         movie_3sec = movie[:90, :, :]
#         movie_3sec_1000hz = movie_3sec.repeat(34, axis=0)[:3000]
#         # print(movie_3sec_1000hz.shape)
#         # exit()
#         # print(movie.repeat(33, axis=0))
#         # exit()

#         movie_3sec_gs[500:, :, :] = movie_3sec_1000hz  # movie[:90, :, :]
#         print(movie_3sec_gs)
#         print(movie_3sec_gs.shape)
#         # exit()

#         fname = os.path.join(output_dir, '{}.3500ms.gs.30fps.{}x{}.npy'.format(movie_name, y, x))
#         np.save(fname, movie_3sec_gs)
# """


# def get_touchofevil_movies(output_dir='movies', create_config=True, config_dir='lgnmodel_configs'):
#     """Fetches the BOb touch-of-evil (natural movies), and converts them to npy format for using with lgnmodel

#     :param output_dir:
#     :param create_config:
#     :param config_dir:
#     """
#     boc = BrainObservatoryCache(manifest_file='boc/manifest.json')
#     data_set = boc.get_ophys_experiment_data(506248008)

#     movie = data_set.get_stimulus_template('natural_movie_one')
#     t, x, y = movie.shape
#     movie_name = 'natural_movie_one.30s.30fps.304x608.orig.npy'
#     movie_path = os.path.join(output_dir, movie_name)
#     np.save(movie_path, movie)
#     if create_config:
#         if not os.path.exists(config_dir):
#             os.makedirs(config_dir)

#         json.dump(
#             {
#                 'movie_file': movie_path,
#                 'fps': 30,
#                 'duration': t / 30.0,
#                 'gray_screen': 0.0,
#                 'output_dir': os.path.join('results_lgnmodel', 'output_natural_movie_one_30fps_orig'),
#                 'force_overwrite': True,
#                 'normalized': False,
#                 'res_row': x,
#                 'res_col': y,
#                 'base_name': 'natural_movie_one_30fps_orig',
#                 'n_trials': 1
#             },
#             open(os.path.join(config_dir, 'natural_movie_one.30s.30fps.304x608.orig.json'), 'w'),
#             indent=2
#         )

#     movie = data_set.get_stimulus_template('natural_movie_two')
#     t, x, y = movie.shape
#     movie_name = 'natural_movie_two.30s.30fps.304x608.orig.npy'
#     movie_path = os.path.join(output_dir, movie_name)
#     np.save(movie_path, movie)
#     if create_config:
#         json.dump(
#             {
#                 'movie_file': movie_path,
#                 'fps': 30,
#                 'duration': t / 30.0,
#                 'gray_screen': 0.0,
#                 'output_dir': os.path.join('results_lgnmodel', 'output_natural_movie_two_30fps_orig'),
#                 'force_overwrite': True,
#                 'normalized': False,
#                 'res_row': x,
#                 'res_col': y,
#                 'base_name': 'natural_movie_two_30fps_orig',
#                 'n_trials': 1
#             },
#             open(os.path.join(config_dir, 'natural_movie_two.30s.30fps.304x608.orig.json'), 'w'),
#             indent=2
#         )

#     data_set = boc.get_ophys_experiment_data(649409874)
#     movie = data_set.get_stimulus_template('natural_movie_three')
#     t, x, y = movie.shape
#     movie_name = 'natural_movie_three.30s.30fps.304x608.orig.npy'
#     movie_path = os.path.join(output_dir, movie_name)
#     np.save(movie_path, movie)
#     if create_config:
#         json.dump(
#             {
#                 'movie_file': movie_path,
#                 'fps': 30,
#                 'duration': t / 30.0,
#                 'gray_screen': 0.0,
#                 'output_dir': os.path.join('results_lgnmodel', 'output_natural_movie_three_30fps_orig'),
#                 'force_overwrite': True,
#                 'normalized': False,
#                 'res_row': x,
#                 'res_col': y,
#                 'base_name': 'natural_movie_three_30fps_orig',
#                 'n_trials': 1
#             },
#             open(os.path.join(config_dir, 'natural_movie_three.120s.30fps.304x608.orig.json'), 'w'),
#             indent=2
#         )


# def convert_movies(output_dir='movies', movie_name='natural_movie_one'):
#     movie_path = '{}.{}s.30fps.304x608.orig.npy'.format(movie_name, '120' if movie_name.endswith('three') else '30')
#     m_data = np.load(os.path.join(output_dir, movie_path))  # .astype(np.float)

#     movie_updated = np.full((int(30 * 3.5), 304, 608), fill_value=127, dtype=np.uint8)
#     movie_updated[15:, :, :] = m_data[:30 * 3, :, :]
#     movie_updated_path = '{}.3s_gs.30fps.304x608.orig.npy'.format(movie_name)
#     np.save(os.path.join(output_dir, movie_updated_path), movie_updated)

#     movie_norm = np.zeros((int(30 * 3.5), 304, 608))
#     movie_norm[15:, :, :] = (m_data[:30 * 3, :, :] * 2.0) / 255.0 - 1.0
#     movie_norm_path = '{}.3s_gs.30fps.304x608.normalized.npy'.format(movie_name)
#     np.save(os.path.join(output_dir, movie_norm_path), movie_norm)


# def upscale_movie(output_dir='movies', movie_name='natural_movie_one', fps=1000):
#     movie_path = '{}.{}s.30fps.304x608.orig.npy'.format(movie_name, '120' if movie_name.endswith('three') else '30')
#     m_data = np.load(os.path.join(output_dir, movie_path))
#     m_data = m_data[:(30 * 3), :, :]
#     m_data = m_data.repeat(34, axis=0)[:(fps * 3)]

#     movie_updated = np.full((int(fps * 3.5), 304, 608), fill_value=127, dtype=np.uint8)
#     movie_updated[int(fps * 0.5):, :, :] = m_data[:(fps * 3), :, :]
#     movie_updated_path = '{}.3s_gs.1000fps.304x608.orig.npy'.format(movie_name)
#     np.save(os.path.join(output_dir, movie_updated_path), movie_updated)

#     movie_norm = np.zeros((int(fps * 3.5), 304, 608))
#     movie_norm[int(fps * 0.5):, :, :] = (m_data[:(fps * 3), :, :] * 2.0) / 255.0 - 1.0
#     movie_norm_path = '{}.3s_gs.1000fps.304x608.normalized.npy'.format(movie_name)
#     np.save(os.path.join(output_dir, movie_norm_path), movie_norm)


def get_natural_scenes(output_dir='bob_images'):
    """Fetches the 118 Brain Obs natural scene images from the data, saves them in npy format"""
    logging.info(f'Saving "natural_scenes" grey-screen images to {output_dir}')
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    boc = BrainObservatoryCache(manifest_file='boc/manifest.json')
    data_set = boc.get_ophys_experiment_data(501498760)
    scenes = data_set.get_stimulus_template('natural_scenes')

    for i in range(0, len(scenes)):
        scene = scenes[i]
        base_name = os.path.join(output_dir, 'scene.{:03d}.gray_{}x{}'.format(i, scene.shape[0], scene.shape[1]))
        file_name = f'{base_name}.png'
        if not os.path.exists(file_name):
            plt.imsave('{}.png'.format(base_name), scene, cmap='gray')
            np.save('{}.npy'.format(base_name), scene)


def create_ns_movie(n_images=118, ips=10, fps=1000.0, res_row=120, res_col=240, gs_pre=500, gs_post=100, rng_seed=None,
                    images_dir='bob_images', output_dir='movies', movie_file=None, overwrite=True, normalize=False):
    frames_per_image = int(float(fps) / ips)  # num of frames that will be given to each image
    frames_per_gs_pre = int(gs_pre / 1000.0 * fps)  # num frames for initial gray screen
    frames_per_gs_post = int(gs_post / 1000.0 * fps)  # num frames for ending gray screen
    n_frames = frames_per_gs_pre + frames_per_image*n_images + frames_per_gs_post
    set_num = '{:02d}'.format(0)
    total_time = n_frames/float(fps)*1000.0

    # Find the movie path, make sure not to overwrite existing movie files
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    if movie_file is not None:
        movie_path = movie_file if os.path.isabs(movie_file) else os.path.join(output_dir, movie_file)
    else:
        movie_path = os.path.join(output_dir, 'ns_{}images.set{}.{}ms.{}fps.{}ips{}.npy'.format(
            n_images, set_num, int(total_time), int(fps), int(ips), '.normalized' if normalize else ''
        ))

        if not overwrite and os.path.exists(movie_path):
            for i in range(1, 100):
                set_num = '{:02d}'.format(i)
                movie_path = 'movies/ns_{}images.set{}.{}ms.{}fps.{}ips{}.npy'.format(
                    n_images, set_num, int(total_time), int(fps), int(ips), '.normalized' if normalize else ''
                )
                if not os.path.exists(movie_path):
                    break
            else:
                raise ValueError('Could not create valid movie path.')


    # Keep track of the order and placement of the images, metadata will be written to csv
    csv_path = movie_path[:-4] + '.metadata.csv'
    img_names = []
    frames_ids = []
    image_times = []

    if frames_per_gs_pre > 0:
        img_names.append('grayscreen')
        frames_ids.append((0, frames_per_gs_pre))
        image_times.append((0, frames_per_gs_pre/float(fps)))

    output_mat = np.zeros((n_frames, res_row, res_col), dtype=float if normalize else np.uint8)
    mat_beg = frames_per_gs_pre
    if rng_seed:
        np.random.seed(rng_seed)
    images_perm = np.random.permutation(range(n_images))
    for image_num, image_id in enumerate(images_perm):
        # Find the image and convert to a 120x240 matrix
        # img_path = movie_jpegs[i]
        img_path = os.path.join(images_dir, 'scene.{:03d}.gray_918x1174.png'.format(image_id))
        pic = Image.open(img_path).convert('L')
        pic = pic.resize((res_col, res_row))  # PIL the row/col order is swapped
        pic_data = np.asarray(pic)
        if normalize:
            pic_data = pic_data.astype(dtype=float) * 2.0 / 255.0 - 1.0

        mat_end = mat_beg + frames_per_image
        output_mat[mat_beg:mat_end, :, :] = pic_data

        # keep track of metadata
        img_names.append(img_path[len(images_dir)+1:])
        frames_ids.append((mat_beg, mat_end))
        image_times.append((mat_beg/float(fps), mat_end/float(fps)))

        mat_beg = mat_end

    np.save(movie_path, output_mat)

    if frames_per_gs_post > 0:
        img_names.append('grayscreen')
        f_beg = n_frames - frames_per_gs_post
        f_end = n_frames
        frames_ids.append((f_beg, f_end))
        image_times.append((f_beg/float(fps), f_end/float(fps)))

    pd.DataFrame({
        'image_id': img_names,
        'begin_frame': [f[0] for f in frames_ids],
        'end_frame': [f[1] for f in frames_ids],
        'begin_time': [t[0] for t in image_times],
        'end_time': [t[1] for t in image_times]
    }).to_csv(csv_path, sep=' ', index=False)


def check_movie(movie_file, frames=None):
    movie_array = np.load(movie_file)
    logging.info(f'Movie {movie_file} has dimensions {movie_array.shape} and data-type {movie_array.dtype}.')

    if movie_array.dtype in [np.float64, float]:
        vmin, vmax = -1.0, 1.0
    else:
        vmin, vmax = 0, 256
    
    if frames is None:
        frames = np.linspace(0, movie_array.shape[0]-1, 5).astype(int)   
    
    fig, axes = plt.subplots(1, len(frames), figsize=(16, 5))
    for idx, frame_num in enumerate(frames):
        axes[idx].imshow(movie_array[frame_num, :, :], cmap='gray', vmin=vmin, vmax=vmax)
        axes[idx].set_title(f'frame #{frame_num}', fontsize='xx-small')
        axes[idx].get_yaxis().set_visible(False)
   
    plt.show()


def create_gray_screen_movie(grey_screen_time, output_dir='movies', movie_file=None, fps=1000.0, res_row=120, 
                             res_col=240, normalize=False):
    """Creates grey-screen for BOb natural scene images"""
    n_frames = int(grey_screen_time*fps/1000.0)

    if movie_file is not None:
        movie_path = movie_file if os.path.isabs(movie_file) else os.path.join(output_dir, movie_file)
    else:
        movie_path = os.path.join(
            output_dir, 
            'grey_screen.{}ms.{}fps{}.npy'.format(grey_screen_time, int(fps), '.normalized' if normalize else '')
        )
    
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
       
    if normalize:
        gs_mat = np.full((n_frames, res_row, res_col), fill_value=0.0, dtype=float)
    else:
        gs_mat = np.full((n_frames, res_row, res_col), fill_value=127, dtype=np.uint8)

    np.save(movie_path, gs_mat)


def get_touchofevil_movies(output_dir='movies', res_row=120, res_col=240, fps=1000, normalize=False):
    frame_conv = int(np.floor(fps/30.0))
    def convert_movie(name, movie):
        t, x, y = movie.shape
        n_frames = frame_conv * t
        movie_updated = np.zeros((n_frames, res_row, res_col), dtype=np.uint8)
        c_frame = 0
        for frame in range(t):
            # Resize resolution
            img = Image.fromarray(movie[frame, :, :], mode='L')
            img = img.resize((res_col, res_row))
            img_data = np.asarray(img)
            if normalize:
                img_data = img_data.astype(dtype=float) * 2.0 / 255.0 - 1.0

            # Upscale frame rate
            movie_updated[c_frame:(c_frame + frame_conv), :, :] = img_data
            c_frame += frame_conv

        np.save('{}/{}.{}ms.{}x{}{}.npy'.format(output_dir, name, c_frame, res_row, res_col, '.normalize' if normalize else ''), movie_updated)

    boc = BrainObservatoryCache(manifest_file='boc/manifest.json')
    data_set = boc.get_ophys_experiment_data(506248008)
    movie = data_set.get_stimulus_template('natural_movie_one')
    convert_movie('natural_movie_one', movie)

    movie = data_set.get_stimulus_template('natural_movie_two')
    convert_movie('natural_movie_two', movie)

    data_set = boc.get_ophys_experiment_data(649409874)
    movie = data_set.get_stimulus_template('natural_movie_three')
    convert_movie('natural_movie_three', movie)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate a movie from Allen Brain Observatory Data.')
    parser.add_argument('movie_type', type=str, nargs=1, choices=['grey-screen', 'natural-scenes', 'touch-of-evil', 'natural-movies', 'check'],
                        help='Options: grey-screen, natural-scenes, touch-of-evil, natural-movies')
    parser.add_argument('--bob-images-dir', type=str, nargs=1, default='bob_images', help='Directory where Natural Scene image files will be stored.')
    parser.add_argument('--n-images', type=int, nargs='?', default=118, help='Number of natural-scenes to use (118 unique available).')
    parser.add_argument('--images-per-sec', type=int, nargs='?', default=4, help='Number of unique images to show each second')
    parser.add_argument('--greyscreen-pre', type=int, nargs='?', default=500, help='Adds a grey-screen to beginning of movie, in ms.')
    parser.add_argument('--greyscreen-post', type=int, nargs='?', default=0, help='Adds a grey-screen to end of movie, in ms.')
    parser.add_argument('--greyscreen-time', type=float, nargs='?', default=2000, help='Time, in ms, for a grey-screen movie.')    

    parser.add_argument('--fps', type=float, nargs='?', default=1000., help='Frames-per-second of movie.')
    parser.add_argument('--res-row', type=int, nargs='?', default=120, help='Resolution of the number of rows (x) of movie.')
    parser.add_argument('--res-col', type=int, nargs='?', default=240, help='Resolution of the number of columns (y) of movie.')
    parser.add_argument('--rng-seed', type=int, nargs='?', default=None, help='Random seed number, used by natural-scenes for determining order.')
    parser.add_argument('--movies-dir', type=str, nargs='?', default='movies', help='Directory where movie will be saved too.')
    parser.add_argument('--movie-path', type=str, nargs='?', default=None, help='Specify the npy filename of the generated movie. By default will be autogenerated.')
    parser.add_argument('--normalize', action='store_true', help='If flag is turned on the movie file will having floating-point values between (-1.0, +1.0). Otherwise movie will be an integer matrix with values [0, 255].')
    args = parser.parse_args()


    logging.basicConfig(level=logging.INFO, format='%(asctime)s: %(message)s')

    movie_type = args.movie_type[0]
    if movie_type == 'natural-scenes':
        get_natural_scenes(output_dir=args.bob_images_dir)
    
        create_ns_movie(
            n_images=args.n_images, 
            ips=args.images_per_sec, 
            fps=args.fps, 
            res_row=args.res_row, 
            res_col=args.res_col, 
            gs_pre=args.greyscreen_pre, 
            gs_post=args.greyscreen_post, 
            rng_seed=args.rng_seed,
            images_dir=args.bob_images_dir, 
            output_dir=args.movies_dir, 
            overwrite=True, 
            normalize=args.normalize
        )
    elif movie_type == 'grey-screen':
        create_gray_screen_movie(
            grey_screen_time=args.greyscreen_time,
            output_dir=args.movies_dir, 
            movie_file=args.movie_path,
            fps=args.fps, 
            res_row=args.res_row, 
            res_col=args.res_col, 
            normalize=args.normalize
        )
    elif movie_type in ['natural-movies', 'touch-of-evil']:
        get_touchofevil_movies(
            output_dir=args.movies_dir, 
            res_row=args.res_row, 
            res_col=args.res_col, 
            fps=args.fps,
            normalize=args.normalize
        )

    elif movie_type == 'check':
        check_movie(args.movie_path)



    exit()


    # Get Bob images, create grey-screen, and create movie from natural scenes
    get_natural_scenes()
    # create_gray_screen()
    # create_ns_movie(n_images=118, fps=1000.0, create_config=True, normalize=True)
    # create_ns_movie(ips=4, overwrite=False)
    create_ns_movie(n, ips=4, overwrite=False)

    # get_touchofevil_movies()

    # check_movie('natural_scenes_full_00/movie_natural_scenes.60fps.918x1174.npy')
    # get_natural_movies()
    # a = np.array([[2, 4, 6, 8], [2, 4, 6, 8]])
    # print(a)
    # print(a.repeat(3, axis=1))
    # exit()

    # natural_movie_1000Hz_gs()

    # get_movies()
    # convert_movies(movie_name='natural_movie_one')
    # upscale_movie(movie_name='natural_movie_one')
    # convert_movies(movie_name='natural_movie_three')
    # upscale_movie(movie_name='natural_movie_three')
    # convert_movies(movie_name='natural_movie_two')
    # upscale_movie(movie_name='natural_movie_two')

