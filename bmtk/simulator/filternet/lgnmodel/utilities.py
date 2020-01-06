import array
import skimage.transform as transform
import numpy as np


def get_vanhateren(filename, src_dir):
    with open(filename, 'rb') as handle:
        s = handle.read()
    arr = array.array('H', s)
    arr.byteswap()
    return np.array(arr, dtype='uint16').reshape(1024, 1536)


def convert_tmin_tmax_framerate_to_trange(t_min, t_max, frame_rate):
    duration = t_max-t_min
    number_of_frames = duration*frame_rate  # Assumes t_min/t_max in same time units as frame_rate
    dt = 1./frame_rate
    return t_min+np.arange(number_of_frames+1)*dt


def get_rotation_matrix(rotation, shape):
    """Angle in degrees"""
    shift_y, shift_x = np.array(shape) / 2.
    tf_rotate = transform.SimilarityTransform(rotation=np.deg2rad(rotation))
    tf_shift = transform.SimilarityTransform(translation=[-shift_x, -shift_y])
    tf_shift_inv = transform.SimilarityTransform(translation=[shift_x, shift_y])
    return (tf_shift + (tf_rotate + tf_shift_inv))


def get_translation_matrix(translation):
    shift_x, shift_y = translation
    tf_shift = transform.SimilarityTransform(translation=[-shift_x, shift_y])
    return tf_shift


def get_scale_matrix(scale, shape):
    shift_y, shift_x = np.array(shape) / 2.
    tf_rotate = transform.SimilarityTransform(scale=(1./scale[0], 1./scale[1]))
    tf_shift = transform.SimilarityTransform(translation=[-shift_x, -shift_y])
    tf_shift_inv = transform.SimilarityTransform(translation=[shift_x, shift_y])
    return tf_shift + (tf_rotate + tf_shift_inv)


def apply_transformation_matrix(image, matrix):
    return transform.warp(image, matrix)


def get_convolution_ind(curr_fi, flipped_t_inds, kernel, data):
    
    flipped_and_offset_t_inds = flipped_t_inds + curr_fi
    
    if np.all(flipped_and_offset_t_inds >= 0):
        # No negative entries; still might be over the end though:
        try:
            return np.dot(data[flipped_and_offset_t_inds], kernel)
        
        except IndexError:
             
            # Requested some indices out of range of data:
            indices_within_range = np.where(flipped_and_offset_t_inds < len(data))
            valid_t_inds = flipped_and_offset_t_inds[indices_within_range]
            valid_kernel = kernel[indices_within_range]
            return np.dot(data[valid_t_inds], valid_kernel)
    else:
        # Only some are negative, so restrict:
        indices_within_range = np.where(flipped_and_offset_t_inds >= 0)
        valid_t_inds = flipped_and_offset_t_inds[indices_within_range]
        valid_kernel = kernel[indices_within_range]
         
        return np.dot(data[valid_t_inds], valid_kernel)


def get_convolution(t, frame_rate, flipped_t_inds, kernel, data):
    # Get frame indices: 
    fi = frame_rate*float(t)
    fim = int(np.floor(fi))
    fiM = int(np.ceil(fi))
    
    if fim != fiM:
     
        # Linear interpolation:
        sm = get_convolution_ind(fim, flipped_t_inds, kernel, data)
        sM = get_convolution_ind(fiM, flipped_t_inds, kernel, data)
        return sm*(1-(fi-fim)) + sM*(fi-fim)
    
    else:
        # Requested time is exactly one piece of data:
        return get_convolution_ind(fim, flipped_t_inds, kernel, data)
