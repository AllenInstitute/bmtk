# Copyright 2017. Allen Institute. All rights reserved
#
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the
# following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following
# disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following
# disclaimer in the documentation and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote
# products derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
# INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
import numpy as np
import os
from PIL import Image
import pandas as pd

class NaturalScenes (object):
    def __init__(self, new_size=(64,112), mode='L', dtype=np.float32, start_time=0, trial_length=250, add_channels=False):

        self.new_size = new_size
        self.mode = mode
        self.dtype = dtype
        self.add_channels = add_channels



    def random_sample(self, n):
        sample_indices = np.random.randint(0, self.num_images, n)
        return self.stim_template[sample_indices]

    # also a method random_sample_with_labels ?
    def random_sample_with_labels(self, n):
        pass

    def get_image_input(self,**kwargs):
        return self.stim_template

    def add_gray_screen(self):

        gray_screen = np.ones(self.new_size,dtype=self.dtype)*127  # using 127 as "gray" value
        if self.add_channels:
            gray_screen = gray_screen[:,:,np.newaxis]
        self.stim_template = np.vstack([self.stim_template, gray_screen[np.newaxis,:,:]])

        start = int(self.stim_table.tail(1)['end']) + 1
        end = start+self.trial_length-1  #make trial_length an argument of this function?
        frame = int(self.stim_table.tail(1)['frame']) + 1

        self.stim_table = self.stim_table.append(pd.DataFrame([[frame,start,end]],columns=['frame','start','end']),ignore_index=True)

        self.label_dataframe = self.label_dataframe.append(pd.DataFrame([['gray_screen']],columns=['image_name']),ignore_index=True)
        self.num_images = self.num_images + 1

    @classmethod
    def with_brain_observatory_stimulus(cls, new_size=(64,112), mode='L', dtype=np.float32, start_time=0, trial_length=250, add_channels=False):

        from sys import platform

        if platform=='linux2':
            image_dir = '/data/mat/iSee_temp_shared/CAM_Images.icns'
        elif platform=='darwin':

            image_dir = '/Users/michaelbu/Data/Images/CAM_Images.icns'
            if not os.path.exists(image_dir):
                print("Detected platform:  OS X.  I'm assuming you've mounted \\\\aibsdata\\mat at /Volumes/mat/")
                image_dir = '/Volumes/mat/iSee_temp_shared/CAM_Images.icns'


        elif platform=='win32':
            image_dir = r'\\aibsdata\mat\iSee_temp_shared\CAM_Images.icns'

        #image_dir = '/Users/michaelbu/Data/Images/CAM_Images'  # change this to temp directory on aibsdata
        new_ns =  cls.with_new_stimulus_from_dataframe(image_dir=image_dir, new_size=new_size, mode=mode, dtype=dtype, start_time=start_time, trial_length=trial_length, add_channels=add_channels)

        new_ns.add_gray_screen()

        return new_ns

    @staticmethod
    def generate_stim_table(T,start_time=0,trial_length=250):
        '''frame_length is in milliseconds'''

        start_time_array = trial_length*np.arange(T) + start_time
        column_list  = [np.arange(T),start_time_array, start_time_array+trial_length-1]  # -1 is because the tables in BOb use inclusive intervals, so we'll stick to that convention
        cols = np.vstack(column_list).T
        stim_table = pd.DataFrame(cols,columns=['frame','start','end'])

        return stim_table

    def to_h5(self,sample_indices=None):
        pass

    @classmethod
    def with_new_stimulus_from_folder(cls, image_dir, new_size=(64,112), mode='L', dtype=np.float32, start_time=0, trial_length=250, add_channels=False):

        new_ns = cls(new_size=new_size, mode=mode, dtype=dtype, start_time=start_time, trial_length=trial_length, add_channels=add_channels)

        new_ns.im_list = os.listdir(image_dir)
        new_ns.image_dir = image_dir

        stim_list = []
        for im in new_ns.im_list:
            try:
                im_data = Image.open(os.path.join(new_ns.image_dir,im))
            except IOError:
                print("Skipping file:  ", im)
                new_ns.im_list.remove(im)

            im_data = im_data.convert(new_ns.mode)
            if new_size is not None:
                im_data = im_data.resize((new_ns.new_size[1], new_ns.new_size[0]))
            im_data = np.array(im_data,dtype=new_ns.dtype)
            if add_channels:
                im_data = im_data[:,:,np.newaxis]
            stim_list.append(im_data)

        new_ns.stim_template = np.stack(stim_list)
        new_ns.num_images = new_ns.stim_template.shape[0]

        t,y,x = new_ns.stim_template.shape
        new_ns.new_size = (y,x)

        new_ns.trial_length = trial_length
        new_ns.start_time = start_time
        new_ns.stim_table = new_ns.generate_stim_table(new_ns.num_images,start_time=new_ns.start_time,trial_length=new_ns.trial_length)

        new_ns.label_dataframe = pd.DataFrame(columns=['image_name'])
        new_ns.label_dataframe['image_name'] = new_ns.im_list

        return new_ns

    @classmethod
    def with_new_stimulus_from_dataframe(cls, image_dir, new_size=(64,112), mode='L', dtype=np.float32, start_time=0, trial_length=250, add_channels=False):
        '''image_dir should contain a folder of images called 'images' and an hdf5 file with a
        dataframe called 'label_dataframe.h5' with the frame stored in the key 'labels'.
        dataframe should have columns ['relative_image_path','label_1', 'label_2', ...]'''

        new_ns = cls(new_size=new_size, mode=mode, dtype=dtype, start_time=start_time, trial_length=trial_length, add_channels=add_channels)

        image_path = os.path.join(image_dir,'images')
        label_dataframe = pd.read_hdf(os.path.join(image_dir,'label_dataframe.h5'),'labels')
        new_ns.label_dataframe = label_dataframe

        new_ns.image_dir = image_path
        new_ns.im_list = list(label_dataframe.image_name)

        stim_list = []
        for im in new_ns.im_list:
            try:
                im_data = Image.open(os.path.join(image_path,im))
            except IOError:
                print("Skipping file:  ", im)
                new_ns.im_list.remove(im)

            im_data = im_data.convert(new_ns.mode)
            if new_size is not None:
                im_data = im_data.resize((new_ns.new_size[1], new_ns.new_size[0]))
            im_data = np.array(im_data,dtype=new_ns.dtype)
            if add_channels:
                im_data = im_data[:,:,np.newaxis]
            stim_list.append(im_data)

        new_ns.stim_template = np.stack(stim_list)
        new_ns.num_images = new_ns.stim_template.shape[0]

        if add_channels:
            t,y,x,_ = new_ns.stim_template.shape
        else:
            t,y,x = new_new.stim_template.shape
        new_ns.new_size = (y,x)

        new_ns.trial_length = trial_length
        new_ns.start_time = start_time
        new_ns.stim_table = new_ns.generate_stim_table(new_ns.num_images,start_time=new_ns.start_time,trial_length=new_ns.trial_length)

        return new_ns

    @staticmethod
    def create_image_dir_from_hierarchy(folder, new_path, label_names=None):

        import shutil

        image_dataframe = pd.DataFrame(columns=["image_name"])

        if os.path.exists(new_path):
            raise Exception("path  "+new_path+"  already exists!")

        os.mkdir(new_path)
        os.mkdir(os.path.join(new_path,'images'))
        for path, sub_folders, file_list in os.walk(folder):

            for f in file_list:
                try:
                    im_data = Image.open(os.path.join(path,f))
                except IOError:
                    print("Skipping file:  ", f)
                    im_data = None

                if im_data is not None:
                    shutil.copy(os.path.join(path,f), os.path.join(new_path,'images',f))
                    image_name = f
                    label_vals = os.path.split(os.path.relpath(path,folder))
                    if label_names is not None:
                        current_label_names = label_names[:]
                    else:
                        current_label_names = []

                    if len(label_vals) > current_label_names:
                        labels_to_add = ["label_"+str(i) for i in range(len(current_label_names), len(label_vals))]
                        current_label_names += labels_to_add
                    elif len(label_vals) < current_label_names:
                        current_label_names = current_label_names[:len(label_vals)]

                    vals = [f] + list(label_vals)
                    cols = ['image_name']+current_label_names
                    new_frame = pd.DataFrame([vals],columns=cols)

                    image_dataframe = image_dataframe.append(new_frame,ignore_index=True)

        image_dataframe.to_hdf(os.path.join(new_path,'label_dataframe.h5'),'labels')

    # @staticmethod
    # def add_object_to_image(image, object_image):
    #
    #     new_image = image.copy()
    #     new_image[np.isfinite(object_image)] = object_image[np.isfinite(object_image)]
    #     return new_image

    @staticmethod
    def add_object_to_template(template, object_image):

        if template.ndim==3:
            T,y,x = template.shape
        elif template.ndim==4:
            T,y,x,K = template.shape
        else:
            raise Exception("template.ndim must be 3 or 4")

        if object_image.ndim < template.ndim-1:
            object_image=object_image[:,:,np.newaxis]

        new_template = template.copy()
        new_template[:,np.isfinite(object_image)] = object_image[np.isfinite(object_image)]

        return new_template

    def add_objects_to_foreground(self, object_dict):

        template_list = []

        if self.label_dataframe is None:
            self.label_dataframe = pd.DataFrame(columns=['object'])

        new_label_dataframe_list = []

        for obj in object_dict:
            template_list.append(self.add_object_to_template(self.stim_template,object_dict[obj]))
            obj_dataframe = self.label_dataframe.copy()
            obj_dataframe['object'] = [ obj for i in range(self.num_images) ]
            new_label_dataframe_list.append(obj_dataframe)

        self.stim_template = np.vstack(template_list)
        self.label_dataframe = pd.concat(new_label_dataframe_list,ignore_index=True)

        self.num_images = self.stim_template.shape[0]

        self.stim_table = self.generate_stim_table(self.num_images,start_time=self.start_time,trial_length=self.trial_length)


    @staticmethod
    def create_object_dict(folder, background_shape=(64,112), dtype=np.float32, rotations=False):

        from scipy.misc import imresize

        # resize function to preserve the nans in the background
        def resize_im(im,new_shape):
            def mask_for_nans():
                mask = np.ones(im.shape)
                mask[np.isfinite(im)] = 0
                mask = imresize(mask,new_shape,interp='nearest')

                return mask.astype(np.bool)

            new_im = im.copy()
            new_im = new_im.astype(dtype)
            new_im[np.isnan(new_im)] = -1.
            new_im = imresize(new_im,new_shape,interp='nearest')

            new_im = new_im.astype(dtype)
            new_im[mask_for_nans()] = np.nan

            return new_im

        def im_on_background(im, shift=None):
            bg = np.empty(background_shape)
            bg[:] = np.nan

            buffer_x = (background_shape[1] - im.shape[1])/2
            buffer_y = (background_shape[0] - im.shape[0])/2

            bg[buffer_y:im.shape[0]+buffer_y, buffer_x:im.shape[1]+buffer_x] = im

            return bg

        im_list = os.listdir(folder)

        obj_dict = {}

        for im_file in im_list:
            try:
                im = np.load(os.path.join(folder,im_file))
            except IOError:
                print("skipping file:  ", im_file)
                im = None

            if im is not None:
                new_shape = (np.min(background_shape), np.min(background_shape))
                im = resize_im(im,new_shape)
                obj_dict[im_file[:-4]] = im_on_background(im)
                if rotations:
                    im_rot=im.copy()
                    for i in range(3):
                        im_rot = np.rot90(im_rot)
                        obj_dict[im_file[:-4]+'_'+str(90*(i+1))] = im_on_background(im_rot)

        return obj_dict
