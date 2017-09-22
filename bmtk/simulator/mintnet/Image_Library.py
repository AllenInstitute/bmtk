import numpy as np
import os
from PIL import Image

# Image_Batch
#   .data (image_data)
#   .image_dir, .new_size


# add seed for random
# call should return indices into im_list
class Image_Experiment(object):

    def __init__(self,stuff):

        self.image_dir
        self.new_size
        self.sample_indices
        self.im_list
        # creating of pandas table, template





class Image_Library (object):
    def __init__(self, image_dir,new_size=(128,192)):  # NOTE:  change this so that sequential is a class variable, not an argument to the call
        self.image_dir = image_dir
        self.new_size = new_size

        im_list = os.listdir(image_dir)

        remove_list = []
        for im in im_list:
            if im[-5:]!='.tiff' and im[-5:]!='.JPEG' and im[-4:]!='.jpg':
                remove_list.append(im)

        for im in remove_list:
            im_list.remove(im)

        self.im_list = im_list

        self.current_location = 0 # used for sequential samples
        self.lib_size = len(self.im_list)

    def __call__(self,num_samples, sequential=False):

        image_data = np.zeros([num_samples,self.new_size[0],self.new_size[1],1],dtype=np.float32)

        if sequential:
            if self.lib_size-self.current_location > num_samples:
                sample_indices = np.arange(self.current_location,self.current_location + num_samples)
                self.current_location += num_samples
            else:
                sample_indices = np.arange(self.current_location,self.lib_size)
                self.current_location = 0
        else:
            sample_indices = np.random.randint(0,len(self.im_list),num_samples)

        for i,s in enumerate(sample_indices):
            im = Image.open(os.path.join(self.image_dir,self.im_list[s]))
            im = im.convert('L')
            im = im.resize((self.new_size[1],self.new_size[0]))
            image_data[i,:,:,0] = np.array(im,dtype=np.float32)

        return image_data

    def create_experiment(self):

        data = self()
        return Image_Experiment(stuff)

    def experiment_from_table(self,table):
        pass

    def to_h5(self,sample_indices=None):
        pass

    def template(self):
        pass

    def table(self,*params):
        pass
