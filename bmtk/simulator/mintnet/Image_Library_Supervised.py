from PIL import Image
import numpy as np
import os

class Image_Library_Supervised (object):

    def __init__(self,image_dir,new_size=(256,256)):

        self.categories = os.listdir(image_dir)

        self.num_categories = len(self.categories)  #len(image_dir_list)
        self.image_dir_list = [os.path.join(image_dir,x) for x in self.categories]
        self.new_size = new_size


        # self.categories = []
        # for d in self.image_dir_list:
        #     self.categories += [os.path.basename(d)]

        self.im_lists = {}
        for i,cat in enumerate(self.categories):
            d = self.image_dir_list[i]
            if os.path.basename(d[0])=='.': continue
            self.im_lists[cat] = os.listdir(d)

        for cat in self.im_lists:
            remove_list = []
            for im in self.im_lists[cat]:
                if im[-4:]!='.jpg':
                    remove_list.append(im)

            for im in remove_list:
                self.im_lists[cat].remove(im)


        self.current_location = np.zeros(len(self.categories)) # used for sequential samples
        self.lib_size = [len(self.im_lists[x]) for x in self.categories]
        #self.lib_size = len(self.im_list)

    def __call__(self,num_samples,sequential=False):

        image_data = np.zeros([self.num_categories*num_samples,self.new_size[0],self.new_size[1],1],dtype=np.float32)

        # y_vals = np.tile(np.arange(self.num_categories),(num_samples,1)).T.flatten()
        # y_vals = y_vals.astype(np.float32)

        y_vals = np.zeros([num_samples*self.num_categories,self.num_categories],np.float32)

        for i,cat in enumerate(self.categories):

            y_vals[num_samples*i:num_samples*i+num_samples].T[i] = 1

            if sequential:
                if self.lib_size[i]-self.current_location[i] > num_samples:
                    sample_indices = np.arange(self.current_location[i],self.current_location[i] + num_samples,dtype=np.int64)
                    self.current_location[i] += num_samples
                else:
                    sample_indices = np.arange(self.current_location[i],self.lib_size[i],dtype=np.int64)
                    self.current_location[i] = 0
            else:
                sample_indices = np.random.randint(0,len(self.im_lists[cat]),num_samples)

            for j,s in enumerate(sample_indices):
                im = Image.open(os.path.join(self.image_dir_list[i],self.im_lists[cat][s]))
                im = im.convert('L')
                im = im.resize((self.new_size[1],self.new_size[0]))
                index = j + num_samples*i
                image_data[index,:,:,0] = np.array(im,dtype=np.float32)

        return y_vals, image_data

