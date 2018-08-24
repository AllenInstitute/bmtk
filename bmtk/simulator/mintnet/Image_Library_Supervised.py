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

