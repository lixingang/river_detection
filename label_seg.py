from PIL import Image
import numpy as np
import os
from tqdm import tqdm 
class Grid_labels(object):
    def __init__(self, path,o_dir, grid_num=32, mini=False):
        self.grid_num = grid_num
        self.true_value = 200
        self.o_dir = o_dir
        self.path = path
        self.mini = mini
    def batch_seg(self):
        if self.mini==False:
            label_list = self._read_label_list(self.path)
            for i in tqdm(label_list):
                if not i.split('.')[-1]=='png':
                    continue
                self._seg(i)
        else:
            label_list = self._read_label_list(self.path)
            print(len(label_list))
            for i in tqdm(label_list):
                if not i.split('.')[-1]=='png':
                    continue
                self._seg_mini(i)
        
    def _read_label_list(self, path):
        # return the list of label-image-path
        ll = [os.path.join(path,i) for i in os.listdir(path)]
        return ll
    def _read_and_resize(self, i, scale=(512,512)):
        im = Image.open(i)
        im = np.array(im.resize(scale))
        width, height = im.shape
        return im,width,height
    def _seg(self, path):
        o_name = path.split('/')[-1]
        img, w, h = self._read_and_resize(path)
        assert w%self.grid_num==0 and h%self.grid_num==0  
        w_scale = w//self.grid_num
        h_scale = h//self.grid_num
        label_copy = np.zeros(shape=(w,h))
        for i in range(0,self.grid_num):
            for j in range(0,self.grid_num):
                current_block = img[i*w_scale:(i+1)*w_scale, j*w_scale:(j+1)*w_scale]
                if np.sum(current_block>10):
                    label_copy[i*h_scale:(i+1)*h_scale, j*h_scale:(j+1)*h_scale] = self.true_value
                output = Image.fromarray(label_copy).convert('RGB')
                output.save(os.path.join(self.o_dir, o_name))
    def _seg_mini(self, path):
        o_name = path.split('/')[-1]
        img, w, h = self._read_and_resize(path)
        assert w%self.grid_num==0 and h%self.grid_num==0  
        w_scale = w//self.grid_num
        h_scale = h//self.grid_num
        label_copy = np.zeros(shape=(self.grid_num, self.grid_num))
        for i in range(0,self.grid_num):
            for j in range(0,self.grid_num):
                current_block = img[i*w_scale:(i+1)*w_scale, j*w_scale:(j+1)*w_scale]
                if np.sum(current_block>10):
                    label_copy[i,j] = self.true_value
                output = Image.fromarray(label_copy).convert('RGB')
                output.save(os.path.join(self.o_dir, o_name))

if __name__ == '__main__':
    print('[I] Seg Start')
    g = Grid_labels('labels_pixel','labels_seg_mini', mini=True)
    g.batch_seg()
    print('[I] Seg Done')