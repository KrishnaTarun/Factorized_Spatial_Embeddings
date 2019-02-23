import os
import numpy as np
import tensorflow as tf

import torch
from skimage import io, img_as_float
import sys
from deformation import feature_warping, image_warping
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import skimage  
tf.enable_eager_execution()


np.random.seed(47)
torch.torch.manual_seed(47)
tf.random.set_random_seed(47)

class ImageData(Dataset):

    def __init__(self,root_dir, transform=None):
        
        self.root_dir = root_dir
        self.image_files = os.listdir(root_dir)
        self.transform = transform


    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        
        image_name = os.path.join(self.root_dir, self.image_files[idx])
        
        #convert to float---> image range [0, 1]
        try:
            image = img_as_float(io.imread(image_name))
            assert (image.shape[2] == 3),"image does not have required channels!!"
        
        except Exception as e:
            print(image_name)
            print(str(e))
            # continue


        sample = {'image': image,'name': image_name}

        if self.transform:
            sample = self.transform(sample)

        return sample

class PreprocessData(object):

    
    def __init__(self, scale_size, crop_size, mode="train"):
        assert isinstance(scale_size, int)
        assert isinstance(crop_size, int)
        
        self.scale_size = scale_size
        self.crop_size  = crop_size
        self.mode = mode

    def __call__(self, sample):
        
        image = sample['image']
        name = sample['name']


        #image rane [0 1]-->[-1 1]
        image = image*2-1
        
        #transform
        image = skimage.transform.resize(image, (self.scale_size, self.scale_size),anti_aliasing=True, mode='reflect')

        #crop offset
        top, left = np.floor(np.random.uniform(0,self.scale_size - self.crop_size + 1, 2)).astype(np.int32)
        if self.scale_size > self.crop_size:
            image = image[top:self.crop_size, left:self.crop_size]
        
        elif self.scale_size < self.crop_size:

            raise Exception("scale size cannot be less than crop size")

        if self.mode == "train":
            #deform input images
            input_image, _ =image_warping(image, w=0.0) 
            
            # x' = x.g1.g2
            deformed_image, deformation = image_warping(input_image, w=0.1)
            deformed_image, deformation = deformed_image, deformation 
            deformation = deformation[0]
            
            #crop after warping
            input_image    = input_image[5:128, 5:128]
            deformed_image = deformed_image[5:128, 5:128] 

            #resize    
            input_image    = skimage.transform.resize(input_image, (128, 128),anti_aliasing=True, mode='reflect')
            deformed_image = skimage.transform.resize(deformed_image, (128, 128),anti_aliasing=True, mode='reflect')

            #clip the values again
            input_image = np.clip(input_image, a_min=-1, a_max =1)
            deformed_image = np.clip(deformed_image, a_min=-1, a_max =1)

            #transpose channel and convert to tensor
            input_image    = torch.from_numpy(input_image.transpose(2,0,1))
            deformed_image = torch.from_numpy(deformed_image.transpose(2,0,1))


            return {'image':input_image,'deformed_image':deformed_image,
                    'deformation': deformation}       

        if self.mode =='test':

            #resize    
            input_image = skimage.transform.resize(image, (128, 128), anti_aliasing=True, mode='reflect')

            #clip the values again
            input_image = np.clip(input_image, a_min=-1, a_max =1)
            input_image = torch.from_numpy(input_image.transpose(2,0,1))

            return {'image': input_image,'name': name}




if __name__=="__main__":
    
    face_dataset = ImageData(root_dir='../FSE_tf/celebA/img_align_celeba',\
                                transform=transforms.Compose([PreprocessData(146,146,mode='train')]))
    dataloader = DataLoader(face_dataset, batch_size=32,
                        shuffle=True)


    for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch, sample_batched['image'].size(),sample_batched['deformation'].size())
        # plt.imshow(sample['image'])
        # plt.show()
        # plt.close()
        # break


#     