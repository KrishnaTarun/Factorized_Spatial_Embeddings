import numpy as np
import glob, os, random, math, collections, time, argparse, shutil
from matplotlib import cm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from dataset import ImageData, PreprocessData
import torch
from torch import nn
from model import ConvNet
import skimage  
import matplotlib.pyplot as plt
import cv2
import numpy as np
from skimage import io, img_as_float, draw

BATCH_SIZE = 1
DATA_DIRECTORY = '../FSE_tf/celebA/test_MAFL/'
LANDMARK_N = 8
SCALE_SIZE = 146
CROP_SIZE = 146
OUTPUT_DIR = './OUTPUT'
CHECKPOINT = './OUTPUT/checkpoint' 
MODE = 'test'

resize = transforms.Resize(128)
#---------------------------------#
def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Factorized Spatial Embeddings")
    parser.add_argument("--mode", default=MODE, choices=["train", "test"])
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--input_dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the training or testing images.")
    parser.add_argument("--K", type=int, default=LANDMARK_N,
                        help="Number of landmarks.")
    parser.add_argument("--scale_size", type=int, default=SCALE_SIZE,
                        help="Scale images to this size before cropping to CROP_SIZE")
    parser.add_argument("--crop_size", type=int, default=CROP_SIZE,
                        help="CROP images to this size")
    parser.add_argument("--checkpoint", default=CHECKPOINT,
                        help="Directory with checkpoint to resume training from or use for testing")
    parser.add_argument("--output_dir", default=OUTPUT_DIR,
                        help="Where to put output files")
    parser.add_argument("--img_folder",type=str, default='images',help="save the predicted landmarks")
    
    return parser.parse_args()

args = get_arguments()
def spatialSoft(out, args):
    
    a_size= out.size()[-1]
    batch_size = out.size()[0]

    x = out.reshape((-1,args.K, a_size*a_size))

    x = softmax(x).reshape(batch_size, args.K, a_size, a_size)


    return x
def landmark_colors(n_landmarks):
    """Compute landmark colors.

    Returns:
      An array of RGB values.
    """
    cmap = cm.get_cmap('hsv')
    landmark_color = []
    landmark_color.append((0., 0., 0.))
    for i in range(n_landmarks):
        landmark_color.append(cmap(i/float(n_landmarks))[0:3])
    landmark_color = np.array(landmark_color)
    return landmark_color

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = 'cpu'

softmax = torch.nn.Softmax(dim=-1)

if not os.path.isdir(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

if not os.path.isdir(os.path.join(OUTPUT_DIR, args.img_folder)):
    os.makedirs(os.path.join(OUTPUT_DIR, args.img_folder))
out_name = os.path.join(OUTPUT_DIR, args.img_folder)
#--------------------------------#


face_dataset = ImageData(root_dir=args.input_dir,\
                                transform=transforms.Compose([PreprocessData(args.scale_size,args.crop_size, mode= args.mode)]))
dataloader = DataLoader(face_dataset, batch_size = args.batch_size,
                        shuffle = True)
#--------------- Load Weights ----------------#
net = ConvNet(3, args.K).to(device)
checkpoint = torch.load(args.checkpoint)
net.load_state_dict(checkpoint['model_state_dict'])
#---------------------------------------------#




#-------------visualize----------------------#
net.eval()
with torch.no_grad():

    for i_batch, sample_batched in enumerate(dataloader):
        i_image =  sample_batched['image']
        i_name  = sample_batched['name'][0].split("/")[-1]
        print(i_name)
        i_image = i_image.float().to(device)
        out = net(i_image)
        
        #single batch operation
        p = spatialSoft(out,args).squeeze(0)
        # print(p.size())
        pred = torch.mean(p, dim=0)

        if device=="cpu":
            pred = pred.numpy()
        else:
            pred = pred.cpu().numpy()

        pred_max, _ = torch.max(torch.max(p,dim=1)[0], dim=1)
        pred_max = pred_max.unsqueeze(1)
        pred_max = pred_max.unsqueeze(1)
        pred_max = torch.eq(p, pred_max).float()

        mask = torch.arange(start=1, end=args.K+1).float()
        mask = torch.reshape(mask, (args.K, 1, 1))
        mask = mask.repeat(1, p.size()[1],p.size()[1])
        mask = mask.to(device) * pred_max
        #
        mask, _ = torch.max(mask, dim=0)
        indx = torch.nonzero(mask)
        mask_ = torch.gt(mask,0)
        val = torch.masked_select(mask, mask_, out=None) 
        # print(val, indx)
        
        clr = landmark_colors(args.K)

        def visualizePlot(mask, i_image, val, indx):

            resize_scale = 1.0*(args.crop_size/mask.size()[0])
            if device=="cpu":
                mask = skimage.transform.resize(mask.numpy(),\
                        (args.crop_size, args.crop_size),anti_aliasing=True, mode='reflect')
                i_image = i_image.squeeze(0).permute(1,2,0).numpy().copy()
                val  = val.numpy()
                indx = indx.numpy()
            
            else:
                mask = skimage.transform.resize(mask.cpu().numpy(),\
                        (args.crop_size, args.crop_size),anti_aliasing=True, mode='reflect')
                i_image = i_image.squeeze(0).permute(1,2,0).cpu().numpy()
                val  = val.cpu().numpy()
                indx = indx.cpu().numpy()
            
            val_idx = zip(val,indx)
            
            
            mask = np.greater(mask, 0).astype(float)
            mask = np.expand_dims(mask, axis=2)
            
            #deprocess
            i_image = (i_image+1)/2 
            # i_image = i_image*(1-mask) + mask

            id_=0
            for val, idx in val_idx:
                cX = round(idx[1]*resize_scale)
                cY = round(idx[0]*resize_scale)

                cv2.circle(i_image, (int(cX), int(cY)), 2, clr[id_], -1)
                cv2.putText(i_image, str(int(val)), (int(cX), int(cY)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.25, clr[id_], 1)

                id_+=1

            io.imsave(os.path.join(out_name, i_name), img_as_float(i_image))
            # plt.imshow(i_image)
            # plt.show()

        visualizePlot(mask_, i_image, val, indx)
        # break
        
       




