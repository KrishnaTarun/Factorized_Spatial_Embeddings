
import numpy as np
import glob, os, random, math, collections, time, argparse, shutil
from matplotlib import cm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from dataset import ImageData, PreprocessData
import torch
from torch import nn
from model import ConvNet
from deformation import feature_warping, image_warping
from torchsummary import summary


np.random.seed(47)
torch.torch.manual_seed(47)
#****************************************************************************************************

SAVE_FREQ = 20
# SUMMARY_FREQ = 20
BATCH_SIZE = 32
DATA_DIRECTORY = '../FSE_tf/celebA/img_align_celeba/'
LANDMARK_N = 8
DOWNSAMPLE_M = 4
DIVERSITY = 500.
ALIGN = 1.
LEARNING_RATE = 1.e-4
MOMENTUM = 0.5
WEIGHT_DECAY = 0.0005
SCALE_SIZE = 146
CROP_SIZE = 146
MAX_EPOCH = 100
OUTPUT_DIR = './OUTPUT'
CHECKPOINT = './OUTPUT/checkpoint' 
#-------------------------------#
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
# device='cpu'

softmax = torch.nn.Softmax(dim=-1)
AvgPool = torch.nn.AvgPool2d(DOWNSAMPLE_M , stride=1, padding=0)

if not os.path.isdir(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
#--------------------------------#
def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Factorized Spatial Embeddings")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--input_dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the training or testing images.")
    parser.add_argument("--learning_rate", type=float, default=LEARNING_RATE,
                        help="Learning rate for adam.")
    parser.add_argument("--beta1", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--K", type=int, default=LANDMARK_N,
                        help="Number of landmarks.")
    parser.add_argument("--M", type=int, default=DOWNSAMPLE_M,
                        help="Downsampling value of the diversity loss.")
    parser.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--diversity_weight", type=float, default=DIVERSITY,
                        help="Weight on diversity loss.")
    parser.add_argument("--align_weight", type=float, default=ALIGN,
                        help="Weight on align loss.")
    parser.add_argument("--scale_size", type=int, default=SCALE_SIZE,
                        help="Scale images to this size before cropping to CROP_SIZE")
    parser.add_argument("--crop_size", type=int, default=CROP_SIZE,
                        help="CROP images to this size")
    parser.add_argument("--max_epochs", type=int, default=MAX_EPOCH,
                        help="Number of training epochs")
    parser.add_argument("--checkpoint", default=CHECKPOINT,
                        help="Directory with checkpoint to resume training from or use for testing")
    parser.add_argument("--output_dir", default=OUTPUT_DIR,
                        help="Where to put output files")
    parser.add_argument("--save_freq", type=int, default=SAVE_FREQ, help="Save model every save_freq steps")
    return parser.parse_args()

def spatialSoft(out, args):
    
    a_size= out.size()[-1]
    batch_size = out.size()[0]

    x = out.reshape((batch_size,args.K, a_size*a_size))

    x = softmax(x).reshape(batch_size, args.K, a_size, a_size)


    return x

# def weight_decay():



def diversityLoss(p, args):
    ##apply pooling in 
    pool_out = AvgPool(p)
    
    #TODO: make it sum pooling 
    l, _ = torch.max(pool_out, dim=1)
    l = float(args.K) - l.sum(dim=1).sum(dim=1)

    loss = torch.mean(l).unsqueeze(0)
    return loss

def alignLoss(p1, p2, deformation, args):
    
    a_size= p1.size()[-1]
    batch_size = p1.size()[0]
    
    index = torch.arange(0, a_size).float()
    index = torch.reshape(index, (1, a_size))

    #---------grid u----------------------------
    x1_index = index.unsqueeze(0).unsqueeze(0)
    x1_index = x1_index.repeat(batch_size, args.K, a_size,1)

    y1_index = index.transpose(1,0)
    y1_index = y1_index.unsqueeze(0).unsqueeze(0)
    y1_index = y1_index.repeat(batch_size, args.K, 1, a_size)

    norm1 = torch.pow(x1_index, 2)+torch.pow(y1_index,2)
    

    #-----deformed grid points g(v)/g(u)--------
    x2_index = feature_warping(x1_index.permute(0,2,3,1).numpy(),deformation, padding=3)
    y2_index = feature_warping(y1_index.permute(0,2,3,1).numpy(),deformation, padding=3)

    x2_index = torch.from_numpy(x2_index).float().permute(0,3,1,2)
    y2_index = torch.from_numpy(y2_index).float().permute(0,3,1,2)
    
    norm2 = torch.pow(x2_index, 2)+torch.pow(y2_index,2)
    
    #----------------softArgmax---------------------------
    x1_arg = x1_index.to(device)*p1
    x1_arg = x1_arg.sum(2,keepdim=True).sum(3,keepdim=True)
    y1_arg = y1_index.to(device)*p1
    y1_arg = y1_arg.sum(2,keepdim=True).sum(3,keepdim=True)

    loc1 = torch.cat((x1_arg,y1_arg), dim=3)
    ##########
    x2_arg = x2_index.to(device)*p2
    x2_arg = x2_arg.sum(2,keepdim=True).sum(3,keepdim=True)
    y2_arg = y2_index.to(device)*p2
    y2_arg = y2_arg.sum(2,keepdim=True).sum(3,keepdim=True)

    loc2 = torch.cat((x2_arg,y2_arg), dim=3)
    # --------------------------------------------- 

    
    #terms of linear time implementation of loss
    t1 = norm1.to(device)*p1
    t1 = t1.sum(2,keepdim=True).sum(3)
    t2 = norm2.to(device)*p2
    t2 = t2.sum(2,keepdim=True).sum(3)
    t3 = torch.matmul(loc1, loc2.transpose(3,2)).squeeze(3)
    loss= torch.mean(t1 + t2 - 2.* t3)
    return args.align_weight * loss.unsqueeze(0)

def train(dataloader, net, optimizer, args):

    if os.path.isfile(args.checkpoint):
        print ("loading from checkpoint...")
        checkpoint = torch.load(args.checkpoint)
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']

    steps = 0
    store =1000000
    net.train()
    for idx in range(args.max_epochs):

        for i_batch, sample_batched in enumerate(dataloader):
            # print(i_batch, sample_batched['image'].size(), sample_batched['deformed_image'].size(),sample_batched['deformation'].size())
            i_image, d_image, deformation =  sample_batched['image'],sample_batched['deformed_image'],\
                                           sample_batched['deformation']
            ############
            
            optimizer.zero_grad()
            i_image, d_image, deformation = i_image.float().to(device),d_image.float().to(device), deformation.numpy()

            out1 = net(i_image)
            out2 = net(d_image)
            ############
            
            #Spatial softmax
            p1 = spatialSoft(out1, args)
            p2 = spatialSoft(out2, args)

            #------ALIGN_LOSS---------------
            alignL = alignLoss(p1, p2, deformation, args)
            
            #------DIVERSITY_LOSS-----------
            diverL1 = diversityLoss(p1, args)
            diverL2 = diversityLoss(p2, args)

            loss = alignL + args.diversity_weight * (diverL1 + diverL2)
            loss.backward()
            optimizer.step()

            if steps % args.save_freq==0:
                print("#epoch {} | iter {} | total loss {:.3f}#".
                        format(idx, steps, loss.item()))
                print("Align Loss: {:.3f}".format(alignL.item()))
                print("Diversity Loss: {:.3f}".format((diverL1 + diverL2).item()))  
                print("#------------------------------------------------------------#")      
                if loss.item() < store:
                    torch.save({
                    'epoch': idx,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,}, 
                    args.checkpoint)
                    store = loss.item()

            steps+=1
            
       


def main():

    args = get_arguments()

    face_dataset = ImageData(root_dir=args.input_dir,\
                                    transform=transforms.Compose([PreprocessData(args.scale_size,args.crop_size)]))
    dataloader = DataLoader(face_dataset, batch_size=args.batch_size, shuffle=True)
    
    ########### setup network ##############
    
    net = ConvNet(3, args.K).to(device)
    #----------Weight Initialization---------------
    def init_weights(m):   
        if type(m) == nn.Conv2d:
            nn.init.normal_(m.weight,0,0.02)
    
    net.apply(init_weights)
    #---------------------------------------------
    summary(net, (3, 128, 128))
    # for name, param in net.named_parameters():
    #     if param.requires_grad:
    #         print(name, param.data.size(), type(param))    
    optimizer = torch.optim.Adam(net.parameters(),lr = args.learning_rate, weight_decay=args.weight_decay)
    
    # #########################################
    print()
    train(dataloader, net, optimizer, args)


if __name__ == '__main__':
    main()
