
import numpy as np



def image_warping(img, w):
    t_ = np.array([  # target position  
        [-1., -1.],
        [1., -1.],
        [-1., 1.],
        [1., 1.],
    ])
    grid = t_.reshape([1, 4, 2]).astype(np.float32)
    #---------------------------------------------------- 
    CROP_SIZE = img.shape[1]
    rotation = np.random.uniform(low = -0.5, high=0.5, size=(1,1)).astype(np.float32) 
    
    x_translation = np.random.normal(loc=0.0, scale=0.0+w, size=(1,1)).astype(np.float32)
    y_translation = np.random.normal(loc=0.0, scale=0.0+w, size=(1,1)).astype(np.float32)
    #-------------------------------------------
    x_scale = np.random.uniform( low=0.8-w, high=1.1+w, size=(1,1)).astype(np.float32)
    y_scale = x_scale + np.random.normal( loc=0.0, scale=0.1, size=(1, 1)).astype(np.float32)#tf.random_uniform([1, 1], minval=0.6, maxval=1.2, dtype=tf.float32)
    
    a1 = np.concatenate((x_translation, x_scale*np.cos(rotation), -1.*y_scale*np.sin(rotation)), axis=1)
    a2 = np.concatenate((y_translation, x_scale*np.sin(rotation), y_scale*np.cos(rotation)), axis=1)
    
    A = np.concatenate((a1, a2), axis=0)
    zero = np.zeros((2, 4), np.float32)
    T = np.concatenate((A, zero), axis=1)
    T = np.expand_dims(T, axis=0)

    input_images_expanded = np.reshape(img, (1, CROP_SIZE, CROP_SIZE, 3, 1))

    t_img = TPS(input_images_expanded, grid, T, [CROP_SIZE, CROP_SIZE, 3])

    t_img = np.reshape(t_img, img.shape)

    return t_img, T

def TPS(U, source, T, out_size):
    
    
    def _repeat(x, n_repeats):
        rep = np.transpose(
            np.expand_dims(np.ones((n_repeats, )),axis=1), [1, 0])
        rep = rep.astype(np.int32)
        x = np.matmul(np.reshape(x, (-1, 1)), rep)
        return np.reshape(x, [-1])

    def _interpolate(im, x, y, out_size):
        # constants
        num_batch = im.shape[0]
        height = im.shape[1]
        width = im.shape[2]
        channels = im.shape[3]

        x = x.astype(np.float32)
        y = y.astype(np.float32)
        height_f = height*1.0
        width_f = width*1.0
        out_height = out_size[0]
        out_width = out_size[1]
        zero = np.zeros([], dtype=np.int32)
        max_y = (im.shape[1] - 1)
        max_x = (im.shape[2] - 1)

        # scale indices from [-1, 1] to [0, width/height]
        x = (x + 1.0) * (width_f) / 2.0
        y = (y + 1.0) * (height_f) / 2.0

        # do sampling
        x0 = np.floor(x).astype(np.int32)
        x1 = x0 + 1
        y0 = np.floor(y).astype(np.int32)
        y1 = y0 + 1

        x0 = np.clip(x0, zero, max_x)
        x1 = np.clip(x1, zero, max_x)
        y0 = np.clip(y0, zero, max_y)
        y1 = np.clip(y1, zero, max_y)
        dim2 = width
        dim1 = width * height
        base = _repeat(np.arange(num_batch) * dim1, out_height * out_width)
        base_y0 = base + y0 * dim2
        base_y1 = base + y1 * dim2
        idx_a = base_y0 + x0
        idx_b = base_y1 + x0
        idx_c = base_y0 + x1
        idx_d = base_y1 + x1

        # use indices to lookup pixels in the flat image and restore
        # channels dim
        im_flat = np.reshape(im, (-1, channels))
        im_flat = im_flat.astype(np.float32)
        Ia = np.take(im_flat, idx_a,axis=0)
        Ib = np.take(im_flat, idx_b,axis=0)
        Ic = np.take(im_flat, idx_c,axis=0)
        Id = np.take(im_flat, idx_d,axis=0)

        # and finally calculate interpolated values
        x0_f = x0.astype(np.float32)
        x1_f = x1.astype(np.float32)
        y0_f = y0.astype(np.float32)
        y1_f = y1.astype(np.float32)
        wa = np.expand_dims(((x1_f - x) * (y1_f - y)), 1)
        wb = np.expand_dims(((x1_f - x) * (y - y0_f)), 1)
        wc = np.expand_dims(((x - x0_f) * (y1_f - y)), 1)
        wd = np.expand_dims(((x - x0_f) * (y - y0_f)), 1)
        output = wa * Ia + wb * Ib +wc * Ic + wd * Id
        
        return output

    def _meshgrid(height, width, source):
        x_t = np.repeat(
            np.reshape(np.linspace(-1.0, 1.0, width), (1, width)), height, axis=0)
        y_t = np.repeat(
            np.reshape(np.linspace(-1.0, 1.0, height), (height, 1)), width,axis=1)

        x_t_flat = np.reshape(x_t, (1, 1, -1))
        y_t_flat = np.reshape(y_t, (1, 1, -1))

        num_batch = source.shape[0]
        px = np.expand_dims(source[:, :, 0], 2)  # [bn, pn, 1]
        py = np.expand_dims(source[:, :, 1], 2)  # [bn, pn, 1]
        d2 = np.square(x_t_flat - px) + np.square(y_t_flat - py)
        r = d2 * np.log(d2 + 1e-6)  # [bn, pn, h*w]
        x_t_flat_g = np.repeat(x_t_flat, num_batch, axis=0)  # [bn, 1, h*w]
        y_t_flat_g = np.repeat(y_t_flat, num_batch, axis=0)  # [bn, 1, h*w]
        ones = np.ones_like(x_t_flat_g)  # [bn, 1, h*w]

        grid = np.concatenate((ones, x_t_flat_g, y_t_flat_g, r), axis=1)  # [bn, 3+pn, h*w]
        return grid

    def _transform(T, source, input_dim, out_size):
        num_batch = input_dim.shape[0]
        height  = input_dim.shape[1]
        width   = input_dim.shape[2]
        num_channels = input_dim.shape[3]

        # grid of (x_t, y_t, 1), eq (1) in ref [1]
        height_f = height*1.0
        width_f  = width*1.0
        out_height = out_size[0]
        out_width  = out_size[1]
        grid = _meshgrid(out_height, out_width, source)  # [2, h*w]

         # transform A x (1, x_t, y_t, r1, r2, ..., rn) -> (x_s, y_s)
        # [bn, 2, pn+3] x [bn, pn+3, h*w] -> [bn, 2, h*w]
        T_g = np.matmul(T, grid)  #
        x_s = T_g[0:, 0:1, 0:]
        y_s = T_g[0: ,1, 0:]
        x_s_flat = np.reshape(x_s, [-1])
        y_s_flat = np.reshape(y_s, [-1])

        input_transformed = _interpolate(
            input_dim, x_s_flat, y_s_flat, out_size)

        output = np.reshape(
            input_transformed,(num_batch, out_height, out_width, num_channels))
        return output

    output = _transform(T, source, U, out_size)
    return output


def feature_warping(feature, deformation, padding=0):
    t_ = np.array([  # target position
        [-1., -1.],
        [1., -1.],
        [-1., 1.],
        [1., 1.],
    ])
    feature = np.pad(feature, [[0, 0], [padding, padding], [padding, padding], [0, 0]], mode = "constant", constant_values=0)
    CROP_SIZE = feature.shape[1]
    Batch_SIZE = feature.shape[0]
    DEPTH = feature.shape[3]

    grid = t_.reshape([1, 4, 2]).astype(np.float32)
    grid = np.repeat(grid, Batch_SIZE, axis=0)

    input_images_expanded = np.reshape(feature, (Batch_SIZE, CROP_SIZE, CROP_SIZE, DEPTH, 1))
    t_img = TPS(input_images_expanded, grid, deformation, [CROP_SIZE, CROP_SIZE, DEPTH])
    t_img = np.reshape(t_img, feature.shape)
    t_img = t_img[:, padding: CROP_SIZE-padding, padding: CROP_SIZE-padding,:]
    return t_img

if __name__=="__main__":
    import cv2
    import skimage
    import torch
    a    =  np.random.rand(1,100,100,3)
    im = cv2.resize(cv2.imread("000756.jpg"), (100, 100))
    im = skimage.img_as_float(im)
    im = im*2-1
    # j, _ =  image_warping2(a, w=0.0) 
    j, T =  image_warping2(im, w=0.1)

    index = torch.arange(0, 100).float()
    index = torch.reshape(index, (1, 100))

    #---------grid u----------------------------
    x1_index = index.unsqueeze(0).unsqueeze(0)
    x1_index = x1_index.repeat(1, 3, 100,1)

    x2_index = feature_warping(x1_index.permute(0,2,3,1).numpy(),T, padding=3)
    j = (j+1)/2
    cv2.imshow("img", j)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    
