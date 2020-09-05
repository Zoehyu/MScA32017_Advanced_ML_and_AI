import math
import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff

from train_unet import weights_path, get_model, normalize, PATCH_SZ, N_CLASSES

# added a shift variable for width and height
def predict(x, model, patch_sz=160, n_classes=5, shift_h=0, shift_w=0):
    img_height = x.shape[0]
    img_width = x.shape[1]
    n_channels = x.shape[2]
    # make extended img so that it contains integer number of patches
    npatches_vertical = math.ceil(img_height / patch_sz)
    npatches_horizontal = math.ceil(img_width / patch_sz)
    extended_height = patch_sz * npatches_vertical
    extended_width = patch_sz * npatches_horizontal
    ext_x = np.zeros(shape=(extended_height, extended_width, n_channels), dtype=np.float32)
    # fill extended image with mirrors:
    ext_x[shift_h:img_height+shift_h, shift_w:img_width+shift_w, :] = x
    for i in range(img_height + shift_h, extended_height):
        ext_x[i, :, :] = ext_x[2 * (img_height + shift_h) - i - 1, :, :]
    for i in range(0, shift_h):
        ext_x[i, :, :] = ext_x[2 * shift_h - i - 1, :, :]
    for j in range(img_width + shift_w, extended_width):
        ext_x[:, j, :] = ext_x[:, 2 * (img_width + shift_w) - j - 1, :]
    for j in range(0, shift_w):
        ext_x[:, j, :] = ext_x[:, 2 * shift_w - j - 1, :]

    # now we assemble all patches in one array
    patches_list = []
    for i in range(0, npatches_vertical):
        for j in range(0, npatches_horizontal):
            x0, x1 = i * patch_sz, (i + 1) * patch_sz
            y0, y1 = j * patch_sz, (j + 1) * patch_sz
            patches_list.append(ext_x[x0:x1, y0:y1, :])
    # model.predict() needs numpy array rather than a list
    patches_array = np.asarray(patches_list)
    # predictions:
    patches_predict = model.predict(patches_array, batch_size=4)
    prediction = np.zeros(shape=(extended_height, extended_width, n_classes), dtype=np.float32)
    for k in range(patches_predict.shape[0]):
        i = k // npatches_horizontal
        j = k % npatches_vertical
        x0, x1 = i * patch_sz, (i + 1) * patch_sz
        y0, y1 = j * patch_sz, (j + 1) * patch_sz
        prediction[x0:x1, y0:y1, :] = patches_predict[k, :, :, :]
    return prediction[shift_h:img_height+shift_h, shift_w:img_width+shift_w, :]


def picture_from_mask(mask, threshold=0):
    colors = {
        0: [150, 150, 150],  # Buildings
        1: [223, 194, 125],  # Roads & Tracks
        2: [27, 120, 55],    # Trees
        3: [166, 219, 160],  # Crops
        4: [116, 173, 209]   # Water
    }
    z_order = {
        1: 3,
        2: 4,
        3: 0,
        4: 1,
        5: 2
    }
    pict = 255*np.ones(shape=(3, mask.shape[1], mask.shape[2]), dtype=np.uint8)
    for i in range(1, 6):
        cl = z_order[i]
        for ch in range(3):
            pict[ch,:,:][mask[cl,:,:] > threshold] = colors[cl][ch]
    return pict


if __name__ == '__main__':
    # for i in range(5):
    model = get_model()
    model.load_weights(weights_path)
    test_id = 'test'
    img = normalize(tiff.imread('data/mband/{}.tif'.format(test_id)).transpose([1,2,0]))   # make channels last
    # test set augmentation - average predictions from different shifts
    for shift_h in range(0, int(PATCH_SZ/2)+1, 10):
        for shift_w in range(0, int(PATCH_SZ/2)+1, 10):
            mask0 = predict(img, model, patch_sz=PATCH_SZ, n_classes=N_CLASSES, shift_h=shift_h, shift_w=shift_w).transpose([2,0,1])  # make channels first
            if shift_h == shift_w == 0:
                mask_stack = np.expand_dims(mask0, axis=0)
            else:
                mask_stack = np.concatenate((mask_stack, np.expand_dims(mask0, axis=0)), axis=0)
        # # model ensembling
        # if i == 0:
        #     mask = np.mean(mask_stack, axis=0, keepdims=True)
        # else:
        #     mask = np.concatenate((mask, np.mean(mask_stack, axis=0, keepdims=True)), axis=0)
    mask = np.mean(mask_stack, axis=0)
    # mask = np.mean(mask, axis=0)
    map = picture_from_mask(mask, 0.5)

    tiff.imsave('result.tif', (255*mask).astype('uint8'))
    tiff.imsave('map.tif', map)
