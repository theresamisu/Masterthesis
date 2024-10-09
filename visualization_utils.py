from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np

import torch
import torchvision.transforms.functional as F
from torchvision.utils import draw_segmentation_masks

#segmentation_palette = sns.color_palette(colorcet.glasbey, n_colors=14).as_hex()
segmentation_palette = sns.color_palette("tab20").as_hex()

# Define NDVI index for S2 images
def ndvi(X):
    red = X[3]
    nir = X[7]#
    return (nir-red) / (nir + red)

#Define True Color for S2 images
def true_color(X):
    blue = X[1]/(X[1].max()/255.0)
    green = X[2]/(X[2].max()/255.0)
    red = X[3]/(X[3].max()/255.0)
    tc = np.dstack((red,green,blue)) 
    
    return tc.astype('uint8')

#Define NDVI index for Planet Fusion images
def ndvi_planet(X):
    red = X[2]
    nir = X[3]
    return (nir-red) / (nir + red)

#Define True Color for Planet Fusion images
def true_color_planet(X):
    blue = X[0]/(X[0].max()/255.0)
    green = X[1]/(X[1].max()/255.0)
    red = X[2]/(X[2].max()/255.0)
    tc = np.dstack((red,green,blue)) 
    return tc.astype('uint8')

def rvi(X):
    VV = X[0]
    VH = X[1]
    dop = (VV/(VV+VH))
    m = 1 - dop
    radar_vegetation_index = (np.sqrt(dop))*((4*(VH))/(VV+VH))
    
    return radar_vegetation_index

# inspired https://pytorch.org/vision/stable/auto_examples/plot_visualization_utils.html
def plot_and_save(imgs, fpath):
    if not isinstance(imgs, list):
        plt.clf()
        imgs = [imgs]
        fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
        for i, img in enumerate(imgs):
            img = F.to_pil_image(img)
            axs[0, i].imshow(np.asarray(img))
            #axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    
        plt.savefig(fpath)

def mn_mx_scaler(tensor, axis):
    return (tensor - tensor.min(axis=axis, keepdims=True)) / (tensor.max(axis=axis, keepdims=True) - tensor.min(axis=axis, keepdims=True))

def visualize_example_with_prediction(input_ts, ground_truth, predicted_segmentation, batch_idx, epoch, subset, name, modalities, num_classes):
    viz_dir=Path(f'output_eoko/viz/{name}/{subset}/epoch_{epoch}/batch_idx_{batch_idx}')
    viz_dir.mkdir(exist_ok=True, parents=True)

    input_ts_np = input_ts.cpu().numpy()

    # TxCxHxW
    if modalities=="P":
        inputs_ts_np_bgrnir = input_ts_np[:,:,:,:]
    elif modalities=="ALL": # ALL
        inputs_ts_np_bgrnir = input_ts_np[:,-5:-1,:,:]
    elif modalities == "S2":
        inputs_ts_np_bgrnir = input_ts_np[:,0:4:,:,:]
    elif modalities == "S1":
        inputs_ts_np_vvvh = input_ts_np[:,0:2,:,:]
    else:
        inputs_ts_np_bgrnir = input_ts_np[:,-5:-1, :, :]
    
    if modalities != "S1":
        # save rgb mean over time -> base image
        # input_ts_np.shape should be (C, T, W, H), channel order of first three channels is BGR
        inputs_ts_np_bgrnir = inputs_ts_np_bgrnir.transpose(1,0,2,3)
        base_img = mn_mx_scaler(inputs_ts_np_bgrnir[0:3][::-1].mean(1), axis=(1, 2))
        # rgb_temporal_mean = mn_mx_scaler(input_ts_np[0:3, :, :, :][::-1], axis=1).mean(1)
        plot_and_save(
                (255*base_img).astype(np.uint8).swapaxes(0, 2).swapaxes(1, 0), # channel axis needs to be last for plotting
                viz_dir / "rgb_temporal_mean.png",
                )
        # save ndvi time series per pixel
        ndvi_tensor = mn_mx_scaler((inputs_ts_np_bgrnir[3] - inputs_ts_np_bgrnir[2]) / (inputs_ts_np_bgrnir[3] + inputs_ts_np_bgrnir[2]), axis=None)
        ndvi_spatial_mean = ndvi_tensor.mean(1).mean(1)

        plt.clf()
        plt.plot(ndvi_spatial_mean)
        plt.savefig(viz_dir / 'ndvi_spatial_mean.png')
    else:
        # to CxTxHxW
        inputs_ts_np_vvvh = inputs_ts_np_vvvh.transpose(1,0,2,3)
        base_img = mn_mx_scaler(inputs_ts_np_vvvh.mean(1), axis=(1,2))
        # image needs to have three channels for draw_segmentation_mask function -> convert to rgb
        base_img = rvi(base_img)
        im = plt.imshow(base_img)
        plt.close()
        base_img = im.cmap(im.norm(base_img))[:,:,:3].swapaxes(1, 2).swapaxes(1, 0) #should be 3x24x24
        plot_and_save(
                (255*base_img).astype(np.uint8).swapaxes(0, 2).swapaxes(1, 0),
                viz_dir / "rvi_temporal_mean.png",
                )

    # save ground truth segmentation
    gt_one_hot = torch.nn.functional.one_hot(ground_truth, num_classes).to(torch.bool)
    plot_and_save(draw_segmentation_masks(image=torch.Tensor(255*base_img).to(torch.uint8), 
        masks=gt_one_hot.swapaxes(0, 2).swapaxes(1, 2), 
        alpha=1.0, 
        colors=segmentation_palette
        ),
        viz_dir / 'ground_truth_segmentation.png'
    )

    plot_and_save(draw_segmentation_masks(image=torch.Tensor(255*base_img).to(torch.uint8), 
        masks=gt_one_hot.swapaxes(0, 2).swapaxes(1, 2), 
        alpha=.3, 
        colors=segmentation_palette
        ),
        viz_dir / 'ground_truth_segmentation_alpha_0_3.png'
    )


    # save predicted segmentation
    pred_one_hot = torch.nn.functional.one_hot(predicted_segmentation.to(torch.int64), num_classes).to(torch.bool)
    plot_and_save(
        draw_segmentation_masks(image=torch.Tensor(255*base_img).to(torch.uint8), 
        masks=pred_one_hot.swapaxes(0, 2).swapaxes(1, 2), 
        alpha=1.0, 
        colors=segmentation_palette
        ),
        viz_dir / 'predicted_segmentation.png'
    )

    plot_and_save(draw_segmentation_masks(image=torch.Tensor(255*base_img).to(torch.uint8), 
        masks=pred_one_hot.swapaxes(0, 2).swapaxes(1, 2), 
        alpha=.3, 
        colors=segmentation_palette
        ),
        viz_dir / 'predicted_segmentation_alpha_0_3.png'
    )

    binary_map = predicted_segmentation == ground_truth
    bin_one_hot = torch.nn.functional.one_hot(binary_map.to(torch.int64), 2).to(torch.bool)
    plot_and_save(
        draw_segmentation_masks(image=torch.Tensor(255*base_img).to(torch.uint8), 
        masks=bin_one_hot.swapaxes(0, 2).swapaxes(1, 2), # CxHxW -> HxWxC
        alpha=1.0, 
        colors=["red", "green"]
        ),
        viz_dir / 'classification_errors.png'
    )

    plot_and_save(
        draw_segmentation_masks(image=torch.Tensor(255*base_img).to(torch.uint8), 
        masks=bin_one_hot.swapaxes(0, 2).swapaxes(1, 2), 
        alpha=.3, 
        colors=["red", "green"]
        ),
        viz_dir / 'classification_error_alpha_0_3.png'
    )



    return
