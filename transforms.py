import numpy as np
import torch 
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode

class TransformHandler:

    def __init__(self, scale=120, flip=True, normalize=True):
        self.scale = scale
        self.flip = flip
        self.normalize = normalize

        train_transforms = [Normalize(self.normalize), ToTensor(), Resize(self.scale), RandomFlip(self.flip)]
        test_transforms = [Normalize(self.normalize), ToTensor(), Resize(self.scale)]
        self.train_transforms = transforms.Compose([*train_transforms])
        self.test_transforms = transforms.Compose([*test_transforms])
    
    def train_transform(self) -> transforms.Compose:
        return self.train_transforms
    
    def test_transform(self) -> transforms.Compose:
        return self.test_transforms
    
class Normalize:
    def __init__(self, normalize=True):
        self.normalize = normalize
        self.s1_means = np.array([-1773.50600126, -998.65863114])
        self.s1_std = np.array([279.79852461, 237.89666424])
        
        self.s2_means = np.array([ 531.75705347,
                                  820.82128237,
                                  791.62250193,
                                  2976.07215925,
                                  1229.98542459,
                                  2337.46271015,
                                  2716.73839343,
                                  2858.20823897,
                                  1953.6367865,
                                  1272.57999731])
        self.s2_std = np.array([217.01045106,
                                263.76853166,
                                388.4958599,
                                780.75329206,
                                351.65702763,
                                600.3173342,
                                748.78273836,
                                790.13533117,
                                548.49229615,
                                539.51143184])

        self.planet_means = np.array([453.6972422, 663.97140865, 716.74814954, 2879.03005035])
        self.planet_std = np.array([211.73414837, 272.10834959, 408.35074827, 896.13861989])



    def remove_nan_inf(self, image_stack, means):
        """
        replaces nan's and inf's with channel mean
        """
        image_stack = image_stack.transpose(1,0,2,3)
        
        nan_ind = np.where(np.isnan(image_stack))
        if len(nan_ind[0]) != 0 and means is not None:
            image_stack[nan_ind] = means[nan_ind[0]]
        
        inf_ind = np.where(np.isinf(image_stack))
        if len(inf_ind[0] != 0) and means is not None:
            image_stack[inf_ind] = means[inf_ind[0]]
        
        assert not np.isnan(image_stack).any()
        assert not np.isinf(image_stack).any()
        
        image_stack = image_stack.transpose(1,0,2,3)

        return image_stack
    
    def __call__(self, sample: dict):
        sample["s1_bands"] = self.remove_nan_inf(sample["s1_bands"], self.s1_means)
        sample["s2_bands"] = self.remove_nan_inf(sample["s2_bands"], self.s2_means)
        sample["p_bands"] = self.remove_nan_inf(sample["p_bands"], self.planet_means)
        
        if self.normalize:
            sample["s1_bands"] = ((sample["s1_bands"] - self.s1_means[None,:,None,None]) / self.s1_std[None,:,None,None]).astype(np.float32)
            sample["s2_bands"] = ((sample["s2_bands"] - self.s2_means[None,:,None,None]) / self.s2_std[None,:,None,None]).astype(np.float32)
            sample["p_bands"] = ((sample["p_bands"] - self.planet_means[None,:,None,None]) / self.planet_std[None,:,None,None]).astype(np.float32)

        return sample

class ToTensor:
    
    def __call__(self, sample: dict):
        sample["s1_bands"] = torch.tensor(sample["s1_bands"])
        sample["s2_bands"] = torch.tensor(sample["s2_bands"])
        sample["p_bands"] = torch.tensor(sample["p_bands"])
        sample["s_labels"] = torch.tensor(sample["s_labels"])
        sample["p_labels"] = torch.tensor(sample["p_labels"])

        return sample 


class Resize:

    def __init__(self, scale):
        self.labels_resize = transforms.Resize((scale, scale), interpolation=InterpolationMode.NEAREST)
        self.data_resize = transforms.Resize((scale, scale), interpolation=InterpolationMode.BILINEAR)

    def __call__(self, sample: dict):
        sample["s1_bands"] = self.data_resize(sample["s1_bands"])
        sample["s2_bands"] = self.data_resize(sample["s2_bands"])
        sample["p_bands"] = self.data_resize(sample["p_bands"])

        sample["p_labels"] = torch.squeeze(self.labels_resize(sample["p_labels"][None,:,:]), dim=0)
        sample["s_labels"] = torch.squeeze(self.labels_resize(sample["s_labels"][None,:,:]), dim=0)
        
        return sample

class RandomFlip:

    def __init__(self, flip):
        self.flip = flip

    def __call__(self, sample: dict):
        if self.flip:
            p = np.random.choice([0,1], 2)
            flip_transform = transforms.Compose([transforms.RandomVerticalFlip(p[0]), transforms.RandomHorizontalFlip(p[1])])

            sample["s1_bands"] = flip_transform(sample["s1_bands"])
            sample["s2_bands"] = flip_transform(sample["s2_bands"])
            sample["p_bands"] = flip_transform(sample["p_bands"])
            sample["s_labels"] = flip_transform(sample["s_labels"])
            sample["p_labels"] = flip_transform(sample["p_labels"])

        return sample

class RasterCrop:
    
    """
    splits image into non overlapping patches of size crop_size x crop_size
    if image extent is not divisible by crop_size, the remaining pixels are discarded
    """
    def __init__(self, crop_size):
        self.crop_size = crop_size # eg 24 for TSViT

    def crop(self, image_stack):
        x_dim, y_dim = image_stack.shape[-2:] # vertical and horizontal extent
        num_rows = int(x_dim/self.crop_size)
        num_cols = int(y_dim/self.crop_size)

        cropped_images = []
        for i in range(num_rows):
            for j in range(num_cols):
                cropped_images += [transforms.functional.crop(image_stack, top=i*self.crop_size,left=j*self.crop_size,height=self.crop_size,width=self.crop_size)]
        cropped_images = torch.stack(cropped_images)
        return cropped_images


    def __call__(self, sample: dict):
        sample["s1_bands"] = self.crop(sample["s1_bands"]) # adds another dimensionality: "patchbatch" = P (TxCxHxW -> PxTxCxHxW)
        sample["s2_bands"] = self.crop(sample["s2_bands"])
        sample["p_bands"] = self.crop(sample["p_bands"]) 
        sample["s_labels"] = self.crop(sample["s_labels"])
        sample["p_labels"] = self.crop(sample["p_labels"])

        return sample


def filter_and_pad_timepoints(image_stack, timestamps, selected_time_points):
    """
    @param image_stack: original imagestack of the modality in shape TxCxHxW (numpy array)
    @param timestamps: timestamps dictionary that contains for each temporal index T of an image in the image_stack the day in the year at which it has been acquired
    @param selected_time_points: list of time points (day in year) that we want to keep

    @returns: padded image stack with an image for each selected_time_point if available in the modality. 0 image otherwise
    """
    _, C, H, W = image_stack.shape

    # create a padded image stack where time points for which there is no image in the intial image_stack will be set to 0-images
    padded_stack = np.zeros((len(selected_time_points), C, H, W))
    
    # find the lines in the dataframe that correspond to the selected time points for which there is an image in the modality
    common_timepoints = timestamps.loc[timestamps['timestamp'].isin(selected_time_points)]
    # store the indices of these common time points in the original image_stack
    common_timepoints_indices = common_timepoints.index.values
    # store the actual common time points
    common_timepoints = np.array(common_timepoints.timestamp.values)

    # find the indices of the common time points in padded stack -> the indices where the images corresponding to the common time points will be inserted
    new_indices = [i for i,j in enumerate(selected_time_points) if j in common_timepoints]
    
    # insert the images for the common time points at the correct indices in the padded stack
    padded_stack[new_indices] = image_stack[common_timepoints_indices]
    
    return padded_stack
