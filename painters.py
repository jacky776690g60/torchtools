'''
contains utility functions/classes related to plotting/processing images
'''
import random, math, warnings
from typing import *
from pathlib import Path

from PIL import Image

import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision
from torchvision.utils import save_image
from torchvision.transforms import functional as TF

from .utility import *

__all__ = (
    'DaVinci', 'Mondrian',
)

# =============================================================================
# Classes
# =============================================================================
class __Painters():
    "Base painter class"
    
    def __init__(self) -> None:
        pass
    
    @staticmethod
    def is_tensor_correct(
        img_tensor: TorchTensorType, 
        mode:       Literal['RGB'] | Literal['RGBA'] | Literal['grey'] = 'RGB'
    ) -> bool:
        """
        Check if the argument is a PyTorch tensor and has the correct shape 
        `(3 * H * W)` for RGB or `(4 * H * W)` for RGBA.

        Params:
        -------
        - img_tensor: The image tensor to be checked.
        - mode: The color mode of the tensor, either 'RGB' or 'RGBA'.
        
        Raises:
        -------
        - If not correct tensor, it will raise error.
        """
        if not isinstance(img_tensor, torch.Tensor):
            raise TypeError("img_tensor must be a torch.Tensor")

        valid_channels = {'RGB': 3, 'RGBA': 4, 'grey': 1}
        if mode not in valid_channels:
            raise ValueError(f"Invalid mode {mode}. Expected 'RGB', 'RGBA', or 'grey'.")

        expected_channels = valid_channels[mode]
        if img_tensor.dim() != 3 or img_tensor.shape[0] != expected_channels:
            raise ValueError(f'Expected img tensor to have shape ({expected_channels}, H, W) for {mode} mode.')
    
        return True
    
    
    @classmethod
    def save_tensor_to_img(cls,
        img_tensor: TorchTensorType,
        save_path:  str,
    ) -> None:
        '''
        Save an image tensor to a file path
        '''
        cls.is_tensor_correct(img_tensor)
        copied_tensor = img_tensor.clone().detach()
        
        if copied_tensor.max() > 1.0:               # Normalize the tensor [0, 1]
            copied_tensor = copied_tensor / 255.0
            
        save_image(img_tensor, save_path)           # Save the tensor as an image


    @staticmethod
    def pil_to_tensor(
        img:           PILImageType, 
        color_mode:    str                     = "RGB",
        new_size:      Tuple[int, int]         = None,
        resampling:    PILImageResamplingType  = Image.Resampling.LANCZOS
    ) -> TorchTensorType:
        '''
        Convert the color space of the PIL Image and resample it based on size.
        
        Params:
        -------
        - img: An image opened with PIL.
        - color_mode: Desired color mode e.g., 'RGB', 'L' for grayscale, etc.
        - size: Desired output size as (width, height).
        - resampling: Resampling method for resizing.
            - `NEAREST`, `BOX`, `BILINEAR`, `HAMMING`, `BICUBIC`, `LANCZOS`
        
        Return:
        -------
        - image tensor
        '''    
        if img.mode != color_mode: img = img.convert(color_mode) # Convert image to the desired color mode
        if new_size: img = img.resize(new_size, resampling)      # Resize the image if size is provided
        return TF.to_tensor(img)                                 # Convert to tensor





class DaVinci(__Painters):
    '''
    A utility class for plotting realistic visualizations.

    Provides methods tailored for displaying images, photos.
    '''
    def __init__(self) -> None:
        super().__init__()
    
    @staticmethod
    def plot_random_imgs(
        dataset:  TorchDatasetType, 
        n:        int                = 3, 
        seed:     int                = None, 
        figsize:  Tuple[int, int]    = (16, 8),
        verbose:  bool               = True,
    ) -> PltFigureType:
        '''
        Display random images from a torch dataset using matplotlib.
        
        Params:
        -------
        - `dataset`: Dataset containing images to display.
        - `n`: Number of random images to display. Default is 10. 
            * If n > 7, it's set to 7 for display purposes.
        - `seed`: Seed for reproducibility.
        - `figsize`: Size of the entire figure.
        - `verbose`: extra debug details.

        Returns:
        --------
        - plt.Figure
        '''
        max_display= 7
        assert n <= max_display, f"For display purpose, max n is {max_display}..."
        n = min(n, max_display)

        if seed: random.seed(seed)

        if verbose: print(f"Dataset Total Length: {len(dataset)}")
        rand_indices = random.sample(range(len(dataset)), k=n)

        fig, axes = plt.subplots(1, n, figsize=figsize)

        for ax, img_idx in zip(axes, rand_indices):
            img, label = dataset[img_idx][:2]                         # Assumes dataset returns (image, label)
            img_display: npNDArrayType = img.permute(1, 2, 0).numpy() # Adjust for matplotlib

            title = f"Class: {label}\nShape: {img_display.shape}"
            ax.set_title(title, fontdict=FONT_SUBTITLE_MD)
            ax.imshow(img_display)
            ax.axis("off")

        fig.tight_layout()
        plt.close()
        return fig
    
    
    
    @classmethod
    def plot_img_tensor(cls,
        img_tensor:   TorchTensorType, 
        title:        str                 = "A Tensor", 
        figsize:      tuple[float, float] = (4, 3),
        mode:         str                 = 'RGB',
    ) -> PltFigureType:
        '''
        Plot a single image from a tensor.

        Params:
        -------
        - `img`: A tensor representation of the image. Shape should be `(C, H, W)`
        - `title`: Title for the image plot.
        - `figsize`: Size of the figure.

        Returns:
        -------
        - `plt.Figure`: A matplotlib Figure object.
        '''
        cls.is_tensor_correct(img_tensor, mode=mode)
        
        fig, ax = plt.subplots(figsize=figsize)
        if img_tensor.shape[2] == 1: # grayscale
            ax.imshow(img_tensor.squeeze(0).numpy(), cmap='gray')
        else: # RGB: rearrange for matplotlib (H, W, C)
            ax.imshow(img_tensor.permute(1, 2, 0).numpy())
        
        ax.set_title(f"{title}\nshape: {img_tensor.shape}")
        ax.axis(False)
        plt.close()
        return fig
    
    
    @classmethod
    def plot_transformed_from_imgpath(cls,
        image_paths: List[str], 
        transform:   TorchTransformsComposeType, 
        figsize:     tuple[float, float]         = (6, 3),
    ) -> List[PltFigureType]:
        '''
        Loads images from paths, transforms them, and then plots the original 
        against the transformed version.

        Params:
        -------
        - `image_paths`: List of image paths.
        - `transform`: Torch transform function (Compose or Sequential).
        - `figsize`: Size for each figure.

        Returns:
        -------
        - List of matplotlib Figure objects.
        
        Examples:
        ---------
        >>> painters.DaVinci.transform_and_plot_imgs(
        ...     dataset.paths[:3], transform, nrows=1, ncols=2
        ... )
        '''
        figures = []

        for path in image_paths:
            file_name = Path(path).name
            with Image.open(path) as img:
                fig, ax = plt.subplots(nrows=1, ncols=2, figsize=figsize)
                # Original image
                ax[0].imshow(img)
                ax[0].set_title(f"Original\nSize: {img.size}", fontdict=FONT_SUBTITLE_SM)
                ax[0].axis(False)
                # Transformed image
                transformed_image = transform(img) # a copied
                if isinstance(transformed_image, torch.Tensor):
                    transformed_image = transformed_image.permute(1, 2, 0) # Change shape for matplotlib: (C, H, W) -> (H, W, C)
                ax[1].imshow(transformed_image)
                ax[1].set_title(f"Transformed\nShape: {transformed_image.shape}", fontdict=FONT_SUBTITLE_SM)
                ax[1].axis(False)

                fig.suptitle(f"File: {file_name}", fontsize=12)
                # fig.tight_layout()
                figures.append(fig)
        return figures
                
                
                
    @classmethod
    def plot_transformed_from_imgpath_progressive(cls,
        image_path:     str, 
        transform:      TorchTransformsComposeType, 
        figsize:        Tuple[float, float]         = (4, 8),
    ) -> List[PltFigureType]:
        '''
        Load image from path, transform and plot them progressively

        Params:
        -------
        - `image_path`: path to image
        - `transform`: Torch transform function (Compose or Sequential).
        - `figsize`: Size for figure.

        Returns:
        -------
        - List of matplotlib Figure objects.
        
        Examples:
        ---------
        >>> painters.DaVinci.transform_and_plot_imgs(
        ...     dataset.paths[:3], transform, nrows=1, ncols=2
        ... )
        '''
        # figures = []
        
        TOTAL_TRANSFORM = len(transform.transforms)
        
        file_name = Path(image_path).name
        with Image.open(image_path) as img:
            fig, ax = plt.subplots(nrows=TOTAL_TRANSFORM+1, ncols=1, figsize=figsize)
            
            # Original image
            ax[0,].imshow(img)
            ax[0,].set_title(f"Original\nSize: {img.size}", fontdict=FONT_SUBTITLE_SM)
            ax[0,].axis(False)
            
            transformed = img
            
            for i, tf in enumerate(transform.transforms):
                tmp = transformed = tf(transformed)
                
                if isinstance(transformed, torch.Tensor):
                    tmp = transformed.permute(1, 2, 0)
                
                ax[i+1,].imshow(tmp)
                ax[i+1,].set_title(f"{tf}", fontdict=FONT_SUBTITLE_SM)
                ax[i+1,].axis(False)
            
            fig.suptitle(f"File: {file_name}", fontsize=12)
            fig.tight_layout()
            plt.close()
            return fig
                
                

    


    @staticmethod
    def plot_pred_img(
        model:           TorchNNModuleType, 
        target_img_path: str,
        class_names:     List[str]                   = None, 
        transform:       TorchTransformsComposeType  = None, 
        device:          str                         = DEVICE
    ) -> PltFigureType:
        '''
        Make a prediction on a target image with a trained model and 
        plot the image along with the prediction.
        '''
        # converted to the range [0, 1]. Common for NN training to stay in range
        target_img_tensor = torchvision.io.read_image(str(target_img_path)).type(torch.float32) / 255.0

        if transform: target_img_tensor = transform(target_img_tensor)

        # Move model to device and set it to evaluation mode
        model.to(device).eval()
        
        with torch.inference_mode():
            # Add batch dimension and move image to the target device
            target_img_tensor = target_img_tensor.unsqueeze(0).to(device)
            # Make a prediction
            z = model(target_img_tensor) # logits
            # logits -> probabilities
            probs = torch.softmax(z, dim=1)
            # probabilities -> labels
            y_hat = torch.argmax(probs, dim=1)

        fig = plt.figure()
        plt.imshow(target_img_tensor.cpu().squeeze().permute(1, 2, 0))
        probs_str = f"{probs.max().item():.3f}"
        title = f"Pred: {class_names[y_hat.item()]} | Prob: {probs_str}" if class_names else f"Pred: {y_hat.item()} | Prob: {probs_str}"
        plt.title(title)
        plt.axis(False)
        return fig


    @classmethod
    def plot_patchified_img(cls,
        img_tensor:   torch.Tensor, 
        patch_h:      int,
        patch_w:      int,
        title:        str               = "",
        figsize:      Tuple[int, int]   = (8,6),
        verbose:      bool              = False
    ) -> PltFigureType:
        '''
        plot a patchified version of the image.
        
        Raises:
        -------
        - `AssertionError`: patch size cannot divide the entire image
        
        ```
        Original 6x6 Image:
        ---------------------
        | 1 | 2 | 3 | 1 | 2 | 3 |
        |---|---|---|---|---|---|
        | 4 | 5 | 6 | 4 | 5 | 6 |
        |---|---|---|---|---|---|
        | 7 | 8 | 9 | 7 | 8 | 9 |
        |---|---|---|---|---|---|
        | 1 | 2 | 3 | 1 | 2 | 3 |
        |---|---|---|---|---|---|
        | 4 | 5 | 6 | 4 | 5 | 6 |
        |---|---|---|---|---|---|
        | 7 | 8 | 9 | 7 | 8 | 9 |
        ---------------------

        Patchified into 3x3 Patches:
        ---------------------
        Patch 1       Patch 2
        | 1 | 2 | 3 |  | 1 | 2 | 3 |
        |---|---|---|  |---|---|---|
        | 4 | 5 | 6 |  | 4 | 5 | 6 |
        |---|---|---|  |---|---|---|
        | 7 | 8 | 9 |  | 7 | 8 | 9 |
        Patch 3       Patch 4
        | 1 | 2 | 3 |  | 1 | 2 | 3 |
        |---|---|---|  |---|---|---|
        | 4 | 5 | 6 |  | 4 | 5 | 6 |
        |---|---|---|  |---|---|---|
        | 7 | 8 | 9 |  | 7 | 8 | 9 |

        ```
        '''
        cls.is_tensor_correct(img_tensor)
        img_tensor = img_tensor.permute(1, 2, 0)
        
        imgH, imgW = img_tensor.shape[0:2] # Corrected the shape extraction
        assert imgH % patch_h == 0, f"patch height cannot divide image height. {imgH} % {patch_h} != 0"
        assert imgW % patch_w == 0, f"patch width cannot divide image width. {imgW} % {patch_w} != 0"
        
        p_perCol, p_perRow = int(imgH // patch_h), int(imgW // patch_w)
        p_cnt = int(p_perCol * p_perRow)

        if verbose:
            print(f"Rows: {p_perRow} | Column: {p_perCol}")
            print(f"Size (w * h): {patch_w} * {patch_h}")
            print(f"Total patches: {p_cnt}")

        # Create a series of subplots
        fig, axs = plt.subplots(
                        nrows=p_perCol,
                        ncols=p_perRow, 
                        figsize=figsize,
                        sharex=True,
                        sharey=True
                    )
        
        # To ensure compatibility with different subplot structures (1D vs 2D array)
        axs = np.array(axs) if isinstance(axs, (list, np.ndarray)) else np.array([[axs]])

        # Loop through height and width
        for i, y in enumerate(range(0, imgH, patch_h)):
            for j, x in enumerate(range(0, imgW, patch_w)):
                axs[i, j].imshow( img_tensor[y:y+patch_h, x:x+patch_w, :] )
                axs[i, j].set_ylabel(
                    i+1, 
                    rotation="horizontal", 
                    horizontalalignment="right", 
                    verticalalignment="center"
                )
                axs[i, j].set_xlabel(j+1)
                axs[i, j].set_xticks([])
                axs[i, j].set_yticks([])
                axs[i, j].label_outer()
        fig.suptitle(f"Patchified {title}\npatch_size: {patch_h} * {patch_w}", fontsize=9)
        plt.close(fig)
        return fig
    
    
    
    
    @staticmethod
    def plot_batch_samples(
        data_loader:        TorchDataLoaderType, 
        batch_number:       int                  = 1,
        num_images:         int                  = 4,
    ) -> PltFigureType:
        '''
        Display a batch of images from the DataLoader.

        Params:
        -------
        - `data_loader`: DataLoader from which to fetch the images.
        - `batch_number`: batch number
        - `num_images`: Number of images to display. 

        Returns:
        -------
        - A Matplotlib Figure object with the plotted images.
        '''
        dataloader_iter = iter(data_loader)
        target_batch = None
        
        for _ in range(batch_number):
            target_batch = next(dataloader_iter)
        
        images, labels, image_paths = target_batch
        
        num_rows = math.ceil(num_images / 5)
        num_cols = 5 if num_images > 5 else num_images

        fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(12, 12))

        for i in range(num_images):
            axs[i].imshow(images[i].permute(1, 2, 0))  # Convert the tensor format to image format
            axs[i].set_title(
                f'{labels[i]}\n'
                f'{Path(image_paths[i]).name}'
            , fontsize=9)
            axs[i].axis('off')

        # Hide any unused subplots
        # for i in range(num_images, num_rows * num_cols):
        #     axs[i].axis('off')

        fig.suptitle(f"Batch {batch_number}")
        fig.subplots_adjust(hspace=0.05)  # Adjust vertical spacing between rows
        fig.tight_layout(pad=1.0)

        plt.close(fig)
        return fig
        
        
        
        
class Mondrian(__Painters):
    '''
    A class for painting abstract and statistical visualizations.

    #NOTE
    -----
    Useful for visualizing curves, data trends, and abstract charts.
    '''
    
    @staticmethod
    def plot_training_curves(
        results: Dict[str, List[float]],
        figsize: Tuple[int, int]        = (10, 4)
    ) -> PltFigureType:
        '''
        Plots training curves from the results dictionary.

        Params:
        -------
        - results: A dictionary containing training metrics. Must contain the keys:
            * train_loss
            * train_acc
            * test_loss
            * test_acc
        '''
        required_keys = ["train_loss", "test_loss", "train_acc", "test_acc"]
        if not all(key in results for key in required_keys):
            raise ValueError(f"Dictionary missing one or more required keys. Must contains keys: {required_keys}")
        
        train_loss = results["train_loss"]
        train_acc  = results["train_acc"]
        test_loss  = results["test_loss"]
        test_acc   = results["test_acc"]

        if not (len(train_loss) == len(train_acc) == len(test_loss) == len(test_acc) != 0):
            raise ValueError("All lists in dictionary must have the same length and cannot be empty.")

        # Total epochs
        epochs = range(len(train_loss))

        # Setup a plot
        fig = plt.figure(figsize=figsize)
        # Plot the loss
        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_loss, label="train_loss")
        plt.plot(epochs, test_loss, label="test_loss")
        plt.title("Loss")
        plt.xlabel("Epochs")
        plt.legend()

        # Plot the accuracy
        plt.subplot(1, 2, 2)
        plt.plot(epochs, train_acc, label="train_accuracy")
        plt.plot(epochs, test_acc, label="test_accuracy")
        plt.title("Accuracy")
        plt.xlabel("Epochs")
        plt.legend()

        plt.tight_layout()
        plt.close()
        return fig
    
    
    
    
    @staticmethod
    def plot_feature_maps_from_conv(
        feature_maps:   TorchTensorType,
        num_maps:       int              = 4,
        cmap:           str              = "Blues",
    ) -> PltFigureType:
        '''
        Plots the feature maps from a convolutional layer.

        Params:
        -------
        - `feature_maps`: A tensor containing the feature maps.
        - `num_maps`: Number of feature maps to plot.
        '''
        # Ensure we don't try to plot more maps than available
        if num_maps > (available:= feature_maps.shape[1]):
            warnings.warn("Attempted to plot more maps than available. Plotting max available maps...")
        num_maps = min(num_maps, available)
        
        fig, axes = plt.subplots(1, num_maps, figsize=(num_maps * 3, 3))
        fig.suptitle(f"Shape: {feature_maps.shape}", fontsize=12)
        
        for i in range(num_maps):
            ax = axes[i]
            # Extracting the feature map and converting it to numpy for plotting
            feature_map = feature_maps[0, i].cpu().detach().numpy()
            ax.imshow(feature_map, cmap=cmap)
            ax.axis('off')
            ax.set_title(f'Map {i+1}', fontdict=FONT_SUBTITLE_SM)
        return fig