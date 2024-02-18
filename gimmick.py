"""
This file contains functions that helps speed up model training/testing, etc.
"""
import os, sys, random
from typing import *
from pathlib import Path

import torch
from torch.utils.data import random_split, DataLoader

from .utility import *
from .imagefolder import ImageFolder


class Optics():
    """
    This class contains helper functions for dealing with image-related 
    model processes.
    """
    
    @staticmethod
    def load_presplit_data(
        root_path: str,
        batch_size: int,
        transform: TorchTransformsComposeType, 
        img_extension=".jpeg",
        check_consistency=True,
        num_workers: int = os.cpu_count()
    ) -> tuple[TorchDataLoaderType, TorchDataLoaderType, List[str], Dict[str, int]]:
        """
        Load a dataset that is already split into `train_set` and `test_set`

        Params:
        -------
        - `root_path`: root folder which contains a `train/` and a `test/`
        - `batch_size`: Number of samples per batch in each of the DataLoaders
        - `transform`: torchvision transforms to perform on training and testing data
        - `img_extension`: `.jpeg`, `.png`, ...
        - `check_consistency`: check if all images have same extension
        - `num_workers`: An integer for number of workers per DataLoader

        Returns:
        -------
        tuple[DataLoader, DataLoader, List[str], Dict[str, int]]
            * train_dataloader : DataLoader for training data.
            * test_dataloader : DataLoader for testing data.
            * class_names : List of class names.
            * class_to_idx : Dictionary mapping class names to their respective indices.
             
        Folder Structure:
        -----------------
        ```
        root_dir/
            |__train/
                |__Cat/
                    |_ image1.jpg
                    |_ ...
                |__Dog/
                    |_ image1.jpg
                    |_ ...
            |__test/
                |__Cat/
                    |_ image1.jpg
                    |_ ...
                ...
        ```

        """
        if not all( required in os.listdir(root_path) for required in ["train", "test"]):
            raise ValueError("Root folder must contains train/ and test/ folders.")
        
        root_path = Path(root_path)
        train_dir, test_dir = root_path / "train", root_path / "test"
        
        train_data = ImageFolder(
            img_dir=train_dir, img_ext=img_extension, 
            transform=transform, check_consistency=check_consistency)
        test_data  = ImageFolder(
            img_dir=test_dir, img_ext=img_extension, 
            transform=transform, check_consistency=check_consistency)

        if train_data.classes != test_data.classes:
            raise ValueError("Folders incorrect/mismatch between train and test classes!")
        

        train_dataloader = DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True, # enables faster data transfer to CUDA-enabled GPUs
        )
        test_dataloader = DataLoader(
            test_data,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
        return train_dataloader, test_dataloader, train_data.classes, train_data.class_to_idx
    
    
    
    @staticmethod
    def load_data(
        root_dir: str,
        batch_size: int,
        transform: TorchTransformsComposeType, 
        split_ratio=.8,
        img_extension=".jpg",
        check_consistency=True,
        num_workers: int = os.cpu_count()
    ) -> tuple[TorchDataLoaderType, TorchDataLoaderType, List[str], Dict[str, int]]:
        """
        Load a dataset that is not yet split into `train_set` and `test_set`
        
        Folder structure
        ------
        ```
        root_dir/
            |__Cat/
                |_ image1.jpg
                |_ ...
            |__Dog/
                |_ image1.jpg
                |_ ...
        ```
        """
        root_dir = Path(root_dir)
        full_data = ImageFolder(
            root_dir, 
            img_extension, 
            transform=transform,
            check_consistency=check_consistency
        )
        if (full_length:= len(full_data)) == 0:
            raise RuntimeError("Cannot locate any file. Check the extension or folder structure")
        
        # Calculate lengths for train/test split
        train_length = int(full_length * split_ratio)
        test_length = full_length - train_length
        
        # Split the data randomly
        train_dataset, test_dataset = random_split(full_data, [train_length, test_length])

        # Turn images into DataLoaders
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True, # enables faster data transfer to CUDA-enabled GPUs
        )

        test_dataloader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
        return train_dataloader, test_dataloader, full_data.classes, full_data.class_to_idx



    

        
    
    



def save_model(model: torch.nn.Module, save_path: str, model_name: str):
    """
    Saves a PyTorch model to a target directory.

    Params:
    -------
    - `model`: A target PyTorch model to save.
    - `save_path`: A directory for saving the model to.
    - `model_name`: A filename for the saved model. 
        * Must include either ".pth" ".pt", or ".model" as the extension
    """
    # Check model instance
    if not isinstance(model, torch.nn.Module):
        raise ValueError("Provided model is not a PyTorch model instance.")
    # Create target directory
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    # Create model save path
    assert model_name.endswith((".pth", ".pt", ".model")), "Model name should end with '.pth', '.pt', or '.model'."
    model_save_path = save_path / model_name
    # Prevent accidental overwrite
    assert not model_save_path.exists(), f"A file named {model_name} already exists in the directory. To overwrite, please delete the existing file or choose a different name."
    # Save the model state_dict()
    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(), f=model_save_path)



def set_seeds(seed):
    """
    set universal value for seeds in PyTorch, Numpy, Random, CUDA
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If you are using CUDA
    np.random.seed(seed)
    random.seed(seed)
    # True, ensures that every time you run your code on a GPU, it 
    # produces the same results. It is associated with the NVIDIA 
    # CUDA Deep Neural Network library (cuDNN), which PyTorch uses 
    # under the hood for various operations to accelerate computations 
    # on NVIDIA GPUs.
    torch.backends.cudnn.deterministic = True




# try:
#     def create_writer(experiment_name: str, model_name: str, additional: Optional[str]="") -> tb.writer.SummaryWriter:
#         """
#         Creates a torch.utils.tensorboard.writer.SummaryWriter() 
#             instance that saves to a specific log_dir.

#         log_dir is a combination of 
#             runs/{timestamp}/{experiment_name}/{model_name}/{extra}.

#         ## args
#             - `experiment_name: Name of experiment.
#             - `model_name: Name of model.
#             - `additional (optional)`: Anything extra to add to the directory. Defaults to None.

#         ## return
#             torch.utils.tensorboard.writer.SummaryWriter(): Instance of a writer saving to log_dir.
#         """
#         if IS_TB_INSTALLED: raise ModuleNotFoundError("Install tensorboard module first before using this function.")

#         ROOT_DIR = "runs" # SummaryWriter default is "run"

#         TS = datetime.now().strftime("%Y-%m-%d")
#         print(TS)

#         log_dir = os.path.join(ROOT_DIR, TS, experiment_name, model_name, additional if additional else "")
            
#         print(f"[INFO] Created SummaryWriter, saving to: {log_dir}...")
#         return tb.writer.SummaryWriter(log_dir=log_dir)
# except NameError:
#     print("[WARN] tensorboard module not found. cannot use create_writer()")