'''
This file contains utility functions that enhance certain torch functions
'''
import os, sys, random, time
import collections
from importlib.metadata import distribution
from typing import *
from pathlib import Path
from PIL import Image
from functools import lru_cache


import torch
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset

from .utility import *



IMG_EXTENSIONS = [
    '.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.gif', '.webp'
]




class ImageFolder(Dataset):
    '''
    This class enhances `torch.utils.data.Dataset`
    
    It expects images organized into class-specific folders.
    
    Folder Structure:
    -----------------
    ```
    |__train/
            |__Cat/
                |_ image1.jpg
                |_ ...
            |__Dog/
                |_ image1.jpg
                |_ ...
    ```
    '''
    MAX_IMAGE_CACHE = 100
    
    def __init__(self, 
        img_dir:           str, 
        img_ext:           str, 
        transform:         TorchTransformsComposeType  = None,
        check_consistency: bool                        = True
    ):
        '''
        Params:
        -------
        - `img_dir`: images' root directory
        - `img_ext`: targeted images' extension
            * All images should have the same extension in the directory
        - `transform`: transform function(s) that can be applied to the images. 
            * This is useful for preprocessing images, e.g., resizing, normalization, data augmentation, etc.
        - `check_consistency`: If true, force check the images
        '''
        start = time.time()
        self.__dict__['is_initialized'] = False
        
        self.img_dir                        = img_dir
        self.img_ext                        = img_ext
        self.transform                      = transform
        self.check_consistency              = check_consistency
        self.paths:             List[Path]  = []
        
        mismatch_set                        = set()

        for f in Path(img_dir).glob(f"*/*.*"):
            if (ext:= f.suffix) in IMG_EXTENSIONS:
                if ext == self.img_ext:
                    self.paths.append(f)
                else:
                    mismatch_set.add(ext)

        if len(self) == 0: raise ValueError("Cannot find any image")
        
        if check_consistency and len(mismatch_set) > 0:
            raise ValueError(f"Found {len(mismatch_set)} image files with extensions {mismatch_set} other than {self.img_ext} in {self.img_dir}")
        del mismatch_set

        self.classes, self.class_to_idx = self.find_classes(img_dir)
        '''Sorted alphabetically'''
        self.__dict__['is_initialized'] = True
        
        print(f"{self.__class__.__name__} initialized after {(time.time() - start):.4f} seconds. Found: {len(self)} samples.")
        
        
    def __setattr__(self, attr, val):
        if self.__dict__.get('is_initialized') and attr in (
            'img_dir', 'img_extension', 'transform', 'check_consistency'):
            raise ValueError(f'{attr} attribute should not be set after {self.__class__.__name__} is initialized')

        super().__setattr__(attr, val)
        
        
    def __getitem__(self, 
        key: Union[int, slice]
    ) -> Union[Tuple[torch.Tensor, int], Tuple[list, list]]:
        '''
        Fetch and return a sample (image and label) based on index.
            * If the key is an integer, returns a tuple (image, label).
            * If the key is a slice, returns a tuple of lists ([images], [labels]).

        Examples:
        ---------
        >>> img,  label  = dataset[0]
        >>> imgs, labels = dataset[0:5]
        >>> imgs, labels = dataset[0:5:2]
        '''
        
        if isinstance(key, slice):
            imgs, class_idxs = [], []
            for idx in range(*key.indices(len(self.paths))):
                img, class_idx = self.__get_single_item(idx)
                imgs.append(img)
                class_idxs.append(class_idx)
            return imgs, class_idxs
        elif isinstance(key, int):
            return self.__get_single_item(key)
        else:
            raise ValueError("Key must be either an int or a slice.")

    def __iter__(self):
        self._current = 0 # Initialize a counter for the iterator
        return self

    def __next__(self):
        if self._current < len(self.paths):
            item = self[self._current]
            self._current += 1
            return item
        else: # If no more items, raise the StopIteration exception
            raise StopIteration

    def __len__(self) -> int:
        '''Returns the total number of samples'''
        return len(self.paths)
    
    def __repr__(self):
        return (
            f"{self.__class__.__name__} |"
            f"img_dir={self.img_dir} |"
            f"total_len={len(self.paths)} |"
            f"transform={self.transform} |"
            f"check_consistency={self.check_consistency}"
        )

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__} |"
            f"img_dir = {self.img_dir} |"
            f"transform={self.transform}"
        )
    

    # =============================
    # Private
    # =============================
    def __get_single_item(self, 
        index: int
    ) -> Tuple[TorchTensorType, str]:
        ''' Helper method to get a single item '''
        IMG_PATH   = self.paths[index]
        CLASS_NAME = IMG_PATH.parent.name
        
        if CLASS_NAME not in self.class_to_idx:
            raise RuntimeError(f"Class name '{CLASS_NAME}' not found in classes.")

        IMG        = self.load_image(IMG_PATH)
        tensor = self.transform(IMG) if self.transform else ToTensor()(IMG)
        
        return tensor, CLASS_NAME, str(IMG_PATH)


    # =============================
    # Public
    # =============================
    @lru_cache(maxsize=MAX_IMAGE_CACHE)  # Adjust maxsize as per your memory constraints
    def load_image(self, path: str) -> PILImageType:
        ''' Load an image path as `PIL Image`'''
        return Image.open(path)
    
    def get_class_name(self, class_idx: int) -> str:
        ''' Returns the `class name/folder name` for a given class index. '''
        if not 0 <= class_idx < len(self.classes):
            raise ValueError(f"{class_idx} is out of range (0 ~ {len(self.classes)})")
        return self.classes[class_idx]

    def get_distribution(self) -> Dict[str, int]:
        '''
        Returns the distribution of samples per class.

        Returns:
        --------
        - { k=`class names`, v=`number of samples` }
        '''
        distribution = collections.defaultdict(int)

        for path in self.paths:
            class_name = path.parent.name
            distribution[class_name] += 1

        return dict(distribution)
    
    def get_random_image_tensor(self) -> Tuple[TorchTensorType, str]:
        ''' Get a random image tensor from the dataset using matplotlib. '''
        random_idx = random.randint(0, len(self) - 1)
        image, label = self[random_idx]
        return image, label



    # =============================
    # Static
    # =============================
    @staticmethod
    def find_classes(dir_path: str) -> Tuple[List[str], Dict[str, int]]:
        '''
        Get class names based on directory
        
        Folder Structure:
        -----------------
        ```
        |__train/
                |__Cat/
                    |_ image1.jpg
                    |_ ...
                |__Dog/
                    |_ image1.jpg
                    |_ ...
        ```
        Examples:
        ---------
        >>> classes, class_to_idx = ImageFolderClone.find_classes("root_path/")
        >>> print(classes)
        ['Dog', 'Cat', 'Bird', ...]
        >>> print(class_to_idx)
        {'Dog': 0, 'Cat': 1, 'Bird': 2, ...}
        '''
        classes = [entry.name for entry in os.scandir(dir_path) if entry.is_dir()]
        if not classes:
            raise FileNotFoundError(f"Couldn't find any class in {dir_path}. Check folder structure!")

        class_idx_dict = {class_name: i for i, class_name in enumerate(classes)}
        return classes, class_idx_dict
