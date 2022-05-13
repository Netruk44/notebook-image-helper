# ImageHelper is the base class for all image helpers
#from IPython import display as ipydisplay
from .generated_images import GeneratedImages
import glob
from math import ceil
import os
import re
import torch

class ImageHelper(object):
    # Arguments for __init__
    def _get_default_init_args(self):
        return {
            'device': 'cpu',
            'max_batch_size': 1,
            'default_generate_args': self._get_default_generate_args(),
            'default_generator_args': self._get_default_generator_args(),
            #'show_progress': False, # TODO
            #'show_progress_fn': ipydisplay,
            'debug': False,
        }

    def __init__(self, **kwargs):
        """
        Initialize the ImageHelper.

        Parameters
        ----------
        device: str
            The device to use for the model.
        max_batch_size: int
            The maximum batch size to use for the model.
        default_generate_args: dict
            The default arguments to use for generate_image.
        """

        self.args = self._get_default_init_args()

        for k in kwargs:
            if k not in self.args:
                print(f'WARNING: Ignoring unknown argument {k}')
        
        self.args.update(kwargs)

        self.device = self.args['device']
        self.max_batch_size = self.args['max_batch_size']
        self.default_generate_args = self.args['default_generate_args']
        self.default_generator_args = self.args['default_generator_args']
        self.debug_print = print if self.args['debug'] else lambda *args, **kwargs: None
    
    # Arguments for generate_images
    def _get_default_generate_args(self):
        return {
            'num_samples': 1,
            'output_range': (0, 1),
            'dtype': torch.float32,
            'show_progress': False,
            'show_subprogress': False,
            'generator_args': {},
        }

    def generate_images(self, **kwargs):
        """
        Generate one or more images.

        Parameters
        ----------
        num_samples: int
            The number of images to generate.
        output_range: tuple
            The range of values to output.
        dtype: torch.dtype
            The dtype to use for the image.
        show_progress: bool
            Whether to show a progress bar during generation (using tqdm).
        show_subprogress: bool
            Whether to show a progress bar for each micro-batch (using tqdm).
        """

        generate_args = self.default_generate_args.copy()
        generate_args.update(kwargs)
        self.debug_print(f'Generating images with arguments: {generate_args}')

        num_imgs = generate_args['num_samples']
        show_progress = generate_args['show_progress']
        max_batch_size = self.max_batch_size

        all_samples = []
        cur_sample_count = 0
        bar = None

        if show_progress:
            from tqdm import tqdm
            bar = tqdm(total=num_imgs)
        
        for _ in range(ceil(num_imgs / max_batch_size)):
            batch_size = min(self.max_batch_size, num_imgs - cur_sample_count)

            current_args = generate_args['generator_args'].copy()
            current_args.update({
                'num_samples': batch_size,
                'show_progress': generate_args['show_subprogress'],
            })

            self.debug_print(f'Generating {batch_size} samples...')
            all_samples.append(self._generate_samples(**current_args))

            if show_progress:
                bar.update(batch_size)
            cur_sample_count += batch_size
        
        all_samples = torch.cat(all_samples, dim=0)

        if show_progress:
            bar.close()

        return GeneratedImages(all_samples, self._generated_samples_range()).to_range(generate_args['output_range']).to_dtype(generate_args['dtype'])
        
    # "Protected virtual" methods for subclasses to implement
    # Arguments for _generate_samples
    def _get_default_generator_args(self):
        return {}
        
    def _generate_samples(self, **kwargs):
        raise NotImplementedError
    
    def _generated_samples_range(self):
        raise NotImplementedError
    
    # "Protected" methods for subclasses to use
    def _get_most_recent_file_matching_pattern(self, dir, pattern, extract_number_fn):
        matching_paths = glob.glob(os.path.join(dir, pattern))
        matching_paths = sorted(matching_paths, key=lambda x: extract_number_fn(x))
        return matching_paths[-1]
