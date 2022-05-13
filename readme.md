# Overview

A hobby project. Probably lots of errors lurking beneath the surface, I haven't tested this very thoroughly yet.

Notebook Image Helper provides a generic interface to wrap an image generation model, and provides a few helper methods for manipulating the model and its output. Mainly intended for use in Jupyter notebooks.

Current features include:

  * Automatic splitting of batches using `max_batch_size`.
  * Automation to load the most recent checkpoint from a folder.
  * A simpler interface for generating images with code.
  * Also a simpler interface for interacting with the generated images.

## Example Usage

### OpenAI's "Improved Diffusion" Model

[Link](https://github.com/openai/improved-diffusion)

After training your model, you can generate images with the following code in a Jupyter Notebook:

```python
from image_helper.adapters.improved_diffusion_adapter import UpsampledDiffusionImageHelper

# Load the model
image_helper = UpsampledDiffusionImageHelper(
    # The checkpoint directories, will automatically load the latest checkpoint from the folder.
    model_dir = '/path/to/base_model/checkpoint_dir/',
    up_model_dir = '/path/to/upsampler_model/checkpoint_dir/',

    device = 'cuda:0', # Where to put the model
    max_batch_size = 1, # Will automatically split batches into smaller batches as necessary. Lower VRAM usage at the expense of time.
    
    # Base diffusion model settings
    use_ema = True,
    creation_args = { # Use your training settings for the base model
        'image_size': 64,
        'num_channels': 128,
        'num_res_blocks': 3,
        'learn_sigma': True,
        'diffusion_steps': 4000,
        'noise_schedule': 'cosine',
    },
    
    # Upsampler diffusion model settings
    up_use_ema = False,
    up_creation_args =  { # Use your training settings for the upsampler
        'small_size': 64,
        'large_size': 256,
        'num_channels': 128,
        'num_res_blocks': 3,
        'learn_sigma': True,
        'diffusion_steps': 4000,
        'noise_schedule': 'cosine',
    }
)

# Generate images
images = image_helper.generate_images(
    num_samples = 16, 
    generator_args = {
        'diffusion_steps': 250,
        
        'up': {
            'diffusion_steps': 'ddim25',
            'use_ddim': True,
            'upsample_temp': 0.99,
        },
    }, 
    show_progress = True
)

# Show the images
# Multiple images are shown as a single image using `torchvision.utils.make_grid`.
images.show()

# Save the images
images.save('/path/to/image.png')
```
