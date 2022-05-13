from ..image_helper import ImageHelper
from ..utils import update_dict
import torch

# Improved Diffusion Adapter
# Model Code: https://github.com/openai/improved-diffusion

class DiffusionImageHelper(ImageHelper):
    
    # Default arguments for __init__
    def _get_default_init_args(self):
        init_args = super()._get_default_init_args()

        init_args.update({
            'max_batch_size': 4,  # Default maximum batch size (batch size 4 uses ~3GB VRAM for 64x64 images with near-default settings).

            'model_dir': None,    # Path to directory containing model checkpoints. Will load most recent model.
            'use_ema': True,      # Whether to use the EMA checkpoint.
            
            # Model parameters
            'creation_args': {
                # Example settings below, these values aren't used anywhere.
                # Default values are pulled directly from the improved diffusion module.
                # You should pass the settings you used during training to the DiffusionImageHelper initializer.
                # For all arguments, see: https://github.com/openai/improved-diffusion/blob/783b6740edb79fdb7d063250db2c51cc9545dcd1/improved_diffusion/script_util.py#L11
                # Don't specify timestep_respacing here, it'll get adjusted automatically.

                #'image_size': 64,
                #'num_channels': 128,
                #'num_res_blocks': 2,
                #'learn_sigma': False,
                #'diffusion_steps': 4000,
                #'noise_schedule': 'linear',
            },
        })

        return init_args
    
    def __init__(self, **kwargs):
        self.model = None
        self.diffusion = None
        self.image_size = None
        self.timestep_respacing = None

        super().__init__(**kwargs)

        self._load_model()

    def _get_pattern_and_extraction_fn(self, use_ema):
        if use_ema:
            return 'ema_*.pt', lambda x: int(x.split('_')[-1].split('.')[0])
        else:
            return 'model*.pt', lambda x: int(x.split('.')[0].split('model')[-1])
    
    def _determine_best_model_path(self, model_dir, use_ema):
        pattern, extract_number_fn = self._get_pattern_and_extraction_fn(use_ema)
        best_model = self._get_most_recent_file_matching_pattern(
            dir = model_dir, 
            pattern = pattern, 
            extract_number_fn = extract_number_fn)
        self.debug_print(f'Using model: {best_model}')
        return best_model
        
    def _load_model(self, timestep_respacing = None):
        assert self.args['model_dir'] is not None, '"model_dir" not specified.'

        # Speed hack, don't actually load before the first run.
        if timestep_respacing is None and self.timestep_respacing is None:
            self.debug_print("Skipping load...")
            return

        from improved_diffusion import dist_util, logger
        from improved_diffusion.script_util import(
            NUM_CLASSES,
            model_and_diffusion_defaults,
            create_model_and_diffusion,
            add_dict_to_argparser,
            args_to_dict,
        )

        network_path = self._determine_best_model_path(model_dir = self.args['model_dir'], use_ema = self.args['use_ema'])
        creation_args = model_and_diffusion_defaults()
        creation_args.update(self.args['creation_args'])

        # Update image_size
        self.image_size = creation_args['image_size']

        if timestep_respacing is not None:
            self.timestep_respacing = creation_args['timestep_respacing'] = str(timestep_respacing)
        else:
            # Fall back to diffusion_steps.
            self.timestep_respacing = creation_args['timestep_respacing'] = str(creation_args['diffusion_steps'])
        self.debug_print(f'Using timestep_respacing: {self.timestep_respacing}')
        
        self.debug_print('Constructing network...')
        self.debug_print(f'  with arguments: {creation_args}')
        self.model, self.diffusion = create_model_and_diffusion(**creation_args)
        
        self.debug_print('Loading network...')
        self.debug_print(f'  using checkpoint: {network_path}')
        self.model.load_state_dict(dist_util.load_state_dict(network_path, map_location='cpu'))

        self.debug_print('Transferring to device and initializing...')
        self.model.to(self.device)
        _ = self.model.eval()
    
    # Default arguments for _generate_samples
    def _get_default_generator_args(self):
        generator_args = super()._get_default_generator_args()

        generator_args.update({
            'use_ddim': False,
            'clip_denoised': True,
            'show_progress': False,
            'diffusion_steps': None, # Use this number of diffusion steps.
        })

        return generator_args

    def _generate_samples(self, **kwargs):
        self.debug_print(f'kwargs: {kwargs}')
        generator_args = self.default_generator_args.copy()
        update_dict(generator_args, kwargs)
        self.debug_print(f'Generating samples with arguments: {generator_args}')

        # TODO: Check if the ddim setting matches the timestep_respacing argument.

        if generator_args['diffusion_steps'] is not None and str(self.timestep_respacing) != str(generator_args['diffusion_steps']):
            # Need to recreate diffusion class with new respacing.
            # TODO Optimization: Recreate it by hand somehow so we don't have to reload the entire model.
            self.debug_print(f'Reloading model for new timestep_respacing: {generator_args["diffusion_steps"]} (was {self.timestep_respacing})')
            self._load_model(timestep_respacing = generator_args['diffusion_steps'])

        sample_fn = (self.diffusion.p_sample_loop if not generator_args['use_ddim'] else self.diffusion.ddim_sample_loop)

        return sample_fn(
            self.model,
            (generator_args['num_samples'], 3, self.image_size, self.image_size),
            clip_denoised = generator_args['clip_denoised'],
            model_kwargs = {},
            progress = generator_args['show_progress'],
        )
    
    def _generated_samples_range(self):
        return (-1., 1.)



class UpsampledDiffusionImageHelper(DiffusionImageHelper):
    # Default arguments for __init__
    def _get_default_init_args(self):
        init_args = super()._get_default_init_args()

        init_args.update({
            # TODO: Support different batch sizes between base and upsampler.
            'max_batch_size': 4,     # Default maximum batch size (batch size 4 uses ~6GB VRAM for both 64 and 256 models with near-default settings).

            'up_model_dir': None,    # Path to directory containing upsampler model checkpoints. Will load most recent model.
            'up_use_ema': True,      # Whether to use the EMA checkpoint.
            
            # Model parameters
            'up_creation_args': { },
        })

        return init_args
    
    def __init__(self, **kwargs):
        self.up_model = None
        self.up_diffusion = None
        self.up_image_size = None
        self.up_timestep_respacing = None

        super().__init__(**kwargs)
        self._load_upsample_model()
    
    def _load_upsample_model(self, timestep_respacing = None):
        assert self.args['up_model_dir'] is not None, '"up_model_dir" not specified.'
        
        # Load upsampler model
        from improved_diffusion import dist_util, logger
        from improved_diffusion.script_util import (
            sr_model_and_diffusion_defaults,
            sr_create_model_and_diffusion,
            add_dict_to_argparser,
            args_to_dict,
        )
        
        # Speed hack, don't actually load before the first run.
        if timestep_respacing is None and self.up_timestep_respacing is None:
            self.debug_print("Skipping load...")
            return

        network_path = self._determine_best_model_path(model_dir = self.args['up_model_dir'], use_ema = self.args['up_use_ema'])
        creation_args = sr_model_and_diffusion_defaults()
        creation_args.update(self.args['up_creation_args'])

        # Update image_size
        self.up_image_size = creation_args['large_size']

        if timestep_respacing is not None:
            self.up_timestep_respacing = creation_args['timestep_respacing'] = str(timestep_respacing)
        else:
            # Fall back to diffusion_steps.
            self.up_timestep_respacing = creation_args['timestep_respacing'] = str(creation_args['diffusion_steps'])
        self.debug_print(f'Using up_timestep_respacing: {self.up_timestep_respacing}')

        self.debug_print('Constructing upsampler network...')
        self.debug_print(f'  with arguments: {creation_args}')
        self.up_model, self.up_diffusion = sr_create_model_and_diffusion(**creation_args)

        self.debug_print('Loading upsampler network...')
        self.debug_print(f'  using checkpoint: {network_path}')
        self.up_model.load_state_dict(dist_util.load_state_dict(network_path, map_location='cpu'))

        self.debug_print('Transferring to device and initializing...')
        self.up_model.to(self.device)
        _ = self.up_model.eval()
    
    def _get_default_generator_args(self):
        generator_args = super()._get_default_generator_args()

        generator_args.update({
            'up': {
                'upsample_temp': 1.0,
                **super()._get_default_generator_args(),
            }
        })
        return generator_args
    
    def _generate_samples(self, **kwargs):
        generator_args = self.default_generator_args.copy()
        update_dict(generator_args, kwargs)

        self.debug_print(f'Generating samples...')
        self.debug_print(f'  with arguments: {generator_args}')

        # Generate samples with base model.
        base_samples = super()._generate_samples(**generator_args)

        # Upsample samples.
        upsample_args = generator_args['up']
        upsample_args['num_samples'] = base_samples.shape[0]
        self.debug_print('Upsampling samples...')
        self.debug_print(f'  with arguments: {upsample_args}')
        model_kwargs = dict(
            low_res = base_samples,
        )

        # TODO: Check if the ddim setting matches the timestep_respacing argument.

        if upsample_args['diffusion_steps'] is not None and str(self.up_timestep_respacing) != str(upsample_args['diffusion_steps']):
            # Need to recreate diffusion class with new respacing.
            # TODO Optimization: Recreate it by hand somehow so we don't have to reload the entire model.
            self.debug_print(f'Reloading upsampler model for new timestep_respacing: {upsample_args["diffusion_steps"]} (was {self.up_timestep_respacing})')
            self._load_upsample_model(timestep_respacing = upsample_args['diffusion_steps'])

        sample_fn = (self.up_diffusion.p_sample_loop if not upsample_args['use_ddim'] else self.up_diffusion.ddim_sample_loop)
        up_shape = (base_samples.shape[0], 3, self.up_image_size, self.up_image_size)
        up_samples = sample_fn(
            self.up_model,
            up_shape,
            noise=torch.randn(up_shape, device=self.device) * upsample_args['upsample_temp'],
            device=self.device,
            clip_denoised = upsample_args['clip_denoised'],
            progress = upsample_args['show_progress'],
            model_kwargs = model_kwargs,
        )

        return up_samples