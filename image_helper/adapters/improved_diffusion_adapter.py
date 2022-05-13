from ..image_helper import ImageHelper

# Improved Diffusion Adapter
# Model Code: https://github.com/openai/improved-diffusion

class DiffusionImageHelper(ImageHelper):
    def _get_default_init_args(self):
        init_args = super()._get_default_init_args()

        init_args.update({
            #'model_dir': None,   # Path to directory containing model checkpoints. Will load most recent model.
            #'use_ema': True,     # Whether to use the EMA checkpoint.
            'model': None,
            'diffusion': None,
            'image_size': None,   # The size of image that is generated. TODO: Needed? Can't we just check the properties of the model?
            'max_batch_size': 4,  # Default maximum batch size (batch size 4 uses ~3GB VRAM for 64x64 images).
        })

        return init_args
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        assert self.args['model'] is not None, 'Missing kwarg "model"'
        self.model = self.args['model']

        assert self.args['diffusion'] is not None, 'Missing kwarg "diffusion"'
        self.diffusion = self.args['diffusion']

        assert self.args['image_size'] is not None, 'Missing kwarg "image_size"'
        self.image_size = self.args['image_size']
    
    def _get_default_generator_args(self):
        return {
            'use_ddim': False,
            'clip_denoised': True,
            # 'diffusion_steps': 200, # TODO: Requires creation of our own Diffusion class.
            **super()._get_default_generator_args()
        }

    def _generate_samples(self, **kwargs):
        generator_args = self.default_generator_args.copy()
        generator_args.update(kwargs)

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