import sys

params = {}


def init():
    # paths
    params['data_dir'] = 'data'
    params['tb_dir'] = 'tensorboard_events/'
    params['check_dir'] = 'checkpoints/'

    # train configuration
    params['batch_size'] = 2
    params['batches'] = 150000

    # datset information
    params['dataset'] = 'fashion3d'
    params['image_size'] = 256
    params['volume_size'] = 64  # height/width of volume, used by dataset to generate masks, must be 64 for our model
    params['data_workers'] = 7  # parallel workers for bodypart-mask generation and transformation estimation

    # augmentation
    params['augment_color'] = True
    params['augment_transform'] = True

    # volume architecture, change these to create a smaller or larger model
    params['before_count'] = 3  # number of 3D residual blocks before warping
    params['after_count'] = 3  # number of 3D residual blocks after warping
    params['residual_channels'] = 64  # number of 3D channels
    params['depth'] = 32  # depth of the volume

    # adam parameters
    params['alpha'] = 2e-4
    params['beta1'] = 0.5
    params['beta2'] = 0.999

    # loss weighting
    params['feature_loss_weight'] = 3.

    # checkpoints and tensorboard output
    params['steps_per_checkpoint'] = 1000
    params['steps_per_validation'] = 1000
    params['steps_per_scalar_summary'] = 20
    params['steps_per_image_summary'] = 200

    # validation configuration
    params['with_valid'] = True  # if True, training is performed on train and valid and tb outputs are on test split
    params['valid_count'] = 256  # number of samples validation is based on

    params['test_count'] = 20    #number of samples to perform test

    params['name'] = 'fash-3d_w-3d_p'  # name will be appended to both the checkpoint directory and the tebsorboard directory


init()

if len(sys.argv) == 2:
    if sys.argv[1] == 'params':
        for p, v in params.items():
            print('{}:\t{}'.format(p, v))
        raise ValueError

par_names = sys.argv[1::2]
par_vals = sys.argv[2::2]

if len(par_names) != len(par_vals):
    raise ValueError('Number of inputs must be even')

for name, val in zip(par_names, par_vals):
    if name not in params:
        if name == '-f':
            continue
        raise ValueError(f'{name} is not a valid parameter')
    if type(params[name]) == bool:
        params[name] = val == 'True'
    else:
        params[name] = type(params[name])(val)

params['tb_dir'] += params['name'] + '/'
params['check_dir'] += params['name'] + '/'
