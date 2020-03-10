from core.experiment_blueprint import Blueprint


def fn_wrapper(*args, **kwargs):
    f = Blueprint(**kwargs)
    f()

if __name__ == '__main__':
    from pytorch_utils.gpu_fn_scatter import configs_from_grid, scatter_fn_on_devices

    import torch.multiprocessing as mp

    mp.set_start_method('spawn', force=True)

    output_root_dir = '/scratch1/chofer/cifar10/vanilla_1k'

    grid = \
        {
            'output_root_dir': ['/tmp/testing/'],
            'num_runs': [1],
            'num_epochs': [100],
            'tag': ['test'],
            'model': [('VGG', {'vgg_name': 'VGG16'})],
            'batch_size': [128],
            'lr_init': [0.01],
            'weight_decay': [0.001],
            'ds_train': ['cifar100_train'],
            'ds_test': ['cifar100_test'],
            'eval_epoch': [1]
        }

    cfgs = configs_from_grid(grid)

    config = [((), c) for c in cfgs]
    scatter_fn_on_devices(fn_wrapper, config, [0, 1, 2], 3)
