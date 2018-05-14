"""
Defines all hyperparameters for RAM model.
"""

class Config(object):

    # glimpse network
    convert_ratio       = 0.8
    original_size       = 256
    new_size            = 256
    num_channels        = 1  # remove later

    glimpse_size        = 12
    bandwidth           = glimpse_size**2

    n_patches           = 2 # # patches at each t
    num_glimpses        = 50 # samples before decision
    scale               = 2 # how much receptive field is scaled up for each glimpse

    sensor_size         = glimpse_size**2 * n_patches

    minRadius           = 8
    hg_size = hl_size   = 128
    loc_dim             = 2
    g_size              = 256

    # logging

    # training
    batch_size          = 16
    eval_batch_size     = 50
    num_epoch           = 600
    lr_start            = 5e-4
    lr_min              = 1e-5
    loc_std             = 0.05
    max_grad_norm       = 5.
    n_verbose           = 100

    # lstm
    cell_output_size    = 256
    cell_size           = 256
    cell_out_size       = cell_size

    # task
    num_classes         = 5
    n_distractors       = 4 # nr of digit distractors for cluttered task

    # monte carlo sampling
    M                   = 10
