# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Config files parser
"""
import argparse
import yaml

from geneva.utils.logger import Logger


# Global Keys
with open('geneva/config.yml', 'r') as f:
    keys = yaml.load(f)


# Experiment Configurations (Sorted by type then alphabetically)
def parse_config():
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')

    # Integers

    parser.add_argument('--batch_size',
                        type=int,
                        default=64,
                        help="Batch size used in training. Default: 64")

    parser.add_argument('--conditioning_dim',
                        type=int,
                        default=128,
                        help='Dimensionality of the projected text \
                        representation after the conditional augmentation. \
                        Default: 128')

    parser.add_argument('--disc_cond_channels',
                        type=int,
                        default=256,
                        help='For `conditioning` == concat, this flag'
                        'decides the number of channels to be concatenated'
                        'Default: 256')

    parser.add_argument('--embedding_dim',
                        type=int,
                        default=1024,
                        help='The dimensionality of the text representation. \
                        Default: 1024')

    parser.add_argument('--epochs',
                        type=int,
                        default=300,
                        help='Number of epochs for the experiment.\
                        Default: 300')

    parser.add_argument('--hidden_dim',
                        type=int,
                        default=256,
                        help='Dimensionality of the RNN hidden state which is'
                        'used as condition for the generator. Default: 256')

    parser.add_argument('--img_size',
                        type=int,
                        default=128,
                        help='Image size to use for training. \
                        Options = {128}')

    parser.add_argument('--image_feat_dim',
                        type=int,
                        default=512,
                        help='image encoding number of channels for the'
                        'recurrent setup. Default: 512')

    parser.add_argument('--input_dim',
                        type=int,
                        default=1024,
                        help='RNN condition dimension, the dimensionality of'
                        'the image encoder and question projector as well.')

    parser.add_argument('--inception_count',
                        type=int,
                        default=5000,
                        help='Number of images to use for inception score.')

    parser.add_argument('--noise_dim',
                        type=int,
                        default=100,
                        help="Dimensionality of the noise vector that is used\
                        as input to the generator: Default: 100")

    parser.add_argument('--num_workers',
                        type=int,
                        default=8,
                        help="Degree of parallelism to use. Default: 8")

    parser.add_argument('--num_objects',
                        type=int,
                        default=58,
                        help='Number of object the auxiliary objective of '
                        'object detection is trained on')

    parser.add_argument('--projected_text_dim',
                        type=int,
                        default=1024,
                        help='Pre-fusion text projection dimension for the'
                        'recurrent setup. Default: 1024')

    parser.add_argument('--save_rate',
                        type=int,
                        default=5000,
                        help='Number of iterations between saving current\
                        model sample generations. Default: 5000')

    parser.add_argument('--sentence_embedding_dim',
                        type=int,
                        default=1024,
                        help='Dimensionality of the sentence encoding training'
                        'on top of glove word embedding. Default: 1024')

    parser.add_argument('--vis_rate',
                        type=int,
                        default=20,
                        help='Number of iterations between each visualization.\
                        Default: 20')

    # Floats

    parser.add_argument('--aux_reg',
                        type=float,
                        default=5,
                        help='Weighting factor for the aux loss. Default: 5')

    parser.add_argument('--cond_kl_reg',
                        type=float,
                        default=None,
                        help='CA Net KL penalty regularization weighting'
                        'factor. Default: None, means no regularization.')

    parser.add_argument('--discriminator_lr',
                        type=float,
                        default=0.0004,
                        help="Learning rate used for optimizing the \
                        discriminator. Default = 0.0004")

    parser.add_argument('--discriminator_beta1',
                        type=float,
                        default=0,
                        help="Beta1 value for Adam optimizers. Default = 0")

    parser.add_argument('--discriminator_beta2',
                        type=float,
                        default=0.9,
                        help="Beta2 value for Adam optimizer. Default = 0.9")

    parser.add_argument('--discriminator_weight_decay',
                        type=float,
                        default=0,
                        help='Weight decay for the discriminator. Default= 0')

    parser.add_argument('--feature_encoder_lr',
                        type=float,
                        default=2e-3,
                        help='Learning rate for image encoder and condition'
                        'encoder. Default= 2e-3')

    parser.add_argument('--generator_lr',
                        type=float,
                        default=0.0001,
                        help="Learning rate used for optimizing the generator \
                        Default = 0.0001")

    parser.add_argument('--generator_beta1',
                        type=float,
                        default=0,
                        help="Beta1 value for Adam optimizers. Default = 0")

    parser.add_argument('--generator_beta2',
                        type=float,
                        default=0.9,
                        help="Beta2 value for Adam optimizer. Default = 0.9")

    parser.add_argument('--generator_weight_decay',
                        type=float,
                        default=0,
                        help='Weight decay for the generator. Default= 0')

    parser.add_argument('--gp_reg',
                        type=float,
                        default=None,
                        help='Gradient penalty regularization weighting'
                        'factor. Default: None, means no regularization.')

    parser.add_argument('--grad_clip',
                        type=float,
                        default=4,
                        help='Gradient clipping threshold for RNN and GRU.'
                        'Default: 4')

    parser.add_argument('--gru_lr',
                        type=float,
                        default=0.0001,
                        help='Sentence encoder optimizer learning rate')

    parser.add_argument('--rnn_lr',
                        type=float,
                        default=0.0005,
                        help='RNN optimizer learning rate')

    parser.add_argument('--wrong_fake_ratio',
                        type=float,
                        default=0.5,
                        help='Ratio of wrong:fake losses.'
                        'Default: 0.5')
    # Strings

    parser.add_argument('--activation',
                        type=str,
                        default='relu',
                        help='Activation function to use.'
                        'Options = [relu, leaky_relu, selu]')

    parser.add_argument('--arch',
                        type=str,
                        default='resblocks',
                        help='Network Architecture to use. Two options are'
                        'available {resblocks}')

    parser.add_argument('--conditioning',
                        type=str,
                        default=None,
                        help='Method of Conditioning text. Default is None.'
                        'Options: {concat, projection}')

    parser.add_argument('--condition_encoder_optimizer',
                        type=str,
                        default='adam',
                        help='Image encoder and text projection optimizer')

    parser.add_argument('--criterion',
                        type=str,
                        default='hinge',
                        help='Loss function to use. Options:'
                        '{classical, hinge} Default: hinge')

    parser.add_argument('--dataset',
                        type=str,
                        default='codraw',
                        help='Dataset to use for training. \
                        Options = {codraw, iclevr}')

    parser.add_argument('--discriminator_optimizer',
                        type=str,
                        default='adam',
                        help="Optimizer used while training the discriminator. \
                        Default: Adam")

    parser.add_argument('--disc_img_conditioning',
                        type=str,
                        default='subtract',
                        help='Image conditioning for discriminator, either'
                        'channel subtraction or concatenation.'
                        'Options = {concat, subtract}'
                        'Default: subtract')

    parser.add_argument('--exp_name',
                        type=str,
                        default='TellDrawRepeat',
                        help='Experiment name that will be used for'
                        'visualization and Logging. Default: TellDrawRepeat')

    parser.add_argument('--embedding_type',
                        type=str,
                        default='gru',
                        help='Type of sentence encoding. Train a GRU'
                        'over word embedding with \'gru\' option.'
                        'Default: gru')

    parser.add_argument('--gan_type',
                        type=str,
                        default='recurrent_gan',
                        help='Gan type: recurrent. Options:'
                        '[recurrent_gan]. Default: recurrent_gan')

    parser.add_argument('--generator_optimizer',
                        type=str,
                        default='adam',
                        help="Optimizer used while training the generator. \
                        Default: Adam")

    parser.add_argument('--gen_fusion',
                        type=str,
                        default='concat',
                        help='Method to use when fusing the image features in'
                        'the generator. options = [concat, gate]')

    parser.add_argument('--gru_optimizer',
                        type=str,
                        default='rmsprop',
                        help='Sentence encoder optimizer type')

    parser.add_argument('--img_encoder_type',
                        type=str,
                        default='res_blocks',
                        help='Building blocks of the image encoder.'
                        ' Default: res_blocks')

    parser.add_argument('--load_snapshot',
                        type=str,
                        default=None,
                        help='Snapshot file to load model and optimizer'
                        'state from')

    parser.add_argument('--log_path',
                        type=str,
                        default='logs',
                        help='Path where to save logs and image generations.')

    parser.add_argument('--results_path',
                        type=str,
                        default='results/',
                        help='Path where to save the generated samples')

    parser.add_argument('--rnn_optimizer',
                        type=str,
                        default='rmsprop',
                        help='Optimizer to use for RNN')

    parser.add_argument('--test_dataset',
                        type=str,
                        help='Test dataset path key.')

    parser.add_argument('--val_dataset',
                        type=str,
                        help='Validation dataset path key.')

    parser.add_argument('--vis_server',
                        type=str,
                        default='http://localhost',
                        help='Visdom server address')

    # Boolean

    parser.add_argument('-debug',
                        action='store_true',
                        help='Debugging flag.(e.g, Do not save weights)')

    parser.add_argument('-disc_sn',
                        action='store_true',
                        help='A flag that decides whether to use spectral norm'
                        'in the discriminator')

    parser.add_argument('-generator_sn',
                        action='store_true',
                        help='A flag that decides whether to use'
                        'spectral norm in the generator.')

    parser.add_argument('-generator_optim_image',
                        action='store_true',
                        help='A flag of whether to optimize the image encoder'
                        'w.r.t the generator.')

    parser.add_argument('-inference_save_last_only',
                        action='store_true',
                        help='A flag that decides whether to only'
                        'save the last image for each dialogue.')

    parser.add_argument('-metric_inception_objects',
                        action='store_true',
                        help='A flag that decides whether to evaluate & report'
                        'object detection accuracy')

    parser.add_argument('-teacher_forcing',
                        action='store_true',
                        help='a flag to indicate to whether to train using'
                        'teacher_forcing. NOTE: With teacher_forcing=False'
                        'more GPU memory will be used.')

    parser.add_argument('-self_attention',
                        action='store_true',
                        help='A flag that decides whether to use'
                        'self-attention layers.')

    parser.add_argument('-skip_connect',
                        action='store_true',
                        help='A flag that decides whether to have a skip'
                        'connection between the GRU output and the LSTM input')

    parser.add_argument('-use_fd',
                        action='store_true',
                        help='a flag which decide whether to use image'
                        'features conditioning in the discriminator.')

    parser.add_argument('-use_fg',
                        action='store_true',
                        help='a flag which decide whether to use image'
                        'features conditioning in the generator.')

    args = parser.parse_args()

    logger = Logger(args.log_path, args.exp_name)
    logger.write_config(str(args))

    return args
