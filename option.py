import os, copy
import configargparse
import socket
from helper.util import get_teacher_name
from utils import update_args, strlist_to_list, check_path

hostname = socket.gethostname()

# parser = configargparse.ArgumentParser('argument for training')
from smilelogging import argparser as parser

subparsers = parser.add_subparsers()

parser.add_argument('--print_freq',
                    type=int,
                    default=100,
                    help='print frequency')
parser.add_argument('--tb_freq', type=int, default=500, help='tb frequency')
parser.add_argument('--save_freq', type=int, default=-1, help='save frequency')
parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
parser.add_argument('--num_workers',
                    type=int,
                    default=8,
                    help='num of workers to use')
parser.add_argument('--epochs',
                    type=int,
                    default=240,
                    help='number of training epochs')
parser.add_argument('--init_epochs',
                    type=int,
                    default=30,
                    help='init training for two-stage methods')

# optimization
parser.add_argument('--learning_rate',
                    type=float,
                    default=0.05,
                    help='learning rate')
parser.add_argument('--lr_decay_epochs',
                    type=str,
                    default='150,180,210',
                    help='where to decay lr, can be a list')
parser.add_argument('--lr_decay_rate',
                    type=float,
                    default=0.1,
                    help='decay rate for learning rate')
parser.add_argument('--weight_decay',
                    type=float,
                    default=5e-4,
                    help='weight decay')
parser.add_argument('--weight_decay_schedule',
                    type=str,
                    default=None,
                    help='using non-constant weight decay')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

# dataset
parser.add_argument('--dataset',
                    type=str,
                    default='cifar100',
                    choices=['cifar100', 'svhn', 'imagenet', 'imagenet100', 'tinyimagenet'],
                    help='dataset')

# model
parser.add_argument(
    '--model_s',
    type=str,
    default='resnet8',
    choices=[
        'resnet8', 'resnet14', 'resnet20', 'resnet32', 'resnet44', 'resnet56',
        'resnet110', 'resnet8x4', 'resnet32x4', 'wrn_16_1', 'wrn_16_2',
        'wrn_40_1', 'wrn_40_2', 'wrn_28_10', 'vgg8', 'vgg8_3neurons', 'vgg11',
        'vgg13', 'vgg16', 'vgg19', 'ResNet50', 'MobileNetV2',
        'MobileNetV2_0_25', 'ShuffleV1', 'ShuffleV2', 'ShuffleV2_0_3',
        'resnet18', 'resnet34', 'resnet50'
    ])
parser.add_argument('--path_t',
                    type=str,
                    default=None,
                    help='teacher model snapshot')

# distillation
parser.add_argument('--distill',
                    type=str,
                    default='kd',
                    choices=[
                        'kd', 'hint', 'attention', 'similarity', 'correlation',
                        'vid', 'crd', 'kdsvd', 'fsp', 'rkd', 'pkt', 'abound',
                        'factor', 'nst', 'dcs', 'dcs+crd'
                    ])
parser.add_argument('--trial', type=str, default='1', help='trial id')

parser.add_argument('-r',
                    '--gamma',
                    type=float,
                    default=1,
                    help='weight for classification')
parser.add_argument('-a',
                    '--alpha',
                    type=float,
                    default=None,
                    help='weight balance for KD')
parser.add_argument('-b',
                    '--beta',
                    type=float,
                    default=None,
                    help='weight balance for other losses')

# KL distillation
parser.add_argument('--kd_T',
                    type=float,
                    default=4,
                    help='temperature for KD distillation')
parser.add_argument('--kd_S',
                    type=float,
                    default=4,
                    help='temperature for KD distillation')

# NCE distillation
parser.add_argument('--feat_dim',
                    default=128,
                    type=int,
                    help='feature dimension')
parser.add_argument('--mode',
                    default='exact',
                    type=str,
                    choices=['exact', 'relax'])
parser.add_argument('--nce_k',
                    default=16384,
                    type=int,
                    help='number of negative samples for NCE')
parser.add_argument('--nce_t',
                    default=0.07,
                    type=float,
                    help='temperature parameter for softmax')
parser.add_argument('--nce_m',
                    default=0.5,
                    type=float,
                    help='momentum for non-parametric updates')

# hint layer
parser.add_argument('--hint_layer',
                    default=2,
                    type=int,
                    choices=[0, 1, 2, 3, 4])

# --- @mst
parser.add_argument('--save_img_interval', type=int, default=1000000000)

parser.add_argument('--n_branch_fc_T', type=int, default=1)
parser.add_argument('--n_branch_fc_S', type=int, default=1)
parser.add_argument('--branch_layer_S', type=str, default='[]')
parser.add_argument('--branch_layer_T', type=str, default='[]')
parser.add_argument('--s_branch_target',
                    type=str,
                    default='t_branch',
                    choices=['t_branch', 't_logits'])
parser.add_argument('--branch_dropout_rate',
                    type=str,
                    default='[0,0,0,0,0,0]',
                    help='multi-head kd loss dropout rate')
parser.add_argument('--crd_multiheads', action="store_true")
parser.add_argument('--d_z', type=int, default=1000)
parser.add_argument('--update_data_interval', type=int, default=2)
parser.add_argument('--update_data_start', type=int, default=100)
parser.add_argument('--resume_student',
                    type=str,
                    default='',
                    help='resume student model')
parser.add_argument('--finetune_student',
                    type=str,
                    default='',
                    help='finetune student model')
parser.add_argument('--save_crd_loss', action='store_true')
parser.add_argument('--fix_student', action='store_true')
parser.add_argument('--reinit_student', action='store_true')
parser.add_argument('--fix_embed', action='store_true')
parser.add_argument('--embed', type=str, default='original')
parser.add_argument('--pretrained_embed', type=str, default='')
parser.add_argument('--model_t', type=str, default='resnet34')
parser.add_argument('--model_t_pretrained',
                    type=str,
                    help='the path of our own pretrained models')
parser.add_argument('--model_s_pretrained',
                    type=str,
                    help='the path of our own pretrained models')
parser.add_argument('--max_min_ratio', type=float, default=5)
parser.add_argument('--lr_DA', type=float, default=0)
parser.add_argument('--online_augment', action='store_true')
parser.add_argument('--two_loader', action='store_true')
parser.add_argument(
    '--ratio_CE_loss',
    type=float,
    default=1,
    help='in a batch, how many examples are counted for CE loss')
parser.add_argument(
    '--lw_mix',
    type=str,
    default=None,
    help='various mixup data augmentation ideas: mixup, manifold mixup, cutmix'
)
parser.add_argument('--mix_mode',
                    type=str,
                    choices=[
                        'identity', 'crop', 'flip', 'crop+flip', 'mixup',
                        'manifold_mixup', 'cutout', 'cutmix', 'cutmix_pick',
                        'patchmix', 'deepmixup', 'autoaugment', 'same', 'DA_pick'
                    ])
parser.add_argument('--DA_pick_base',
                    type=str,
                    choices=[
                        'identity', 'crop', 'flip', 'crop+flip', 'mixup',
                        'manifold_mixup', 'cutout', 'cutmix', 'cutmix_pick',
                        'patchmix', 'deepmixup', 'autoaugment', 'same',
                    ])
parser.add_argument('--mix_n_run', type=int, default=1)
parser.add_argument('--cut_size', type=int, default=16)
parser.add_argument('--lw_branch_kld', type=float, default=0.9)
parser.add_argument('--lw_branch_ce', type=float, default=0.1)
parser.add_argument('--cutmix_pick_criterion', type=str, default='kld')
parser.add_argument('--epoch_factor', type=float, default=0)
parser.add_argument(
    '--use_DA',
    type=str,
    default='11',
    help=
    'first 1 indicates using rand crop; second 1 indicates using horizontal flip'
)
parser.add_argument(
    '--no_DA',
    action='store_true',
    help='maintain back-compatibility, deprecated. Use --use_DA')
parser.add_argument(
    '--lw_dcs',
    type=float,
    default=0.5,
    help=
    'when dcs+crd, the <b> argument is taken by crd, so introduce one for dcs')
parser.add_argument('--test_teacher', action='store_true')
parser.add_argument('--save_entropy_log_step',
                    type=int,
                    default=0,
                    help='step at which to save the entropy log')
parser.add_argument(
    '--ceiling_ratio_schedule',
    type=str,
    default='',
    help='ceiling area ratio in cutmix. example: 0:0.75,150:1,180:1,210:1')
parser.add_argument(
    '--floor_ratio_schedule',
    type=str,
    default='',
    help='floor area ratio in cutmix. example: 0:0.75,150:1,180:1,210:1')
parser.add_argument('--check_cutmix_label', action='store_true')
parser.add_argument('--bbox',
                    type=str,
                    default='rand_bbox',
                    choices=['rand_bbox', 'simpler_rand_bbox'],
                    help='the func of generating random bounding box')
parser.add_argument('--t_output_as_target_for_input_mix', action='store_true')
parser.add_argument('--cutmix_pick_scheme', type=str, default='sort')
parser.add_argument(
    '--n_pick',
    type=int,
    default=64,
    help='the number of samples picked out of all augmented samples')
parser.add_argument(
    '--input_mix_no_kld_epoch',
    type=int,
    default=1000000,
    help='after this epoch, NOT apply kld loss to the mixed input')
parser.add_argument('--train_linear_classifier',
                    action='store_true',
                    help='train linear classifier for middle layers')
parser.add_argument('--head_init',
                    type=str,
                    default='default',
                    choices=['default', 'orth'])
parser.add_argument('--branch_width_T', type=int, default=256)
parser.add_argument('--branch_width_S', type=int, default=256)
parser.add_argument(
    '--epoch_stop_head_kd_loss',
    type=int,
    default=10000000,
    help='after this epoch, for MHKD, do not apply kd loss to the heads')
parser.add_argument('--amp',
                    action='store_true',
                    help='automatic mixed precision training')
parser.add_argument('--only_test', type=str, default='')
parser.add_argument('--modify_student_input', type=str, default='')
parser.add_argument('--stack_input', action='store_true')
parser.add_argument('--n_patch',
                    type=int,
                    default=4,
                    help='num of patches in masking')
parser.add_argument('--mask_zero_ratio', type=float, default=0.5)
parser.add_argument('--fix_T_heads',
                    action='store_true',
                    help='fix the teacher heads when using dcs')
parser.add_argument('--test_loader_in_train',
                    action='store_true',
                    help="use test loader in training; for checking the std of teacher's mean prob")

# 'deepmixup' related arguments
parser.add_argument('--utils.ON', action='store_true')
parser.add_argument('--utils.check_ce_var', action='store_true')
parser.add_argument('--utils.check_ce_var_trainset_size',
                    type=int,
                    default=1000)
parser.add_argument('--utils.check_ce_var_v2', action='store_true')
parser.add_argument('--utils.check_ce_var_v3', action='store_true')

# ---
opt = parser.parse_args()

# check args
opt.branch_layer_T = strlist_to_list(opt.branch_layer_T, int)
opt.branch_layer_S = strlist_to_list(opt.branch_layer_S, int)
opt.branch_dropout_rate = strlist_to_list(opt.branch_dropout_rate, float)
opt.print_interval = opt.print_freq
opt.lw_mix = strlist_to_list(opt.lw_mix, float)
opt.entropy_log = [] if opt.save_entropy_log_step > 0 else None  # initialize

# set different learning rate from these 4 models
if opt.dataset in ['cifar100', 'tinyimagenet']:
    if opt.model_s in ['MobileNetV2', 'ShuffleV1', 'ShuffleV2']:
        opt.learning_rate = 0.01

# set the path according to the environment
if hostname.startswith('visiongpu'):
    opt.model_path = '/path/to/my/student_model'
    opt.tb_path = '/path/to/my/student_tensorboards'
else:
    opt.model_path = './save/student_model'
    opt.tb_path = './save/student_tensorboards'

iterations = opt.lr_decay_epochs.split(',')
opt.lr_decay_epochs = list([])
for it in iterations:
    opt.lr_decay_epochs.append(int(it))

# @mst: For easier scale of epochs like 2xtime, 2.5xtime
if opt.epoch_factor:
    opt.lr_decay_epochs = [
        int(x * opt.epoch_factor) for x in opt.lr_decay_epochs
    ]
    opt.epochs = int(opt.epochs * opt.epoch_factor)

if opt.dataset in ['cifar100', 'svhn', 'tinyimagenet'
                   ]:  # For ImageNet, we use preset opt.model_t
    opt.model_t = get_teacher_name(opt.path_t)

if opt.no_DA:  # Maintain back-compatibility
    opt.use_DA = '00'

if opt.mix_mode == 'cutmix_pick':
    assert opt.mix_n_run >= 2

opt = update_args(opt)
