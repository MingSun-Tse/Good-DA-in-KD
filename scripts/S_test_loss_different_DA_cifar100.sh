############################################
# vgg13/vgg8
python train_student.py --path_t ./save/models/vgg13_vanilla/ckpt_epoch_240.pth --distill kd --model_s vgg8 -r 0.1 -a 0.9 -b 0 --t_output_as_target_for_input_mix --lw_mix [1,0,1] --mix_mode identity --project kd__vgg13vgg8__cifar100__identity 

python train_student.py --path_t ./save/models/vgg13_vanilla/ckpt_epoch_240.pth --distill kd --model_s vgg8 -r 0.1 -a 0.9 -b 0 --t_output_as_target_for_input_mix --lw_mix [1,0,1] --mix_mode flip --project kd__vgg13vgg8__cifar100__flip

python train_student.py --path_t ./save/models/vgg13_vanilla/ckpt_epoch_240.pth --distill kd --model_s vgg8 -r 0.1 -a 0.9 -b 0 --t_output_as_target_for_input_mix --lw_mix [1,0,1] --mix_mode crop+flip --project kd__vgg13vgg8__cifar100__cropflip

python train_student.py --path_t ./save/models/vgg13_vanilla/ckpt_epoch_240.pth --distill kd --model_s vgg8 -r 0.1 -a 0.9 -b 0 --t_output_as_target_for_input_mix --lw_mix [1,0,1] --mix_mode cutout --project kd__vgg13vgg8__cifar100__cutout

python train_student.py --path_t ./save/models/vgg13_vanilla/ckpt_epoch_240.pth --distill kd --model_s vgg8 -r 0.1 -a 0.9 -b 0 --t_output_as_target_for_input_mix --lw_mix [1,0,1] --mix_mode autoaugment --project kd__vgg13vgg8__cifar100__autoaugment

python train_student.py --path_t ./save/models/vgg13_vanilla/ckpt_epoch_240.pth --distill kd --model_s vgg8 -r 0.1 -a 0.9 -b 0 --t_output_as_target_for_input_mix --lw_mix [1,0,1] --mix_mode mixup --project kd__vgg13vgg8__cifar100__mixup

python train_student.py --path_t ./save/models/vgg13_vanilla/ckpt_epoch_240.pth --distill kd --model_s vgg8 -r 0.1 -a 0.9 -b 0 --t_output_as_target_for_input_mix --lw_mix [1,0,1] --mix_mode cutmix --project kd__vgg13vgg8__cifar100__cutmix

python train_student.py --path_t ./save/models/vgg13_vanilla/ckpt_epoch_240.pth --distill kd --model_s vgg8 -r 0.1 -a 0.9 -b 0 --t_output_as_target_for_input_mix --lw_mix [1,0,1] --mix_mode cutmix_pick --mix_n_run 2 --cutmix_pick_criterion student_entropy --project kd__vgg13vgg8__cifar100__cutmix_pick_Sentropy

python train_student.py --path_t ./save/models/vgg13_vanilla/ckpt_epoch_240.pth --distill kd --model_s vgg8 -r 0.1 -a 0.9 -b 0 --t_output_as_target_for_input_mix --lw_mix [1,0,1] --mix_mode cutmix_pick --mix_n_run 2 --cutmix_pick_criterion teacher_entropy --project kd__vgg13vgg8__cifar100__cutmix_pick_Tentropy


############################################
# vgg13/MobileNetV2
python train_student.py --path_t ./save/models/vgg13_vanilla/ckpt_epoch_240.pth --distill kd --model_s MobileNetV2 -r 0.1 -a 0.9 -b 0 --t_output_as_target_for_input_mix --lw_mix [1,0,1] --mix_mode identity --project kd__vgg13MobileNetV2__cifar100__identity 

python train_student.py --path_t ./save/models/vgg13_vanilla/ckpt_epoch_240.pth --distill kd --model_s MobileNetV2 -r 0.1 -a 0.9 -b 0 --t_output_as_target_for_input_mix --lw_mix [1,0,1] --mix_mode flip --project kd__vgg13MobileNetV2__cifar100__flip

python train_student.py --path_t ./save/models/vgg13_vanilla/ckpt_epoch_240.pth --distill kd --model_s MobileNetV2 -r 0.1 -a 0.9 -b 0 --t_output_as_target_for_input_mix --lw_mix [1,0,1] --mix_mode crop+flip --project kd__vgg13MobileNetV2__cifar100__cropflip

python train_student.py --path_t ./save/models/vgg13_vanilla/ckpt_epoch_240.pth --distill kd --model_s MobileNetV2 -r 0.1 -a 0.9 -b 0 --t_output_as_target_for_input_mix --lw_mix [1,0,1] --mix_mode cutout --project kd__vgg13MobileNetV2__cifar100__cutout

python train_student.py --path_t ./save/models/vgg13_vanilla/ckpt_epoch_240.pth --distill kd --model_s MobileNetV2 -r 0.1 -a 0.9 -b 0 --t_output_as_target_for_input_mix --lw_mix [1,0,1] --mix_mode autoaugment --project kd__vgg13MobileNetV2__cifar100__autoaugment

python train_student.py --path_t ./save/models/vgg13_vanilla/ckpt_epoch_240.pth --distill kd --model_s MobileNetV2 -r 0.1 -a 0.9 -b 0 --t_output_as_target_for_input_mix --lw_mix [1,0,1] --mix_mode mixup --project kd__vgg13MobileNetV2__cifar100__mixup

python train_student.py --path_t ./save/models/vgg13_vanilla/ckpt_epoch_240.pth --distill kd --model_s MobileNetV2 -r 0.1 -a 0.9 -b 0 --t_output_as_target_for_input_mix --lw_mix [1,0,1] --mix_mode cutmix --project kd__vgg13MobileNetV2__cifar100__cutmix

python train_student.py --path_t ./save/models/vgg13_vanilla/ckpt_epoch_240.pth --distill kd --model_s MobileNetV2 -r 0.1 -a 0.9 -b 0 --t_output_as_target_for_input_mix --lw_mix [1,0,1] --mix_mode cutmix_pick --mix_n_run 2 --cutmix_pick_criterion student_entropy --project kd__vgg13MobileNetV2__cifar100__cutmix_pick_Sentropy

python train_student.py --path_t ./save/models/vgg13_vanilla/ckpt_epoch_240.pth --distill kd --model_s MobileNetV2 -r 0.1 -a 0.9 -b 0 --t_output_as_target_for_input_mix --lw_mix [1,0,1] --mix_mode cutmix_pick --mix_n_run 2 --cutmix_pick_criterion teacher_entropy --project kd__vgg13MobileNetV2__cifar100__cutmix_pick_Tentropy



############################################
# resnet56/resnet20
python train_student.py --path_t ./save/models/resnet56_vanilla/ckpt_epoch_240.pth --distill kd --model_s resnet20 -r 0.1 -a 0.9 -b 0 --t_output_as_target_for_input_mix --lw_mix [1,0,1] --mix_mode identity --project kd__resnet56resnet20__cifar100__identity 

python train_student.py --path_t ./save/models/resnet56_vanilla/ckpt_epoch_240.pth --distill kd --model_s resnet20 -r 0.1 -a 0.9 -b 0 --t_output_as_target_for_input_mix --lw_mix [1,0,1] --mix_mode flip --project kd__resnet56resnet20__cifar100__flip

python train_student.py --path_t ./save/models/resnet56_vanilla/ckpt_epoch_240.pth --distill kd --model_s resnet20 -r 0.1 -a 0.9 -b 0 --t_output_as_target_for_input_mix --lw_mix [1,0,1] --mix_mode crop+flip --project kd__resnet56resnet20__cifar100__cropflip

python train_student.py --path_t ./save/models/resnet56_vanilla/ckpt_epoch_240.pth --distill kd --model_s resnet20 -r 0.1 -a 0.9 -b 0 --t_output_as_target_for_input_mix --lw_mix [1,0,1] --mix_mode cutout --project kd__resnet56resnet20__cifar100__cutout

python train_student.py --path_t ./save/models/resnet56_vanilla/ckpt_epoch_240.pth --distill kd --model_s resnet20 -r 0.1 -a 0.9 -b 0 --t_output_as_target_for_input_mix --lw_mix [1,0,1] --mix_mode autoaugment --project kd__resnet56resnet20__cifar100__autoaugment

python train_student.py --path_t ./save/models/resnet56_vanilla/ckpt_epoch_240.pth --distill kd --model_s resnet20 -r 0.1 -a 0.9 -b 0 --t_output_as_target_for_input_mix --lw_mix [1,0,1] --mix_mode mixup --project kd__resnet56resnet20__cifar100__mixup

python train_student.py --path_t ./save/models/resnet56_vanilla/ckpt_epoch_240.pth --distill kd --model_s resnet20 -r 0.1 -a 0.9 -b 0 --t_output_as_target_for_input_mix --lw_mix [1,0,1] --mix_mode cutmix --project kd__resnet56resnet20__cifar100__cutmix

python train_student.py --path_t ./save/models/resnet56_vanilla/ckpt_epoch_240.pth --distill kd --model_s resnet20 -r 0.1 -a 0.9 -b 0 --t_output_as_target_for_input_mix --lw_mix [1,0,1] --mix_mode cutmix_pick --mix_n_run 2 --cutmix_pick_criterion student_entropy --project kd__resnet56resnet20__cifar100__cutmix_pick_Sentropy

python train_student.py --path_t ./save/models/resnet56_vanilla/ckpt_epoch_240.pth --distill kd --model_s resnet20 -r 0.1 -a 0.9 -b 0 --t_output_as_target_for_input_mix --lw_mix [1,0,1] --mix_mode cutmix_pick --mix_n_run 2 --cutmix_pick_criterion teacher_entropy --project kd__resnet56resnet20__cifar100__cutmix_pick_Tentropy

############################################
# resnet56/ShuffleV2
python train_student.py --path_t ./save/models/resnet56_vanilla/ckpt_epoch_240.pth --distill kd --model_s ShuffleV2 -r 0.1 -a 0.9 -b 0 --t_output_as_target_for_input_mix --lw_mix [1,0,1] --mix_mode identity --project kd__resnet56ShuffleV2__cifar100__identity 

python train_student.py --path_t ./save/models/resnet56_vanilla/ckpt_epoch_240.pth --distill kd --model_s ShuffleV2 -r 0.1 -a 0.9 -b 0 --t_output_as_target_for_input_mix --lw_mix [1,0,1] --mix_mode flip --project kd__resnet56ShuffleV2__cifar100__flip

python train_student.py --path_t ./save/models/resnet56_vanilla/ckpt_epoch_240.pth --distill kd --model_s ShuffleV2 -r 0.1 -a 0.9 -b 0 --t_output_as_target_for_input_mix --lw_mix [1,0,1] --mix_mode crop+flip --project kd__resnet56ShuffleV2__cifar100__cropflip

python train_student.py --path_t ./save/models/resnet56_vanilla/ckpt_epoch_240.pth --distill kd --model_s ShuffleV2 -r 0.1 -a 0.9 -b 0 --t_output_as_target_for_input_mix --lw_mix [1,0,1] --mix_mode cutout --project kd__resnet56ShuffleV2__cifar100__cutout

python train_student.py --path_t ./save/models/resnet56_vanilla/ckpt_epoch_240.pth --distill kd --model_s ShuffleV2 -r 0.1 -a 0.9 -b 0 --t_output_as_target_for_input_mix --lw_mix [1,0,1] --mix_mode autoaugment --project kd__resnet56ShuffleV2__cifar100__autoaugment

python train_student.py --path_t ./save/models/resnet56_vanilla/ckpt_epoch_240.pth --distill kd --model_s ShuffleV2 -r 0.1 -a 0.9 -b 0 --t_output_as_target_for_input_mix --lw_mix [1,0,1] --mix_mode mixup --project kd__resnet56ShuffleV2__cifar100__mixup

python train_student.py --path_t ./save/models/resnet56_vanilla/ckpt_epoch_240.pth --distill kd --model_s ShuffleV2 -r 0.1 -a 0.9 -b 0 --t_output_as_target_for_input_mix --lw_mix [1,0,1] --mix_mode cutmix --project kd__resnet56ShuffleV2__cifar100__cutmix

python train_student.py --path_t ./save/models/resnet56_vanilla/ckpt_epoch_240.pth --distill kd --model_s ShuffleV2 -r 0.1 -a 0.9 -b 0 --t_output_as_target_for_input_mix --lw_mix [1,0,1] --mix_mode cutmix_pick --mix_n_run 2 --cutmix_pick_criterion student_entropy --project kd__resnet56ShuffleV2__cifar100__cutmix_pick_Sentropy 

python train_student.py --path_t ./save/models/resnet56_vanilla/ckpt_epoch_240.pth --distill kd --model_s ShuffleV2 -r 0.1 -a 0.9 -b 0 --t_output_as_target_for_input_mix --lw_mix [1,0,1] --mix_mode cutmix_pick --mix_n_run 2 --cutmix_pick_criterion teacher_entropy --project kd__resnet56ShuffleV2__cifar100__cutmix_pick_Tentropy

############################################
# wrn_40_2/wrn_16_2
python train_student.py --path_t ./save/models/wrn_40_2_vanilla/ckpt_epoch_240.pth --distill kd --model_s wrn_16_2 -r 0.1 -a 0.9 -b 0 --t_output_as_target_for_input_mix --lw_mix [1,0,1] --mix_mode identity --project kd__wrn_40_2wrn_16_2__cifar100__identity 

python train_student.py --path_t ./save/models/wrn_40_2_vanilla/ckpt_epoch_240.pth --distill kd --model_s wrn_16_2 -r 0.1 -a 0.9 -b 0 --t_output_as_target_for_input_mix --lw_mix [1,0,1] --mix_mode flip --project kd__wrn_40_2wrn_16_2__cifar100__flip

python train_student.py --path_t ./save/models/wrn_40_2_vanilla/ckpt_epoch_240.pth --distill kd --model_s wrn_16_2 -r 0.1 -a 0.9 -b 0 --t_output_as_target_for_input_mix --lw_mix [1,0,1] --mix_mode crop+flip --project kd__wrn_40_2wrn_16_2__cifar100__cropflip

python train_student.py --path_t ./save/models/wrn_40_2_vanilla/ckpt_epoch_240.pth --distill kd --model_s wrn_16_2 -r 0.1 -a 0.9 -b 0 --t_output_as_target_for_input_mix --lw_mix [1,0,1] --mix_mode cutout --project kd__wrn_40_2wrn_16_2__cifar100__cutout

python train_student.py --path_t ./save/models/wrn_40_2_vanilla/ckpt_epoch_240.pth --distill kd --model_s wrn_16_2 -r 0.1 -a 0.9 -b 0 --t_output_as_target_for_input_mix --lw_mix [1,0,1] --mix_mode autoaugment --project kd__wrn_40_2wrn_16_2__cifar100__autoaugment

python train_student.py --path_t ./save/models/wrn_40_2_vanilla/ckpt_epoch_240.pth --distill kd --model_s wrn_16_2 -r 0.1 -a 0.9 -b 0 --t_output_as_target_for_input_mix --lw_mix [1,0,1] --mix_mode mixup --project kd__wrn_40_2wrn_16_2__cifar100__mixup

python train_student.py --path_t ./save/models/wrn_40_2_vanilla/ckpt_epoch_240.pth --distill kd --model_s wrn_16_2 -r 0.1 -a 0.9 -b 0 --t_output_as_target_for_input_mix --lw_mix [1,0,1] --mix_mode cutmix --project kd__wrn_40_2wrn_16_2__cifar100__cutmix

python train_student.py --path_t ./save/models/wrn_40_2_vanilla/ckpt_epoch_240.pth --distill kd --model_s wrn_16_2 -r 0.1 -a 0.9 -b 0 --t_output_as_target_for_input_mix --lw_mix [1,0,1] --mix_mode cutmix_pick --mix_n_run 2 --cutmix_pick_criterion student_entropy --project kd__wrn_40_2wrn_16_2__cifar100__cutmix_pick_Sentropy 

python train_student.py --path_t ./save/models/wrn_40_2_vanilla/ckpt_epoch_240.pth --distill kd --model_s wrn_16_2 -r 0.1 -a 0.9 -b 0 --t_output_as_target_for_input_mix --lw_mix [1,0,1] --mix_mode cutmix_pick --mix_n_run 2 --cutmix_pick_criterion teacher_entropy --project kd__wrn_40_2wrn_16_2__cifar100__cutmix_pick_Tentropy

############################################
# wrn_40_2/vgg8
python train_student.py --path_t ./save/models/wrn_40_2_vanilla/ckpt_epoch_240.pth --distill kd --model_s vgg8 -r 0.1 -a 0.9 -b 0 --t_output_as_target_for_input_mix --lw_mix [1,0,1] --mix_mode identity --project kd__wrn_40_2vgg8__cifar100__identity 

python train_student.py --path_t ./save/models/wrn_40_2_vanilla/ckpt_epoch_240.pth --distill kd --model_s vgg8 -r 0.1 -a 0.9 -b 0 --t_output_as_target_for_input_mix --lw_mix [1,0,1] --mix_mode flip --project kd__wrn_40_2vgg8__cifar100__flip

python train_student.py --path_t ./save/models/wrn_40_2_vanilla/ckpt_epoch_240.pth --distill kd --model_s vgg8 -r 0.1 -a 0.9 -b 0 --t_output_as_target_for_input_mix --lw_mix [1,0,1] --mix_mode crop+flip --project kd__wrn_40_2vgg8__cifar100__cropflip

python train_student.py --path_t ./save/models/wrn_40_2_vanilla/ckpt_epoch_240.pth --distill kd --model_s vgg8 -r 0.1 -a 0.9 -b 0 --t_output_as_target_for_input_mix --lw_mix [1,0,1] --mix_mode cutout --project kd__wrn_40_2vgg8__cifar100__cutout

python train_student.py --path_t ./save/models/wrn_40_2_vanilla/ckpt_epoch_240.pth --distill kd --model_s vgg8 -r 0.1 -a 0.9 -b 0 --t_output_as_target_for_input_mix --lw_mix [1,0,1] --mix_mode autoaugment --project kd__wrn_40_2vgg8__cifar100__autoaugment

python train_student.py --path_t ./save/models/wrn_40_2_vanilla/ckpt_epoch_240.pth --distill kd --model_s vgg8 -r 0.1 -a 0.9 -b 0 --t_output_as_target_for_input_mix --lw_mix [1,0,1] --mix_mode mixup --project kd__wrn_40_2vgg8__cifar100__mixup

python train_student.py --path_t ./save/models/wrn_40_2_vanilla/ckpt_epoch_240.pth --distill kd --model_s vgg8 -r 0.1 -a 0.9 -b 0 --t_output_as_target_for_input_mix --lw_mix [1,0,1] --mix_mode cutmix --project kd__wrn_40_2vgg8__cifar100__cutmix

python train_student.py --path_t ./save/models/wrn_40_2_vanilla/ckpt_epoch_240.pth --distill kd --model_s vgg8 -r 0.1 -a 0.9 -b 0 --t_output_as_target_for_input_mix --lw_mix [1,0,1] --mix_mode cutmix_pick --mix_n_run 2 --cutmix_pick_criterion student_entropy --project kd__wrn_40_2vgg8__cifar100__cutmix_pick_Sentropy

python train_student.py --path_t ./save/models/wrn_40_2_vanilla/ckpt_epoch_240.pth --distill kd --model_s vgg8 -r 0.1 -a 0.9 -b 0 --t_output_as_target_for_input_mix --lw_mix [1,0,1] --mix_mode cutmix_pick --mix_n_run 2 --cutmix_pick_criterion teacher_entropy --project kd__wrn_40_2vgg8__cifar100__cutmix_pick_Tentropy


############################################
# resnet32x4/resnet8x4
python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill kd --model_s resnet8x4 -r 0.1 -a 0.9 -b 0 --t_output_as_target_for_input_mix --lw_mix [1,0,1] --mix_mode identity --project kd__resnet32x4resnet8x4__cifar100__identity 

python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill kd --model_s resnet8x4 -r 0.1 -a 0.9 -b 0 --t_output_as_target_for_input_mix --lw_mix [1,0,1] --mix_mode flip --project kd__resnet32x4resnet8x4__cifar100__flip

python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill kd --model_s resnet8x4 -r 0.1 -a 0.9 -b 0 --t_output_as_target_for_input_mix --lw_mix [1,0,1] --mix_mode crop+flip --project kd__resnet32x4resnet8x4__cifar100__cropflip

python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill kd --model_s resnet8x4 -r 0.1 -a 0.9 -b 0 --t_output_as_target_for_input_mix --lw_mix [1,0,1] --mix_mode cutout --project kd__resnet32x4resnet8x4__cifar100__cutout

python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill kd --model_s resnet8x4 -r 0.1 -a 0.9 -b 0 --t_output_as_target_for_input_mix --lw_mix [1,0,1] --mix_mode autoaugment --project kd__resnet32x4resnet8x4__cifar100__autoaugment

python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill kd --model_s resnet8x4 -r 0.1 -a 0.9 -b 0 --t_output_as_target_for_input_mix --lw_mix [1,0,1] --mix_mode mixup --project kd__resnet32x4resnet8x4__cifar100__mixup

python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill kd --model_s resnet8x4 -r 0.1 -a 0.9 -b 0 --t_output_as_target_for_input_mix --lw_mix [1,0,1] --mix_mode cutmix --project kd__resnet32x4resnet8x4__cifar100__cutmix

python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill kd --model_s resnet8x4 -r 0.1 -a 0.9 -b 0 --t_output_as_target_for_input_mix --lw_mix [1,0,1] --mix_mode cutmix_pick --mix_n_run 2 --cutmix_pick_criterion student_entropy --project kd__resnet32x4resnet8x4__cifar100__cutmix_pick_Sentropy

python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill kd --model_s resnet8x4 -r 0.1 -a 0.9 -b 0 --t_output_as_target_for_input_mix --lw_mix [1,0,1] --mix_mode cutmix_pick --mix_n_run 2 --cutmix_pick_criterion teacher_entropy --project kd__resnet32x4resnet8x4__cifar100__cutmix_pick_Tentropy



############################################
# ResNet50/vgg8
python train_student.py --path_t ./save/models/ResNet50_vanilla/ckpt_epoch_240.pth --distill kd --model_s vgg8 -r 0.1 -a 0.9 -b 0 --t_output_as_target_for_input_mix --lw_mix [1,0,1] --mix_mode identity --project kd__ResNet50vgg8__cifar100__identity 

python train_student.py --path_t ./save/models/ResNet50_vanilla/ckpt_epoch_240.pth --distill kd --model_s vgg8 -r 0.1 -a 0.9 -b 0 --t_output_as_target_for_input_mix --lw_mix [1,0,1] --mix_mode flip --project kd__ResNet50vgg8__cifar100__flip

python train_student.py --path_t ./save/models/ResNet50_vanilla/ckpt_epoch_240.pth --distill kd --model_s vgg8 -r 0.1 -a 0.9 -b 0 --t_output_as_target_for_input_mix --lw_mix [1,0,1] --mix_mode crop+flip --project kd__ResNet50vgg8__cifar100__cropflip

python train_student.py --path_t ./save/models/ResNet50_vanilla/ckpt_epoch_240.pth --distill kd --model_s vgg8 -r 0.1 -a 0.9 -b 0 --t_output_as_target_for_input_mix --lw_mix [1,0,1] --mix_mode cutout --project kd__ResNet50vgg8__cifar100__cutout

python train_student.py --path_t ./save/models/ResNet50_vanilla/ckpt_epoch_240.pth --distill kd --model_s vgg8 -r 0.1 -a 0.9 -b 0 --t_output_as_target_for_input_mix --lw_mix [1,0,1] --mix_mode autoaugment --project kd__ResNet50vgg8__cifar100__autoaugment

python train_student.py --path_t ./save/models/ResNet50_vanilla/ckpt_epoch_240.pth --distill kd --model_s vgg8 -r 0.1 -a 0.9 -b 0 --t_output_as_target_for_input_mix --lw_mix [1,0,1] --mix_mode mixup --project kd__ResNet50vgg8__cifar100__mixup

python train_student.py --path_t ./save/models/ResNet50_vanilla/ckpt_epoch_240.pth --distill kd --model_s vgg8 -r 0.1 -a 0.9 -b 0 --t_output_as_target_for_input_mix --lw_mix [1,0,1] --mix_mode cutmix --project kd__ResNet50vgg8__cifar100__cutmix

python train_student.py --path_t ./save/models/ResNet50_vanilla/ckpt_epoch_240.pth --distill kd --model_s vgg8 -r 0.1 -a 0.9 -b 0 --t_output_as_target_for_input_mix --lw_mix [1,0,1] --mix_mode cutmix_pick --mix_n_run 2 --cutmix_pick_criterion student_entropy --project kd__ResNet50vgg8__cifar100__cutmix_pick_Sentropy

python train_student.py --path_t ./save/models/ResNet50_vanilla/ckpt_epoch_240.pth --distill kd --model_s vgg8 -r 0.1 -a 0.9 -b 0 --t_output_as_target_for_input_mix --lw_mix [1,0,1] --mix_mode cutmix_pick --mix_n_run 2 --cutmix_pick_criterion teacher_entropy --project kd__ResNet50vgg8__cifar100__cutmix_pick_Tentropy



############################################
# resnet32x4/ShuffleV2
python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill kd --model_s ShuffleV2 -r 0.1 -a 0.9 -b 0 --t_output_as_target_for_input_mix --lw_mix [1,0,1] --mix_mode identity --project kd__resnet32x4ShuffleV2__cifar100__identity 1

python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill kd --model_s ShuffleV2 -r 0.1 -a 0.9 -b 0 --t_output_as_target_for_input_mix --lw_mix [1,0,1] --mix_mode flip --project kd__resnet32x4ShuffleV2__cifar100__flip 1

python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill kd --model_s ShuffleV2 -r 0.1 -a 0.9 -b 0 --t_output_as_target_for_input_mix --lw_mix [1,0,1] --mix_mode crop+flip --project kd__resnet32x4ShuffleV2__cifar100__cropflip

python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill kd --model_s ShuffleV2 -r 0.1 -a 0.9 -b 0 --t_output_as_target_for_input_mix --lw_mix [1,0,1] --mix_mode cutout --project kd__resnet32x4ShuffleV2__cifar100__cutout

python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill kd --model_s ShuffleV2 -r 0.1 -a 0.9 -b 0 --t_output_as_target_for_input_mix --lw_mix [1,0,1] --mix_mode autoaugment --project kd__resnet32x4ShuffleV2__cifar100__autoaugment 1 

python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill kd --model_s ShuffleV2 -r 0.1 -a 0.9 -b 0 --t_output_as_target_for_input_mix --lw_mix [1,0,1] --mix_mode mixup --project kd__resnet32x4ShuffleV2__cifar100__mixup 1

python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill kd --model_s ShuffleV2 -r 0.1 -a 0.9 -b 0 --t_output_as_target_for_input_mix --lw_mix [1,0,1] --mix_mode cutmix --project kd__resnet32x4ShuffleV2__cifar100__cutmix

python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill kd --model_s ShuffleV2 -r 0.1 -a 0.9 -b 0 --t_output_as_target_for_input_mix --lw_mix [1,0,1] --mix_mode cutmix_pick --mix_n_run 2 --cutmix_pick_criterion student_entropy --project kd__resnet32x4ShuffleV2__cifar100__cutmix_pick_Sentropy 1

python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill kd --model_s ShuffleV2 -r 0.1 -a 0.9 -b 0 --t_output_as_target_for_input_mix --lw_mix [1,0,1] --mix_mode cutmix_pick --mix_n_run 2 --cutmix_pick_criterion teacher_entropy --project kd__resnet32x4ShuffleV2__cifar100__cutmix_pick_Tentropy