############################################
# vgg13
python train_student.py --path_t ./save/models/vgg13_vanilla/ckpt_epoch_240.pth --distill kd --model_s vgg8 -r 0.1 -a 0.9 -b 0 --t_output_as_target_for_input_mix --lw_mix [1,0,1] --learning_rate 0 --fix_student --utils.ON --utils.check_ce_var --epochs 10 --mix_mode identity --project kd__vgg13vgg8__cifar100__CheckTProbStd_identity 

python train_student.py --path_t ./save/models/vgg13_vanilla/ckpt_epoch_240.pth --distill kd --model_s vgg8 -r 0.1 -a 0.9 -b 0 --t_output_as_target_for_input_mix --lw_mix [1,0,1] --learning_rate 0 --fix_student --utils.ON --utils.check_ce_var --epochs 10 --mix_mode flip --project kd__vgg13vgg8__cifar100__CheckTProbStd_flip

python train_student.py --path_t ./save/models/vgg13_vanilla/ckpt_epoch_240.pth --distill kd --model_s vgg8 -r 0.1 -a 0.9 -b 0 --t_output_as_target_for_input_mix --lw_mix [1,0,1] --learning_rate 0 --fix_student --utils.ON --utils.check_ce_var --epochs 10 --mix_mode crop+flip --project kd__vgg13vgg8__cifar100__CheckTProbStd_cropflip

python train_student.py --path_t ./save/models/vgg13_vanilla/ckpt_epoch_240.pth --distill kd --model_s vgg8 -r 0.1 -a 0.9 -b 0 --t_output_as_target_for_input_mix --lw_mix [1,0,1] --learning_rate 0 --fix_student --utils.ON --utils.check_ce_var --epochs 10 --mix_mode cutout --project kd__vgg13vgg8__cifar100__CheckTProbStd_cutout

python train_student.py --path_t ./save/models/vgg13_vanilla/ckpt_epoch_240.pth --distill kd --model_s vgg8 -r 0.1 -a 0.9 -b 0 --t_output_as_target_for_input_mix --lw_mix [1,0,1] --learning_rate 0 --fix_student --utils.ON --utils.check_ce_var --epochs 10 --mix_mode autoaugment --project kd__vgg13vgg8__cifar100__CheckTProbStd_autoaugment

python train_student.py --path_t ./save/models/vgg13_vanilla/ckpt_epoch_240.pth --distill kd --model_s vgg8 -r 0.1 -a 0.9 -b 0 --t_output_as_target_for_input_mix --lw_mix [1,0,1] --learning_rate 0 --fix_student --utils.ON --utils.check_ce_var --epochs 10 --mix_mode mixup --project kd__vgg13vgg8__cifar100__CheckTProbStd_mixup

python train_student.py --path_t ./save/models/vgg13_vanilla/ckpt_epoch_240.pth --distill kd --model_s vgg8 -r 0.1 -a 0.9 -b 0 --t_output_as_target_for_input_mix --lw_mix [1,0,1] --learning_rate 0 --fix_student --utils.ON --utils.check_ce_var --epochs 10 --mix_mode cutmix --project kd__vgg13vgg8__cifar100__CheckTProbStd_cutmix

python train_student.py --path_t ./save/models/vgg13_vanilla/ckpt_epoch_240.pth --distill kd --model_s vgg8 -r 0.1 -a 0.9 -b 0 --t_output_as_target_for_input_mix --lw_mix [1,0,1] --finetune_student Experiments/*-141344/weights/ckpt.pth --learning_rate 0 --fix_student --utils.ON --utils.check_ce_var --epochs 10 --mix_mode cutmix_pick --mix_n_run 2 --cutmix_pick_criterion student_entropy --project kd__vgg13vgg8__cifar100__CheckTProbStd_cutmix_pick_Sentropy

python train_student.py --path_t ./save/models/vgg13_vanilla/ckpt_epoch_240.pth --distill kd --model_s vgg8 -r 0.1 -a 0.9 -b 0 --t_output_as_target_for_input_mix --lw_mix [1,0,1] --finetune_student Experiments/*-141344/weights/ckpt.pth --learning_rate 0 --fix_student --utils.ON --utils.check_ce_var --epochs 10 --mix_mode cutmix_pick --mix_n_run 2 --cutmix_pick_criterion teacher_entropy --project kd__vgg13vgg8__cifar100__CheckTProbStd_cutmix_pick_Tentropy

############################################
# resnet56
python train_student.py --path_t ./save/models/resnet56_vanilla/ckpt_epoch_240.pth --distill kd --model_s resnet20 -r 0.1 -a 0.9 -b 0 --t_output_as_target_for_input_mix --lw_mix [1,0,1] --learning_rate 0 --fix_student --utils.ON --utils.check_ce_var --epochs 10 --mix_mode identity --project kd__resnet56resnet20__cifar100__CheckTProbStd_identity

python train_student.py --path_t ./save/models/resnet56_vanilla/ckpt_epoch_240.pth --distill kd --model_s resnet20 -r 0.1 -a 0.9 -b 0 --t_output_as_target_for_input_mix --lw_mix [1,0,1] --learning_rate 0 --fix_student --utils.ON --utils.check_ce_var --epochs 10 --mix_mode flip --project kd__resnet56resnet20__cifar100__CheckTProbStd_flip

python train_student.py --path_t ./save/models/resnet56_vanilla/ckpt_epoch_240.pth --distill kd --model_s resnet20 -r 0.1 -a 0.9 -b 0 --t_output_as_target_for_input_mix --lw_mix [1,0,1] --learning_rate 0 --fix_student --utils.ON --utils.check_ce_var --epochs 10 --mix_mode crop+flip --project kd__resnet56resnet20__cifar100__CheckTProbStd_cropflip

python train_student.py --path_t ./save/models/resnet56_vanilla/ckpt_epoch_240.pth --distill kd --model_s resnet20 -r 0.1 -a 0.9 -b 0 --t_output_as_target_for_input_mix --lw_mix [1,0,1] --learning_rate 0 --fix_student --utils.ON --utils.check_ce_var --epochs 10 --mix_mode cutout --project kd__resnet56resnet20__cifar100__CheckTProbStd_cutout

python train_student.py --path_t ./save/models/resnet56_vanilla/ckpt_epoch_240.pth --distill kd --model_s resnet20 -r 0.1 -a 0.9 -b 0 --t_output_as_target_for_input_mix --lw_mix [1,0,1] --learning_rate 0 --fix_student --utils.ON --utils.check_ce_var --epochs 10 --mix_mode autoaugment --project kd__resnet56resnet20__cifar100__CheckTProbStd_autoaugment

python train_student.py --path_t ./save/models/resnet56_vanilla/ckpt_epoch_240.pth --distill kd --model_s resnet20 -r 0.1 -a 0.9 -b 0 --t_output_as_target_for_input_mix --lw_mix [1,0,1] --learning_rate 0 --fix_student --utils.ON --utils.check_ce_var --epochs 10 --mix_mode mixup --project kd__resnet56resnet20__cifar100__CheckTProbStd_mixup

python train_student.py --path_t ./save/models/resnet56_vanilla/ckpt_epoch_240.pth --distill kd --model_s resnet20 -r 0.1 -a 0.9 -b 0 --t_output_as_target_for_input_mix --lw_mix [1,0,1] --learning_rate 0 --fix_student --utils.ON --utils.check_ce_var --epochs 10 --mix_mode cutmix --project kd__resnet56resnet20__cifar100__CheckTProbStd_cutmix

python train_student.py --path_t ./save/models/resnet56_vanilla/ckpt_epoch_240.pth --distill kd --model_s resnet20 -r 0.1 -a 0.9 -b 0 --t_output_as_target_for_input_mix --lw_mix [1,0,1] --finetune_student Experiments/*-142440/weights/ckpt.pth --learning_rate 0 --fix_student --utils.ON --utils.check_ce_var --epochs 10 --mix_mode cutmix_pick --mix_n_run 2 --cutmix_pick_criterion student_entropy --project kd__resnet56resnet20__cifar100__CheckTProbStd_cutmix_pick_Sentropy

python train_student.py --path_t ./save/models/resnet56_vanilla/ckpt_epoch_240.pth --distill kd --model_s resnet20 -r 0.1 -a 0.9 -b 0 --t_output_as_target_for_input_mix --lw_mix [1,0,1] --finetune_student Experiments/*-142440/weights/ckpt.pth --learning_rate 0 --fix_student --utils.ON --utils.check_ce_var --epochs 10 --mix_mode cutmix_pick --mix_n_run 2 --cutmix_pick_criterion teacher_entropy --project kd__resnet56resnet20__cifar100__CheckTProbStd_cutmix_pick_Tentropy

############################################
# resnet32x4
python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill kd --model_s resnet8x4 -r 0.1 -a 0.9 -b 0 --learning_rate 0 --fix_student --utils.ON --utils.check_ce_var --epochs 10 --project kd__resnet32x4resnet8x4__cifar100__CheckTProbStd_none

python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill kd --model_s resnet8x4 -r 0.1 -a 0.9 -b 0 --t_output_as_target_for_input_mix --lw_mix [1,0,1] --learning_rate 0 --fix_student --utils.ON --utils.check_ce_var --epochs 10 --mix_mode identity --project kd__resnet32x4resnet8x4__cifar100__CheckTProbStd_identity 

python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill kd --model_s resnet8x4 -r 0.1 -a 0.9 -b 0 --t_output_as_target_for_input_mix --lw_mix [1,0,1] --learning_rate 0 --fix_student --utils.ON --utils.check_ce_var --epochs 10 --mix_mode crop --project kd__resnet32x4resnet8x4__cifar100__CheckTProbStd_crop

python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill kd --model_s resnet8x4 -r 0.1 -a 0.9 -b 0 --t_output_as_target_for_input_mix --lw_mix [1,0,1] --learning_rate 0 --fix_student --utils.ON --utils.check_ce_var --epochs 10 --mix_mode flip --project kd__resnet32x4resnet8x4__cifar100__CheckTProbStd_flip

python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill kd --model_s resnet8x4 -r 0.1 -a 0.9 -b 0 --t_output_as_target_for_input_mix --lw_mix [1,0,1] --learning_rate 0 --fix_student --utils.ON --utils.check_ce_var --epochs 10 --mix_mode crop+flip --project kd__resnet32x4resnet8x4__cifar100__CheckTProbStd_cropflip

python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill kd --model_s resnet8x4 -r 0.1 -a 0.9 -b 0 --t_output_as_target_for_input_mix --lw_mix [1,0,1] --learning_rate 0 --fix_student --utils.ON --utils.check_ce_var --epochs 10 --mix_mode cutout --project kd__resnet32x4resnet8x4__cifar100__CheckTProbStd_cutout

python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill kd --model_s resnet8x4 -r 0.1 -a 0.9 -b 0 --t_output_as_target_for_input_mix --lw_mix [1,0,1] --learning_rate 0 --fix_student --utils.ON --utils.check_ce_var --epochs 10 --mix_mode autoaugment --project kd__resnet32x4resnet8x4__cifar100__CheckTProbStd_autoaugment

python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill kd --model_s resnet8x4 -r 0.1 -a 0.9 -b 0 --t_output_as_target_for_input_mix --lw_mix [1,0,1] --learning_rate 0 --fix_student --utils.ON --utils.check_ce_var --epochs 10 --mix_mode mixup --project kd__resnet32x4resnet8x4__cifar100__CheckTProbStd_mixup

python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill kd --model_s resnet8x4 -r 0.1 -a 0.9 -b 0 --t_output_as_target_for_input_mix --lw_mix [1,0,1] --learning_rate 0 --fix_student --utils.ON --utils.check_ce_var --epochs 10 --mix_mode cutmix --project kd__resnet32x4resnet8x4__cifar100__CheckTProbStd_cutmix

python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill kd --model_s resnet8x4 -r 0.1 -a 0.9 -b 0 --t_output_as_target_for_input_mix --lw_mix [1,0,1] --finetune_student Experiments/kd-wrn*-141533/weights/ckpt.pth --learning_rate 0 --fix_student --utils.ON --utils.check_ce_var --epochs 10 --mix_mode cutmix_pick --mix_n_run 2 --cutmix_pick_criterion student_entropy --project kd__resnet32x4resnet8x4__cifar100__CheckTProbStd_cutmix_pick_Sentropy

python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill kd --model_s resnet8x4 -r 0.1 -a 0.9 -b 0 --t_output_as_target_for_input_mix --lw_mix [1,0,1] --finetune_student Experiments/kd-wrn*-141533/weights/ckpt.pth --learning_rate 0 --fix_student --utils.ON --utils.check_ce_var --epochs 10 --mix_mode cutmix_pick --mix_n_run 2 --cutmix_pick_criterion teacher_entropy --project kd__resnet32x4resnet8x4__cifar100__CheckTProbStd_cutmix_pick_Tentropy


############################################
# wrn_40_2
python train_student.py --path_t ./save/models/wrn_40_2_vanilla/ckpt_epoch_240.pth --distill kd --model_s wrn_16_2 -r 0.1 -a 0.9 -b 0 --learning_rate 0 --fix_student --utils.ON --utils.check_ce_var --epochs 10 --project kd__wrn_40_2wrn_16_2__cifar100__CheckTProbStd_none

python train_student.py --path_t ./save/models/wrn_40_2_vanilla/ckpt_epoch_240.pth --distill kd --model_s wrn_16_2 -r 0.1 -a 0.9 -b 0 --t_output_as_target_for_input_mix --lw_mix [1,0,1] --learning_rate 0 --fix_student --utils.ON --utils.check_ce_var --epochs 10 --mix_mode identity --project kd__wrn_40_2wrn_16_2__cifar100__CheckTProbStd_identity 

python train_student.py --path_t ./save/models/wrn_40_2_vanilla/ckpt_epoch_240.pth --distill kd --model_s wrn_16_2 -r 0.1 -a 0.9 -b 0 --t_output_as_target_for_input_mix --lw_mix [1,0,1] --learning_rate 0 --fix_student --utils.ON --utils.check_ce_var --epochs 10 --mix_mode crop --project kd__wrn_40_2wrn_16_2__cifar100__CheckTProbStd_crop

python train_student.py --path_t ./save/models/wrn_40_2_vanilla/ckpt_epoch_240.pth --distill kd --model_s wrn_16_2 -r 0.1 -a 0.9 -b 0 --t_output_as_target_for_input_mix --lw_mix [1,0,1] --learning_rate 0 --fix_student --utils.ON --utils.check_ce_var --epochs 10 --mix_mode flip --project kd__wrn_40_2wrn_16_2__cifar100__CheckTProbStd_flip

python train_student.py --path_t ./save/models/wrn_40_2_vanilla/ckpt_epoch_240.pth --distill kd --model_s wrn_16_2 -r 0.1 -a 0.9 -b 0 --t_output_as_target_for_input_mix --lw_mix [1,0,1] --learning_rate 0 --fix_student --utils.ON --utils.check_ce_var --epochs 10 --mix_mode crop+flip --project kd__wrn_40_2wrn_16_2__cifar100__CheckTProbStd_cropflip

python train_student.py --path_t ./save/models/wrn_40_2_vanilla/ckpt_epoch_240.pth --distill kd --model_s wrn_16_2 -r 0.1 -a 0.9 -b 0 --t_output_as_target_for_input_mix --lw_mix [1,0,1] --learning_rate 0 --fix_student --utils.ON --utils.check_ce_var --epochs 10 --mix_mode cutout --project kd__wrn_40_2wrn_16_2__cifar100__CheckTProbStd_cutout

python train_student.py --path_t ./save/models/wrn_40_2_vanilla/ckpt_epoch_240.pth --distill kd --model_s wrn_16_2 -r 0.1 -a 0.9 -b 0 --t_output_as_target_for_input_mix --lw_mix [1,0,1] --learning_rate 0 --fix_student --utils.ON --utils.check_ce_var --epochs 10 --mix_mode autoaugment --project kd__wrn_40_2wrn_16_2__cifar100__CheckTProbStd_autoaugment

python train_student.py --path_t ./save/models/wrn_40_2_vanilla/ckpt_epoch_240.pth --distill kd --model_s wrn_16_2 -r 0.1 -a 0.9 -b 0 --t_output_as_target_for_input_mix --lw_mix [1,0,1] --learning_rate 0 --fix_student --utils.ON --utils.check_ce_var --epochs 10 --mix_mode mixup --project kd__wrn_40_2wrn_16_2__cifar100__CheckTProbStd_mixup

python train_student.py --path_t ./save/models/wrn_40_2_vanilla/ckpt_epoch_240.pth --distill kd --model_s wrn_16_2 -r 0.1 -a 0.9 -b 0 --t_output_as_target_for_input_mix --lw_mix [1,0,1] --learning_rate 0 --fix_student --utils.ON --utils.check_ce_var --epochs 10 --mix_mode cutmix --project kd__wrn_40_2wrn_16_2__cifar100__CheckTProbStd_cutmix

python train_student.py --path_t ./save/models/wrn_40_2_vanilla/ckpt_epoch_240.pth --distill kd --model_s wrn_16_2 -r 0.1 -a 0.9 -b 0 --t_output_as_target_for_input_mix --lw_mix [1,0,1] --finetune_student Experiments/kd-wrn*-141533/weights/ckpt.pth --learning_rate 0 --fix_student --utils.ON --utils.check_ce_var --epochs 10 --mix_mode cutmix_pick --mix_n_run 2 --cutmix_pick_criterion student_entropy --project kd__wrn_40_2wrn_16_2__cifar100__CheckTProbStd_cutmix_pick_Sentropy

python train_student.py --path_t ./save/models/wrn_40_2_vanilla/ckpt_epoch_240.pth --distill kd --model_s wrn_16_2 -r 0.1 -a 0.9 -b 0 --t_output_as_target_for_input_mix --lw_mix [1,0,1] --finetune_student Experiments/kd-wrn*-141533/weights/ckpt.pth --learning_rate 0 --fix_student --utils.ON --utils.check_ce_var --epochs 10 --mix_mode cutmix_pick --mix_n_run 2 --cutmix_pick_criterion teacher_entropy --project kd__wrn_40_2wrn_16_2__cifar100__CheckTProbStd_cutmix_pick_Tentropy

############################################
# ResNet50
python train_student.py --path_t ./save/models/ResNet50_vanilla/ckpt_epoch_240.pth --distill kd --model_s vgg8 -r 0.1 -a 0.9 -b 0 --learning_rate 0 --fix_student --utils.ON --utils.check_ce_var --epochs 10 --project kd__ResNet50vgg8__cifar100__CheckTProbStd_none

python train_student.py --path_t ./save/models/ResNet50_vanilla/ckpt_epoch_240.pth --distill kd --model_s vgg8 -r 0.1 -a 0.9 -b 0 --t_output_as_target_for_input_mix --lw_mix [1,0,1] --learning_rate 0 --fix_student --utils.ON --utils.check_ce_var --epochs 10 --mix_mode identity --project kd__ResNet50vgg8__cifar100__CheckTProbStd_identity 

python train_student.py --path_t ./save/models/ResNet50_vanilla/ckpt_epoch_240.pth --distill kd --model_s vgg8 -r 0.1 -a 0.9 -b 0 --t_output_as_target_for_input_mix --lw_mix [1,0,1] --learning_rate 0 --fix_student --utils.ON --utils.check_ce_var --epochs 10 --mix_mode crop --project kd__ResNet50vgg8__cifar100__CheckTProbStd_crop

python train_student.py --path_t ./save/models/ResNet50_vanilla/ckpt_epoch_240.pth --distill kd --model_s vgg8 -r 0.1 -a 0.9 -b 0 --t_output_as_target_for_input_mix --lw_mix [1,0,1] --learning_rate 0 --fix_student --utils.ON --utils.check_ce_var --epochs 10 --mix_mode flip --project kd__ResNet50vgg8__cifar100__CheckTProbStd_flip

python train_student.py --path_t ./save/models/ResNet50_vanilla/ckpt_epoch_240.pth --distill kd --model_s vgg8 -r 0.1 -a 0.9 -b 0 --t_output_as_target_for_input_mix --lw_mix [1,0,1] --learning_rate 0 --fix_student --utils.ON --utils.check_ce_var --epochs 10 --mix_mode crop+flip --project kd__ResNet50vgg8__cifar100__CheckTProbStd_cropflip

python train_student.py --path_t ./save/models/ResNet50_vanilla/ckpt_epoch_240.pth --distill kd --model_s vgg8 -r 0.1 -a 0.9 -b 0 --t_output_as_target_for_input_mix --lw_mix [1,0,1] --learning_rate 0 --fix_student --utils.ON --utils.check_ce_var --epochs 10 --mix_mode cutout --project kd__ResNet50vgg8__cifar100__CheckTProbStd_cutout

python train_student.py --path_t ./save/models/ResNet50_vanilla/ckpt_epoch_240.pth --distill kd --model_s vgg8 -r 0.1 -a 0.9 -b 0 --t_output_as_target_for_input_mix --lw_mix [1,0,1] --learning_rate 0 --fix_student --utils.ON --utils.check_ce_var --epochs 10 --mix_mode autoaugment --project kd__ResNet50vgg8__cifar100__CheckTProbStd_autoaugment

python train_student.py --path_t ./save/models/ResNet50_vanilla/ckpt_epoch_240.pth --distill kd --model_s vgg8 -r 0.1 -a 0.9 -b 0 --t_output_as_target_for_input_mix --lw_mix [1,0,1] --learning_rate 0 --fix_student --utils.ON --utils.check_ce_var --epochs 10 --mix_mode mixup --project kd__ResNet50vgg8__cifar100__CheckTProbStd_mixup

python train_student.py --path_t ./save/models/ResNet50_vanilla/ckpt_epoch_240.pth --distill kd --model_s vgg8 -r 0.1 -a 0.9 -b 0 --t_output_as_target_for_input_mix --lw_mix [1,0,1] --learning_rate 0 --fix_student --utils.ON --utils.check_ce_var --epochs 10 --mix_mode cutmix --project kd__ResNet50vgg8__cifar100__CheckTProbStd_cutmix

python train_student.py --path_t ./save/models/ResNet50_vanilla/ckpt_epoch_240.pth --distill kd --model_s vgg8 -r 0.1 -a 0.9 -b 0 --t_output_as_target_for_input_mix --lw_mix [1,0,1] --finetune_student Experiments/*-141344/weights/ckpt.pth --learning_rate 0 --fix_student --utils.ON --utils.check_ce_var --epochs 10 --mix_mode cutmix_pick --mix_n_run 2 --cutmix_pick_criterion student_entropy --project kd__ResNet50vgg8__cifar100__CheckTProbStd_cutmix_pick_Sentropy

python train_student.py --path_t ./save/models/ResNet50_vanilla/ckpt_epoch_240.pth --distill kd --model_s vgg8 -r 0.1 -a 0.9 -b 0 --t_output_as_target_for_input_mix --lw_mix [1,0,1] --finetune_student Experiments/*-141344/weights/ckpt.pth --learning_rate 0 --fix_student --utils.ON --utils.check_ce_var --epochs 10 --mix_mode cutmix_pick --mix_n_run 2 --cutmix_pick_criterion teacher_entropy --project kd__ResNet50vgg8__cifar100__CheckTProbStd_cutmix_pick_Tentropy