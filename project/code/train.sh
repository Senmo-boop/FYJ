#CUDA_VISIBLE_DEVICES=0,1,2 ./tools/dist_train.sh ./my_config_3x/swinir_fr3x.py 3
#CUDA_VISIBLE_DEVICES=0,1,2 ./tools/dist_train.sh ./my_config_4x/swinir_fr4x.py 3
#CUDA_VISIBLE_DEVICES=0 ./tools/dist_train.sh ./my_config_2x/swinir2x.py 1
#CUDA_VISIBLE_DEVICES=1 ./tools/dist_train.sh ./my_config_3x/swinir3x.py 1
#CUDA_VISIBLE_DEVICES=2 ./tools/dist_train.sh ./my_config_4x/swinir4x.py 1
#
#CUDA_VISIBLE_DEVICES=3 ./tools/dist_train.sh ./my_config_2x/swinir_tr2x.py 1
#CUDA_VISIBLE_DEVICES=0 ./tools/dist_train.sh ./my_config_3x/swinir_tr3x.py 1
#CUDA_VISIBLE_DEVICES=1 ./tools/dist_train.sh ./my_config_4x/swinir_tr4x.py 1
#
#CUDA_VISIBLE_DEVICES=2 ./tools/dist_train.sh ./my_config_2x/swinir_sae2x.py 1
#CUDA_VISIBLE_DEVICES=3 ./tools/dist_train.sh ./my_config_3x/swinir_sae3x.py 1
#CUDA_VISIBLE_DEVICES=0 ./tools/dist_train.sh ./my_config_4x/swinir_sae4x.py 1
#
#CUDA_VISIBLE_DEVICES=1 ./tools/dist_train.sh ./my_config_2x/swinir_fr2x.py 1
#CUDA_VISIBLE_DEVICES=2 ./tools/dist_train.sh ./my_config_3x/swinir_fr3x.py 1
#CUDA_VISIBLE_DEVICES=3 ./tools/dist_train.sh ./my_config_4x/swinir_fr4x.py 1


#CUDA_VISIBLE_DEVICES=0,1 nohup ./tools/dist_train.sh ./my_config_2x/swinir_all2x.py 2 >> exp1.log &
#CUDA_VISIBLE_DEVICES=2,3 nohup ./tools/dist_train.sh ./my_config_3x/swinir_all3x.py 2 >> exp2.log &
#CUDA_VISIBLE_DEVICES=0,1 nohup ./tools/dist_train.sh ./my_config_4x/swinir_all4x.py 2 >> exp3.log &

CUDA_VISIBLE_DEVICES=1 ./tools/dist_train.sh ./my_config_4x/rcan4x.py 1
CUDA_VISIBLE_DEVICES=1 ./tools/dist_train.sh ./my_config_4x/rdn4x.py 1
CUDA_VISIBLE_DEVICES=1 ./tools/dist_train.sh ./my_config_4x/real_esrgan4x.py 1
CUDA_VISIBLE_DEVICES=1 ./tools/dist_train.sh ./my_config_4x/srcnn4x.py 1
CUDA_VISIBLE_DEVICES=1 ./tools/dist_train.sh ./my_config_4x/srgan4x.py 1
CUDA_VISIBLE_DEVICES=1 ./tools/dist_train.sh ./my_config_4x/swinoir4x.py 1

CUDA_VISIBLE_DEVICES=2 ./tools/dist_train.sh ./my_config_3x/rcan3x.py 1
CUDA_VISIBLE_DEVICES=2 ./tools/dist_train.sh ./my_config_3x/rdn3x.py 1
CUDA_VISIBLE_DEVICES=2 ./tools/dist_train.sh ./my_config_3x/real_esrgan3x.py 1
CUDA_VISIBLE_DEVICES=2 ./tools/dist_train.sh ./my_config_3x/srcnn3x.py 1
CUDA_VISIBLE_DEVICES=2 ./tools/dist_train.sh ./my_config_3x/srgan3x.py 1
CUDA_VISIBLE_DEVICES=2 ./tools/dist_train.sh ./my_config_3x/swinoir3x.py 1


CUDA_VISIBLE_DEVICES=3 ./tools/dist_train.sh ./my_config_2x/rcan2x.py 1
CUDA_VISIBLE_DEVICES=3 ./tools/dist_train.sh ./my_config_2x/rdn2x.py 1
CUDA_VISIBLE_DEVICES=3 ./tools/dist_train.sh ./my_config_2x/real_esrgan2x.py 1
CUDA_VISIBLE_DEVICES=3 ./tools/dist_train.sh ./my_config_2x/srcnn2x.py 1
CUDA_VISIBLE_DEVICES=3 ./tools/dist_train.sh ./my_config_2x/srgan2x.py 1
CUDA_VISIBLE_DEVICES=3 ./tools/dist_train.sh ./my_config_2x/swinoir2x.py 1