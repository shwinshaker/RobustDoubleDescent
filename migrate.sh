##################################################
# File  Name: migrate.sh
#     Author: shwin
# Creat Time: Thu 26 Nov 2020 10:59:03 AM PST
##################################################

#!/bin/bash

paths=( 
        # 'checkpoints/sgd_wrn-28-2_lr=5e-02_bs=64_wd=0_0005_mom=0_9_sub=ssl_idx_label_split_labeled=4000_pseudolabel_iter=40_th=0.999_type=value_nonratio_func=loglinear_min=0.99_epoch_func=linear_min=60'
        # 'main.py'
        # 'config.py'
        # 'repeat.sh'
        # 'src/preprocess/dataloader.py'
        # 'config-ssl.yaml'
        # 'config-ssl-eval.yaml'
        # 'data_subsets'
	# 'checkpoints1/sgd_PreActResNet18_gain=1_0_ad_pgd_10_alpha=1_wd=0_0005_mom=0_9_pgd_10-0'
	# 'checkpoints2/tiny-imagenet_sgd_PreActResNet18_gain=1_0_ad_pgd_10_alpha=1_wd=0_0005_mom=0_9_pgd_10'
	# 'data/tiny-imagenet-200/words.txt'
	'data_subsets/mean_rb_id_pgd10_epoch_friend_10000_tiny-imagenet.npy'
)

new_paths=(
        # 'checkpoints/sgd_wrn-28-2_lr=5e-02_bs=64_wd=0_0005_mom=0_9_sub=ssl_idx_label_split_labeled=4000_pseudolabel_iter=40_th=0.999_type=value_nonratio_func=loglinear_min=0.99_epoch_func=linear_min=60'
        # 'main.py'
        # 'config.py'
        # 'repeat.sh'
        # 'src/preprocess/dataloader.py'
        # 'config-ssl.yaml'
        # 'data_subsets'
        # logs/log.txt
        # 'config-ssl.yaml'
        # 'config-ssl-eval.yaml'
	# 'config-continue.yaml'
	# 'checkpoints/sgd_PreActResNet18_gain=1_0_ad_pgd_10_alpha=1_wd=0_0005_mom=0_9_pgd_10-0'
	# 'configs/config-preact-kd.yaml'
	# 'checkpoints/tiny-imagenet_sgd_PreActResNet18_gain=1_0_ad_pgd_10_alpha=1_wd=0_0005_mom=0_9_pgd_10'
	# 'data/tiny-imagenet-200/words.txt'
	'data_subsets/mean_rb_id_pgd10_epoch_friend_10000_tiny-imagenet.npy'
)

for ((i=0; i<${#paths[@]}; i++));do
    echo
    echo ">>>>>>>> "${new_paths[$i]}
    if [ -f "./${new_paths[$i]}" ] || [ -d "./${new_paths[$i]}" ]; then
        read -p "path ${new_paths[$i]} already exists. Delete[d], Continue[c], Skip[s] or Terminate[*]? " ans
        case $ans in
           d ) rm -rf ${new_paths[$i]};;
           c ) ;;
           s ) continue;;
           * ) exit;;
        esac
        # exit -1
    fi
    # scp -r chengyu@rackjesh:/home/chengyu/Initialization/${paths[$i]} ./${new_paths[$i]}
    # scp -r chengyu@rackjesh:${paths[$i]} ${new_paths[$i]}
    scp -r chengyu@fifth:/home/chengyu/Initialization/${paths[$i]} ./${new_paths[$i]}
    # scp -r chengyu@descartes:/home/chengyu/Initialization/${paths[$i]} ./${new_paths[$i]}
    # scp -r chengyu@workspace:/home/chengyu/Initialization/${paths[$i]} ./${new_paths[$i]}
    # scp -r jingbo@deepx:/home/jingbo/Chengyu/Initialization/${paths[$i]} ./${new_paths[$i]}
    # rsync -avP --exclude='*.pth.tar' --exclude='ad_running_avg.npy' ftiasch@wu02-1080ti:/home/ftiasch/chengyu/Initialization/${paths[$i]}/ ./${new_paths[$i]}
    # rsync -avP --exclude='*.pth.tar' --exclude='*.pt' chengyu@rackjesh:/home/chengyu/Initialization/${paths[$i]}/ ./${new_paths[$i]}
    # rsync --ignore-existing -avP chengyu@descartes:/home/chengyu/Initialization/${paths[$i]}/ ./${new_paths[$i]}
done
