##################################################
# File  Name: launch.sh
#     Author: shwin
# Creat Time: Tue 12 Nov 2019 09:56:32 AM PST
##################################################

#!/bin/bash

# ---------------
trim() {
  local s2 s="$*"
  until s2="${s#[[:space:]]}"; [ "$s2" = "$s" ]; do s="$s2"; done
  until s2="${s%[[:space:]]}"; [ "$s2" = "$s" ]; do s="$s2"; done
  echo "$s"
}

trim_comment() {
    local s="$*"
    echo $(echo $s | sed 's/#.*$//g')
}

checkpoint="$1"
eps='8' # 24
state='last'
[[ ! -z $3 ]] && state="$3"
script="script/robustCertify.py"

mode='fast' # 'square' # 'standard'
virt='False'
out='certify_'"$state"
log='log_certify_'"$state"
if [[ $mode != 'standard' ]]; then
    out=$out'_'$mode
    log=$log'_'$mode
fi
if [[ $eps != "8" ]]; then
    out=$out'_eps='$eps
    log=$log'_eps='$eps
fi
if [[ $virt != "False" ]]; then
    out=$out'_virtual'
    log=$log'_virtual'
fi
out=$out'.out'
log=$log'.txt'

norm='True' 
# norm='False' # For author code

gpu_id='0' # 1 4 5 7
[[ ! -z $2 ]] && gpu_id="$2"

path="$checkpoint"
out_path=$path"/"$out
log_path=$path"/"$log
echo $path
if [ -f $out_path ]; then
    echo '-----------------------------------------------------------------------------------------'
    tail -n 5 $out_path
    read -p "Out path $out_path already exists. Delete[d] or Terminate[*]? " ans
    case $ans in
	d ) rm $out_path;;
	* ) exit;;
    esac
fi
if [ -f $log_path ]; then
    echo '-----------------------------------------------------------------------------------------'
    tail -n 5 $log_path
    read -p "Log path $log_path already exists. Delete[d] or Terminate[*]? " ans
    case $ans in
	d ) rm $log_path;;
	* ) exit;;
    esac
fi
model=$(trim $(trim_comment $(grep '^model:' "$path"/config.yaml | awk -F ':' '{print$2}')))
depth=$(trim $(trim_comment $(grep '^depth:' "$path"/config.yaml | awk -F ':' '{print$2}')))
width=$(trim $(trim_comment $(grep '^width:' "$path"/config.yaml | awk -F ':' '{print$2}')))
echo 'Model: '"$model"-"$depth"-"$width"
dataset=$(trim $(trim_comment $(grep '^dataset:' "$path"/config.yaml | awk -F ':' '{print$2}')))
echo 'Dataset: '"$dataset"

python -u $script --dataset $dataset --eps $eps -p $path -lp $log -m $model --depth $depth --width $width -d $state -g $gpu_id --norm $norm --virt $virt --mode $mode > $out_path 2>&1 &
pid=$!
echo "[$pid] [$gpu_id] [Path]: $path"
echo "s [$pid] [$gpu_id] $(date) [Path]: $path [state]: $state" >> logs/log-certify.txt
