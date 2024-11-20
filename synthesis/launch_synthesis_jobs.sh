#!/bin/bash


input_list='bands.txt' # input catalog, 
                                            #/!\ check format lines when changing /!\
Npatch=1 # Number of cluster per slurm job

i=0
patch=()
while IFS=' ' read ID chan pixscale pixar n_levels lvl_sep_max mu_lim
do
    ((i=i+1))
    # ignore header
    if [[  $ID == "#" ]];then
        continue
    fi

    echo "$i | Add $ID to patch"
        patch+=($ID) # ..add cluster to patch

        # if patch greater than Npatch..
        if [ ${#patch[@]} -ge $Npatch ];then

            #�| ..launch patch..
            echo "Launch ${patch[@]}"
            sbatch start_synthesis_slurm_job_new.sh ${patch[@]}

            # ..and reset the patch
            patch=()
        fi
done < $input_list

#Launch last patch
if [ ${#patch[@]} -le $Npatch ];then
    echo "Launch ${patch[@]}"
    sbatch start_synthesis_slurm_job_new.sh ${patch[@]}
fi

exit 