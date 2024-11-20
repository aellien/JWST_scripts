#!/bin/bash


input_list='bands.txt' # input catalog, 
                                            #/!\ check format lines when changing /!\
Npatch=1 # Number of cluster per slurm job

i=0
patch=()
while IFS=' ' read ID chan pixscale pixar n_levels lvl_sep_max mu_lim
do
    # ignore header
    if [[  $ID == "#" ]];then
        continue
    fi
        # if patch smaller than Npatch..
        if [ ${#patch[@]} -lt $Npatch ];then
            echo "Add $ID to patch"
            patch+=($ID) # ..add cluster to patch
        else

            # ..else launch patch..
            echo "Launch ${patch[@]}"
            sbatch start_synthesis_slurm_job.sh ${patch[@]} 

            # ..and reset the patch
            patch=()
        fi
        
done < $input_list

#Launch last patch
if [ ${#patch[@]} -lt $Npatch ];then
    echo "Launch ${patch[@]}"
    sbatch start_synthesis_slurm_job.sh ${patch[@]} 
fi

exit 0
 