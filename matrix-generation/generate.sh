export SIZE=1000
export DENSITY=1
export CONDITION=2
export REPETITION=1
matlab -nojvm -nodisplay -nosplash -r "size=$SIZE;density=$DENSITY;cond=$CONDITION;rep=$REPETITION;create_matrix;quit"
