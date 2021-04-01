batch_sizes="6 24 96 384 1536";
thread_sizes="1 2 4 8 16 32 64"

patch_sizes=($patch_sizes)
batch_sizes=($batch_sizes)

base_csv=$1
ref_csv=$2
threads=28

count=${#batch_sizes[@]}

for i in `seq 1 $count`
do
    echo ${patch_sizes[$i-1]} ${batch_sizes[$i-1]}
    python openslide-test-v1.py -i $base_csv -b ${batch_sizes[$i-1]} -t $threads
done
