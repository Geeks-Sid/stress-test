patch_sizes="4096 2048 1024 512 256";
batch_sizes="6 24 96 384 1536";

patch_sizes=($patch_sizes)
batch_sizes=($batch_sizes)

base_csv=$1
ref_csv=$2
threads=28

count=${#batch_sizes[@]}

for i in `seq 1 $count`
do
    echo ${patch_sizes[$i-1]} ${batch_sizes[$i-1]}
    bash openslide-test-v1.py -i $base_csv -r $ref_csv -b ${batch_sizes[$i-1]} -p ${patch_sizes[$i-1]} -t $threads
done
