patch_sizes="64 128 256 512 1024 2048 4096";
batch_sizes="24576 6144 1536 384 96 24 6";

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
