batch_sizes="6 24 96 384 1536";
thread_sizes="1 2 4 8 16 32 64"

patch_sizes=($patch_sizes)
batch_sizes=($batch_sizes)

base_csv=$1

for b in batch_sizes
do
    for t in thread_sizes
    do
        echo $b $t
        python gandlf-test-v1.py -i $base_csv -b $b -t $t
    done
done
