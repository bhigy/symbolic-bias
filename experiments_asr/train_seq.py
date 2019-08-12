for expdir in conv2-batch16 conv2-maxpool-batch16; do
    cd $expdir
    python run.py > log.txt 2>&1
    cd ..
done
