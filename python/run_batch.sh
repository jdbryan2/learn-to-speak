for k in {1..10}
do 
    echo Batch $k
    python batch_learn.py --past 12 --future 12 --period 20 --dim 10 --init random >> data/batch_zeros_100_10.txt
    #python batch_learn.py |& tee -a data/batch/log.txt
done
