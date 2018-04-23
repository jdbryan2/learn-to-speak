for k in {1..10}
do 
    echo Batch $k
    python batch_learn.py --past 100 --future 10 --period 10 --init zeros >> data/batch_zeros_100_10.txt
    #python batch_learn.py |& tee -a data/batch/log.txt
done
