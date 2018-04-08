for k in {1..10}
do 
    echo $k
    python batch_learn.py |& tee -a data/batch/log.txt
done
