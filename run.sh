conda activate portfolio
for efd in 2 8 16;
do
  for eah in 1 2 4;
  do
    for el in 1 2 4;
    do
      for bs in 16 8 4 2;
      do
        python3 main.py --efd $efd --dfd $efd --eah $eah --dah $eah --el $el --dl $el --bs $bs
      done
    done
  done
done