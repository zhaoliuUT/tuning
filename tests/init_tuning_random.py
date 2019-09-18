import time, sys, os, copy
import numpy as np

# -----------Noncyclic model with inequality constraints-----------
if __name__ == '__main__':
    if len(sys.argv) >= 6:
        avg = float(sys.argv[1])
        fp = float(sys.argv[2])
        fm = float(sys.argv[3])
        numBin = int(sys.argv[4])
        numNeuro = int(sys.argv[5])

        print 'initial average = %.3f'%avg
        print 'FP = %.2f'%fp
        print 'FM = %.2f'%fm
        print 'number of bins = %d'%numBin
        print 'number of neurons = %d'%numNeuro
    else:
        raise Exception('Error: wrong input format!')

# genereate random tuning curve with fixed average = avg

tuning = np.random.uniform(fm, fp, (numNeuro,numBin))
# tuning = np.linspace(fm, fp, numNeuro*numBin).reshape(numNeuro, numBin)

init_average = avg
print("initial average = ", init_average)
ratio = (init_average - fm)/(fp - fm)
count = 0
for i in range(numNeuro):
    while np.any(tuning[i] < fm) or np.any(tuning[i] > fp) or np.fabs(np.average(tuning[i]) - init_average)>1e-4:
        tuning[i] = np.random.uniform(0, 1, numBin)
        tuning[i] /= np.average(tuning[i])
        tuning[i] = tuning[i]*ratio*(fp - fm) + fm
        count +=1
        if count > 1000:
            raise Exception('Finding initial random curve: iteration limit exceeded!')
            break
    print(i, fp,fm, np.average(tuning[i]), np.min(tuning[i]), np.max(tuning[i]))
    print(count)

#FILE_NAME = "tuning-"
#if ADD_TIME:
#    timestr = time.strftime("%m%d-%H%M%S")
#else:
#    timestr = ""
filename = "tuning-" + time.strftime("%m%d-%H%M%S")
print(filename)
np.save(filename, tuning)

