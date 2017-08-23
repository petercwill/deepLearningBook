import ml
import os
import numpy as np
import matplotlib.pylab as plt

def main():
    BASE = os.getcwd()
    DATA_DIR = "/samp_data"
    fin = BASE+DATA_DIR+"/winequality-red.csv"



    train_size = [2**i for i in range (4,14)]
    num_trials = 10
    train_errors = np.zeros((num_trials, len(train_size)))
    test_errors = np.zeros((num_trials, len(train_size)))


    for trial in range(num_trials):

        print("trial %s" %trial)
        for i,b in enumerate(train_size):
            ([trainX, trainY], [testX, testY]) = ml.utils.readData(fin)
            trainX = trainX[:b]
            trainY = trainY[:b]

            m = ml.model.LSQ(trainX)
            f = ml.costFunction.MSEL1(5)
            print("size %s" %b)
            opt = ml.optimizer.GD(f, m, trainX, trainY)
            theta_star = opt.run()[0]

            m.theta = theta_star
            train_predictY = m.predict(trainX)
            train_errors[trial, i] = np.mean(np.square(train_predictY - trainY))

            test_pred = m.predict(testX)
            test_errors[trial, i] = np.mean(np.square(test_pred - testY))

    avg_train_errors = np.mean(train_errors, axis=0)
    avg_test_errors = np.mean(test_errors, axis=0)
    std_train_errors = np.std(train_errors, axis=0)
    std_test_errors = np.std(test_errors, axis=0)


    fig, ax = plt.subplots()
    ax.plot(train_size, avg_train_errors, marker ='o', linestyle="--", label="Avg Err. Train")
    ax.fill_between(
        train_size,
        avg_train_errors - std_train_errors,
        avg_train_errors + std_train_errors,
        facecolor='blue',
        alpha=0.5,
        label='+/- 1 Std. Dev.')
    ax.plot(train_size, avg_test_errors, marker ='o', linestyle="--", label="Avg Err. Test")
    ax.fill_between(
        train_size,
        avg_test_errors - std_train_errors,
        avg_test_errors + std_train_errors,
        facecolor='orange',
        alpha=0.5,
        label='+/- 1 Std. Dev.')
    ax.legend(loc='upper right')
    ax.set_xlabel('No. Batches')
    ax.set_ylabel('MSE')
    ax.set_xscale('log',base=2)
    plt.grid()
    plt.legend()
    plt.savefig('test.png')
    plt.show()

if __name__ == '__main__':
    main()
