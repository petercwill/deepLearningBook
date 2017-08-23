import numpy as np
def readData(fin, split=.8):
    with open(fin) as f:
        header = f.readline()
        lines = f.readlines()

    nrows = len(lines)
    ncols = len(header.split(";"))
    data = np.zeros((nrows,ncols))
    for (i,s) in enumerate(lines):
        data[i] = s.split(";")

    np.random.shuffle(data)
    train, test = np.vsplit(data, [int(nrows*split)])

    return (np.hsplit(train, [ncols-1]), np.hsplit(test, [ncols-1]))


def kfolds(dataX, dataY, nfolds, model):
    data = np.hstack((np.ones((dataX.shape[0],1)), dataX, dataY))
    nrows, ncols = data.shape

    np.random.shuffle(data)
    folds = np.array_split(data, nfolds)

    error_array = np.zeros((2,nfolds))

    for i in range(nfolds):
        validateX, validateY = np.hsplit(folds.pop(0),[ncols - 1])
        trainX, trainY = np.hsplit(np.vstack(folds), [ncols - 1])

        model.fit(trainX, trainY)


        train_error = model.eval(trainX) - trainY
        validate_error = model.eval(validateX) - validateY

        # fig, (ax1, ax2) = plt.subplots(1,2)
        # ax1.hist(train_error)
        # ax2.hist(validate_error)
        # plt.show()

        train_MSE = np.mean(np.square(train_error))
        validate_MSE = np.mean(np.square(validate_error))

        # train_SE = np.std(train_error)/train.shape[0]**.5
        # validate_SE = np.std(validate_error)/validate.shape[0]**.5

        error_array[:,i] = [train_MSE, validate_MSE]

        folds.append(np.hstack((validateX, validateY)))

    print(np.mean(error_array, axis=1))
    print(np.std(error_array, axis=1))
