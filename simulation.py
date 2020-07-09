import pandas as pd
import numpy as np

def simulate(base, train, feature):
    ser1 = train.sample(n = 5000)[feature]
    min_val = train[feature].min()
    max_val = train[feature].max()
    mean_val = train[feature].mean()
    median_val = train[feature].median()
    diff = (max_val - min_val)/10
    ser2 = pd.Series (np.random.randint(min_val-diff, min_val, size=2500))
    ser3 = pd.Series (np.random.randint(max_val, max_val+diff, size=2500))
    # ser2 = train[(train[feature] < train[feature].quantile(0.5))].sample(n=2500)[feature]
    # ser3 = train[(train[feature] > train[feature].quantile(0.5))].sample(n=2500)[feature]
    # ser2 = ser2.apply(lambda x: x - (ser2.max()-min_val))
    # ser3 = ser3.apply(lambda x: x + (max_val-ser3.min()))
    result = ser1.append([ser2, ser3], ignore_index = True)
    base[feature] = result
    return base

    # a = [0] * 32
    # b = [0] * 32
    # flaglt = 1
    # flaggt = 1
    # aprev = 2
    # bprev = 2
    # for j in range (0, 31):
    #     lt = train.copy()
    #     gt = train.copy()

    #     lt[i] = lt[i].apply(lambda x: x - (j/100)*x)
    #     gt[i] = gt[i].apply(lambda x: x + (j/100)*x)

    #     ylt = lt.pop('class')
    #     Xlt = lt
    #     ygt = gt.pop('class')
    #     Xgt = gt

    #     # creating Dmatrix of the modified data
    #     X_test_lt = xgb.DMatrix(data=Xlt)
    #     X_test_gt = xgb.DMatrix(data=Xgt)

    #     # performing prediction on new test dataset
    #     predslt =  model.predict(X_test_lt)
    #     predsgt =  model.predict(X_test_gt)

    #     a[j] = np.mean(predslt)
    #     b[j] = np.mean(predsgt)
        
    #     if j!=0:
    #         if (abs(a[j]-aprev) < 0.01) and flaglt==1:
    #             a[31] = (j)
    #             flaglt = 0

    #         if (abs(b[j]-bprev) < 0.01) and flaggt==1:
    #             b[31] = (j)
    #             flaggt = 0
        
    #     aprev=a[j]
    #     bprev=b[j]

    # if flaglt==1:
    #     a[31]=30

    # if flaggt==1:
    #     b[31]=30

    # serlt = pd.Series(a)
    # sergt = pd.Series(b)
    # colnamelt = i+ "_dec"
    # colnamegt = i+ "_inc"
    # df[colnamelt] = serlt
    # df[colnamegt] = sergt
    # # df.iloc[0:9].plot()
    # # df.plot(x='percent_change')
    # # plt.show()
    # print (df)