import pandas as pd
import warnings
from sklearn.metrics import r2_score
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import tensorflow
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import Sequential, load_model
np.set_printoptions(suppress=True)
from sklearn.preprocessing import MinMaxScaler
from pylab import *
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
import numpy as np
from sklearn import metrics
from sklearn.metrics import mean_squared_error #均方误差
from sklearn.metrics import mean_absolute_error #平方绝对误差
import matplotlib

data=pd.read_excel('高频指标.xlsx')


#########LSTM多变量模型#############
def split_sequences(sequences, n_steps):
    X, y = list(), list()
    for i in range(len(sequences)):
        end_ix = i + n_steps
        if end_ix > len(sequences) - 1:
            break
        # 最关键的不一样在这一步
        seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix, :]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


def mean_absolute_percentage_error(y_true, y_pred):
    # 平均绝对百分比误差
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def fitlstmmodel(dataset, n_steps=1):
    # dataset：数据标准化后的dataset
    # n_steps：分片大小，默认为1
    # 依次为：['央行货币政策(日)', '超额存款准备金率(超储率):金融机构', '中国大宗商品价格指数:总指数', '人民币存贷款利率','公开市场操作(周)']

    in_seq1 = dataset[:, 0].reshape((dataset.shape[0], 1))
    in_seq2 = dataset[:, 1].reshape((dataset.shape[0], 1))
    in_seq3 = dataset[:, 2].reshape((dataset.shape[0], 1))
    in_seq4 = dataset[:, 3].reshape((dataset.shape[0], 1))
    in_seq5 = dataset[:, 4].reshape((dataset.shape[0], 1))

    dataset = np.hstack((in_seq1, in_seq2, in_seq3, in_seq4, in_seq5))
    X, y = split_sequences(dataset, n_steps)
    n_features = X.shape[2]
    model = Sequential()
    model.add(LSTM(500, activation='relu', return_sequences=True, input_shape=(n_steps, n_features)))
    model.add(LSTM(500, activation='relu'))

    # 和多对一不同点在于，这里多对多的Dense的神经元=features数目
    model.add(Dense(n_features))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=200, verbose=1, shuffle=False)
    model.save('lstm_model.h5')
    last_input = np.array(dataset[-1:, :])
    return X, y, last_input, n_features, n_steps

dataset=data[data.columns[1:]].to_numpy()

# 将整型变为float
dataset = dataset.astype('float32')
#对数据集合进行标准化
scaler = MinMaxScaler(feature_range=(0, 1))

dataset=scaler.fit_transform(dataset)
#输入为标准化后的dataset 	#输出：X为lstm的输入，y为lstm的输出，x_input_last为最后一行dataset的数据，用于预测未来的输入,n_features是特征维度(选择的指标数：11)，n_steps是切片分层
X,y,last_input,n_features,n_steps=fitlstmmodel(dataset,n_steps=1)

#输入1为lstm的输入X，输入2为lstm的输出y，用于训练模型,输入3为标准化模型
#输出：testPredict为预测close的训练数据，testY为close的真实数据
#该函数目标输出训练的RMSE以及预测与训练数据的对比

###预测与评分
def Predict_RMSE_BA(X, y, scaler):
    model = load_model('lstm_model.h5')
    trainPredict = model.predict(X)
    testPredict = scaler.inverse_transform(trainPredict)
    testY = scaler.inverse_transform(y)

    # 依次为：['央行货币政策(日)', '超额存款准备金率(超储率):金融机构', '中国大宗商品价格指数:总指数', '人民币存贷款利率',
    # '公开市场操作(周)', '货币供应量', '工资', '国家财政收支', '人口', '工业生产者购进价格指数（PPIRM）', '就业']
    count = 0
    for i in ['央行货币政策(日)', '超额存款准备金率(超储率)-金融机构', '中国大宗商品价格指数-总指数', '人民币存贷款利率', '公开市场操作(周)']:
        score(testY[:, count], testPredict[:, count])
        plt.plot(testY[:, count], label='True')
        plt.plot(testPredict[:, count], color='orange', label='Predict')
        plt.xlabel('年份')
        plt.ylabel('趋势')
        plt.title('LSTM拟合效果图 2001-2021 %s' % i)
        plt.legend()  # 显示图例
        plt.savefig('./Q2fig/LSTM拟合效果图 2001-2021 %s趋势.jpg' % i)
        plt.show()
        count += 1
    return testPredict, testY


def score(y_true, y_pre):
    # MSE
    print("MAPE :")
    print(mean_absolute_percentage_error(y_true, y_pre))
    # RMSE
    print("RMSE :")
    print(np.sqrt(metrics.mean_squared_error(y_true, y_pre)))
    # MAE
    print("MAE :")
    print(metrics.mean_absolute_error(y_true, y_pre))
    # R2
    print("R2 :")
    print(np.abs(r2_score(y_true, y_pre)))


testPredict, testY = Predict_RMSE_BA(X, y, scaler)


def Predict_future_plot(predict_forword_number, x_input, n_features, n_steps, scaler, testPredict, testY):
    model = load_model('lstm_model.h5')
    predict_list = []
    predict_list.append(x_input)
    while len(predict_list) < predict_forword_number:
        x_input = predict_list[-1].reshape((-1, n_steps, n_features))
        yhat = model.predict(x_input, verbose=0)
        # 预测新值
        predict_list.append(yhat)
    # 取出

    Predict_forword = scaler.inverse_transform(np.array([i.reshape(-1, 1)[:, 0].tolist() for i in predict_list]))
    return Predict_forword[1:, :].tolist()


y_pre = Predict_future_plot(6, last_input, n_features, n_steps, scaler, testPredict, testY)

predictdata=pd.DataFrame(range(2023,2028),columns=['年份'])
predictdata=pd.concat([predictdata,pd.DataFrame(y_pre,columns=data.columns[1:])],axis=1)
data=pd.concat([data,predictdata])
data = data.reset_index(drop=True)

for i in data.columns[1:]:
    plt.plot(data['年份'].values[0:22],data[i].values[0:22],label=i)
    plt.scatter(data['年份'].values[22:27],data[i].values[22:27],label=i+'-预测',c='orange',marker='x')
    plt.legend()
    plt.xlabel( '年份')
    plt.ylabel( '趋势')
    plt.title( '2001-2027 宏观经济环境-%s趋势'%i.replace(':','-'))
    plt.savefig('./predict_fig/2001-2027 宏观经济环境-%s趋势.jpg'%i.replace(':','-'))

data.to_excel('Q2data.xlsx',index=None)




