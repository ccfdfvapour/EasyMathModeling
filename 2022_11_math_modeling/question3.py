import pandas as pd
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import math
from numpy import array
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False


data=pd.read_excel('./材料/附件2：大类资产指数行情数据（3、4问）/大类资产指数行情数据.xlsx')
data['年份']=data['时间'].apply(lambda x:x.year)
data=data.groupby(data['年份']).mean()
data=data.reset_index().copy()
data.to_excel('Q3_clean.xlsx',index=None)

#处理数据后
data=pd.read_excel('Q3_clean___.xlsx')
data.fillna(method='bfill',inplace=True)

temp=data[[ '中证1000'], ['南华商品指数'],['中债-综合财富(7-10年)指数'], ['货币基金']]
data = temp
temp.to_excel('资产指数.xlsx',index=None)

# 计算每年指数回报率
index_returns = temp[temp.columns[1:]].pct_change()
#计算收益率的标准差
nn=index_returns.std()
# 计算夏普比率
index_returns['中证1000_夏普比率']=index_returns['中证1000'].apply(lambda x:(x-0.03)/nn['中证1000'])
index_returns['南华商品指数_夏普比率']=index_returns['南华商品指数'].apply(lambda x:(x-0.03)/nn['南华商品指数'])
index_returns['中债-综合财富(7-10年)指数_夏普比率']=index_returns['中债-综合财富(7-10年)指数'].apply(lambda x:(x-0.03)/nn['中债-综合财富(7-10年)指数'])
index_returns['货币基金_夏普比率']=index_returns['货币基金'].apply(lambda x:(x-0.03)/nn['货币基金'])

index_returns.to_excel('风险收益特征.xlsx',index=None)

df=index_returns

for k in ['中证1000_夏普比率', '南华商品指数_夏普比率', '中债-综合财富(7-10年)指数_夏普比率', '货币基金_夏普比率']:
    plt.bar(df['索引'], df[k], label='夏普比率')
    plt.legend()
    plt.xlabel('年份')
    plt.ylabel('夏普比率')
    plt.title('各种经济状态下的  %s 风险收益特征' % k)
    plt.savefig('./Q3fig/不同经济状态下的风险收益特征(%s).jpg' % k)





