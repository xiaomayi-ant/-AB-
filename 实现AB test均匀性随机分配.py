#背景原理
#现在有一个user，他有一个user_id，这个user_id的值是一个数字；
#每run一个实验，我们都有一个特定的salt，把这个user_id和这个salt拼接起来成一个长字符串；
#把2中得到的长字符串扔进一个哈希函数(可以是MD5或者SHA1)，这里用MD5，然后生成一条哈希数据(Hashed data)；
#取3中得到的哈希数据的头六位字符，转换成一个十六进制整数；
#拿4中得到的整数去除以最大的六位十六进制数字(0xffffff)，注意用浮点数相除，会得到一个介于0和1之间的浮点数。
#根据第5步得到的浮点数是否大于预设阈值，决定这个用户的分组。举个例子，如果我们想得到50-50的平均分配，那么我们会预先设定一个阈值0.5，如果第5步得到的是0.4，那么这个用户被分到控制组，因为它小于0.5，如果第5步得到的是0.6，这个用户被分配到试验组。

#优点
#这样做，实验可以复现(reproducible)，不仅方便精确定位用户，而且实验出问题了也方便debugging。
#一个实验对应一个salt，每个实验不一样，这样保证不同实验有完全不一样的随机分配。因为一家公司一天可能做很多实验，如果一个用户老是被分到试验组，他用户体验会比较那啥（多重treatment冲击！）
#最后得到的随机分配结果还是比较'Random'的，虽然本质上都是假Random。
from tqdm import tqdm_notebook
import hashlib
import pandas  as pd
import scipy.stats
from sklearn.metrics import mutual_info_score
import statsmodels.api as  sm
import numpy as np
from matplotlib import pyplot as plt

#随机分配函数 ab_split
def ab_split(user_id,salt,control_group_size):
    '''
      Input:
      user_id, salt, control_group_size(阈值)

      output:
      't' (for test) or 'c' (for control), based on the ID and salt.
      The control_group_size is a float, between 0 and 1, that sets how big the
      control group is.
      '''
    #把用户id 和 salt 粘贴起来生成一个长字符串 test_id
    test_id=str(user_id)+'-'+str(salt)

    #用hash 函数MD5 来hash  test_id,生成hased data:test_id_hased
    test_id_hased=hashlib.md5(test_id.encode('ascii')).hexdigest()

    #取test_id_hased的头6位数得到 test_id_first_digits
    test_id_first_digits=test_id_hased[:6]

    #把test_id_first_digits转化成整数 test_id_first_int
    test_id_first_int=int(test_id_first_digits,16)

    #如果ab_split 大于 阈值 control_group_size,则分配该user到实验组
    if ab_split >control_group_size:
        return 't'

    #反之,分配user 到控制组
    else:
        return 'c'

#模拟
#生成10000个用户的数据框user,用户的id位0-9999
users=pd.DataFrame({'id':np.arange(100**2)})

#把用户随机分配到控制组'c'和实验组't'
#阈值设置为,0.5,salt 为'ticket-3'
users['test_group']=users.id.apply(lambda id:ab_split(id,'ticket-3',0.5))

#看一下users 数据框的头五行
print(users.head())

dist=scipy.stats.binom(n=10000,p=0.5)
print('treatment group proportion',(users.test_group=='t').mean())

plt.plot(np.arange(4500,5500),dist.pmf(np.arange(4500,5500)))  # arange 在给定间隔内返回均匀间隔的值
plt.title('Probability of observing a particular size of the control group\n'
      'The blue line shows our observed case\n'
      'The red lines show the 95% probability bounds\n')
plt.xlabel('Size of control group')

plt.axvline((users.test_group=='t').sum(),c='black')
plt.axvline(dist.isf(0.95),c='red')
plt.axvline(dist.isf(0.05),c='red')
plt.show()

print(sm.stats.Runs((users.test_group=='t').values.astype('int')).runs_test()[1])
