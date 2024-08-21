# 0、环境介绍
本文用到的Python库函数为 **pandas** 、 **seaborn** 和 **scipy**，pandas用于读取和处理数据，seaborn用于绘图，scipy用于检验数据是否服从正态分布。
# 1、数据准备
首先使用pandas读取 train.csv，得到名为 df_train 的 Dataframe，代码如下：
```
import pandas as pd
import seaborn as sns
from scipy import stats

df_train = pd.read_csv('train.csv')
```
df_train 的部分数据如下图所示：
![0_原始数据.PNG](https://i-blog.csdnimg.cn/blog_migrate/fd6a5594a14b530803f544938eae01a9.png)

# 2、绘制变量分布图
从 df_train 中提取出 'loanAmnt', 'term', 'interestRate','installment' 等前4列，分别绘制变量分布图，并保存图片，代码如下：
```
for i in ['loanAmnt','term','interestRate','installment']:
    fig = sns.distplot(df_train[i])
    fig_save = fig.get_figure()
    fig_save.savefig('{}.png'.format(i),dpi=300)
    fig_save.clear()
```
其中loanAmnt的数据分布如下图所示：
![loanAmnt.png](https://i-blog.csdnimg.cn/blog_migrate/c4449596340beceec03a5f7ce99daaaf.png)
# 3、用KS检验判断特征是否服从正态分布
从上文中的特征分布图可以较为直观的判断 特征 loanAmnt 应该不服从正态分布，下面使用KS检验进一步进行判断，代码如下：
```
"""
kstest方法：KS检验，参数分别是：待检验的数据，检验方法（这里设置成norm正态分布），均值与标准差
结果返回两个值：statistic → D值，pvalue → P值
p值大于0.05，为正态分布
H0:样本符合  
H1:样本不符合 
如何p>0.05接受H0 ,反之 
"""
u = df_train['loanAmnt'].mean()
std = df_train['loanAmnt'].std()
stats.kstest(df_train['loanAmnt'], 'norm', (u,std))
```
执行代码，得到如下结果：
```
KstestResult(statistic=0.1177846832821679, pvalue=0.0)
```
可以看出 p < 0.05，因此特征不是正态分布。

参考链接1：https://www.cnblogs.com/cgmcoding/p/13253934.html
参考链接2：https://www.cnblogs.com/caiyishuai/p/11184166.html
