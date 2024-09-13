（文章来自2022.02.13的博客）  

&emsp;&emsp;之前在看一些竞赛案例的时候遇到了样本不均衡的情况，尝试了不同的采样方式，效果也不是很好，所以在这篇文章讨论一下。
# 1、样本不均衡是不是必须要进行上采样/下采样
## 1.1 数据准备
&emsp;&emsp;这里生成一个包含 2个特征 的 2分类 数据集，同时把数据集中2类样本数据在样本空间的分布差异设置的比较明显，代码如下：

```python
import pandas as pd
import matplotlib.pyplot as plt
from random import uniform

# 标签为0的类别，第一个特征是0～1之间的随机数，第二个特征是2～3之间的随机数
res1 = []
for i in range(50):
    res1.append([uniform(0,1), uniform(2,3), 0])

# 标签为1的类别，第一个特征是2～3之间的随机数，第二个特征是0～1之间的随机数
res2 = []
for j in range(500):
    res2.append([uniform(2,3), uniform(0,1), 1])
    
res = res1 + res2

# 把res转换成Dataframe，并且对列名重新设置
df = pd.DataFrame(res)
df.columns = ['x_1', 'x_2', 'y']

# 画出数据集的散点图
fig = plt.figure(figsize=(10,6))
plt.scatter(df.x_1, df.x_2)
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.show()
```

&emsp;&emsp;生成的数据如下（0、1类别的比例是1:10）：
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/2cccfdd99198044cb994eaaad5e3e89a.png)
## 1.2 训练、测试模型分类效果
&emsp;&emsp;采用支持向量机模型，对数据进行分类，并且打印评价矩阵：
```python
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn import metrics

# 这里的df是上一步生成的df，把特征和标签拆分
x = df.drop(columns=['y'])
y = df['y']

# 生成训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

# 定义并且训练支持向量机模型
clf = svm.SVC()
clf.fit(x_train, y_train)

# 打印评分矩阵
y_pred = clf.predict(x_test)
print(metrics.classification_report(y_test,y_pred))
```

&emsp;&emsp;打印后的评分矩阵如下：

```python
              precision    recall  f1-score   support

           0       1.00      1.00      1.00        19
           1       1.00      1.00      1.00       146

    accuracy                           1.00       165
   macro avg       1.00      1.00      1.00       165
weighted avg       1.00      1.00      1.00       165
```

&emsp;&emsp;可以看到模型的分类效果是很好的。

## 1.3 绘制支持向量机的分类间隔
&emsp;&emsp;还可以把训练好的模型，绘制出分类边界

```python
import numpy as np


res_new = np.array(res)
x_new = res_new[:,0:2]
y_new = res_new[:,2]

clf = svm.SVC()
clf.fit(x_new, y_new)


fig2 = plt.figure(figsize=(10,6))
plt.scatter(x_new[:, 0], x_new[:, 1], c=y_new, s=30, cmap=plt.cm.Paired)
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# create grid to evaluate model
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = clf.decision_function(xy).reshape(XX.shape)

# plot decision boundary and margins
ax.contour(
    XX, YY, Z, colors="k", levels=[-1, 0, 1], alpha=0.5, linestyles=["--", "-", "--"]
)
# plot support vectors
ax.scatter(
    clf.support_vectors_[:, 0],
    clf.support_vectors_[:, 1],
    s=100,
    linewidth=1,
    facecolors="none",
    edgecolors="k",
)
plt.show()
```

&emsp;&emsp;生成的图像如下：
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/a96bc7845e0dab8c771ab671935d981c.png)

## 1.4 结论1
&emsp;&emsp;根据以上测试，所以得出的结论是：假如数据集里面的特征区分度很好，可以把不同类别样本在样本空间的分布区分的很明显，那么即使样本不均衡也不会影响模型的分类效果，即 **特征决定了分类精度的上限** 。


# 2、上采样/下采样是否真的可以提升模型分类效果
## 2.1 数据准备
&emsp;&emsp;这里也是生成包含 2个特征 的 2分类 数据集，代码如下：
```python
# 倒入需要的模块
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_blobs

# we create two clusters of random points
# 创建2类样本的数据，标签为0的数据1000条记录，标签为1的数据100条记录
n_samples_1 = 1000
n_samples_2 = 100
centers = [[0.0, 0.0], [2.0, 2.0]]
clusters_std = [1.5, 0.5]
X, y = make_blobs(
    n_samples=[n_samples_1, n_samples_2],
    centers=centers,
    cluster_std=clusters_std,
    random_state=0,
    shuffle=False,
)
```

## 2.2 训练模型
&emsp;&emsp;接下来训练支持向量机模型，第一个模型是直接使用样本数据；第二个模型通过修改参数 class_weight ，来对标签为1对样本进行上采样，代码如下：

```python
# fit the model and get the separating hyperplane
clf = svm.SVC(kernel="linear", C=1.0)
clf.fit(X, y)

# fit the model and get the separating hyperplane using weighted classes
wclf = svm.SVC(kernel="linear", class_weight={1: 10})
wclf.fit(X, y)
```

## 2.3 绘制分类边界
&emsp;&emsp;然后绘制出2个模型的分类边界，代码如下：

```python
# plot the samples
fig3 = plt.figure(figsize=(10,6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, edgecolors="k")

# plot the decision functions for both classifiers
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# create grid to evaluate model
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T

# get the separating hyperplane
Z = clf.decision_function(xy).reshape(XX.shape)

# plot decision boundary and margins
a = ax.contour(XX, YY, Z, colors="k", levels=[0], alpha=0.5, linestyles=["-"])

# get the separating hyperplane for weighted classes
Z = wclf.decision_function(xy).reshape(XX.shape)

# plot decision boundary and margins for weighted classes
b = ax.contour(XX, YY, Z, colors="r", levels=[0], alpha=0.5, linestyles=["-"])

plt.legend(
    [a.collections[0], b.collections[0]],
    ["non weighted", "weighted"],
    loc="upper right",
)
plt.show()
```

&emsp;&emsp;2个模型的决策边界如下图所示：
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/da971be10d1633274346344070dc63c0.png)


## 2.4 对比模型分类效果
&emsp;&emsp;用上一步训练好的模型，分别预测输入数据的标签（这里输入的数据还是训练集），然后打印评价矩阵，代码如下：

```python
from sklearn import metrics

y_pre_1 = clf.predict(X)		# 没有使用上采样的模型
y_pre_2 = wclf.predict(X)		# 使用上采样的模型

print(metrics.classification_report(y,y_pre_1))
print(metrics.classification_report(y,y_pre_2))
```
&emsp;&emsp;打印的评价矩阵如下：

```python
# 没有使用上采样的模型
              precision    recall  f1-score   support

           0       0.96      0.98      0.97      1000
           1       0.71      0.59      0.64       100

    accuracy                           0.94      1100
   macro avg       0.84      0.78      0.81      1100
weighted avg       0.94      0.94      0.94      1100

# 分割线----------------------------------------------
# 没有使用上采样的模型
              precision    recall  f1-score   support

           0       1.00      0.90      0.95      1000
           1       0.50      0.97      0.66       100

    accuracy                           0.91      1100
   macro avg       0.75      0.94      0.81      1100
weighted avg       0.95      0.91      0.92      1100
```
&emsp;&emsp;可以发现2个模型里面，进行上采样之后的模型标签为1的样本，准确率下降，召回率提升。2个模型的 f1-score 差异不明显，总的来看2个模型的效果差异不明显。

## 2.5 结论2
&emsp;&emsp;总结上边的测试结果，可以得出：通过上采样/下采样不一定会对模型分类效果有所改进，不过也可以在训练模型的时候进行上采样/下采样的尝试。

# 3 总结
&emsp;&emsp;数据特征 是决定模型效果的关键。

&emsp;&emsp;以上讨论如有错误，还请各位读者指出。

&emsp;&emsp;
&emsp;&emsp;
参考链接：
https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC
https://scikit-learn.org/stable/auto_examples/svm/plot_separating_hyperplane_unbalanced.html
https://scikit-learn.org/stable/auto_examples/svm/plot_separating_hyperplane.html#sphx-glr-auto-examples-svm-plot-separating-hyperplane-py
