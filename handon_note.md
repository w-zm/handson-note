# Hands On Machine Learning with Scikit Learn and TensorFlow

- 公开数据库：

  -- [UC Irvine Machine Learning Repository](http://archive.ics.uci.edu/ml/index.php)

  -- [Kaggle datasets](https://www.kaggle.com/datasets)

  -- [Amazon's AWS datasets](http://aws.amazon.com/fr/datasets/)

- Meta protals(they list open data repositories)：

  -- http://dataportals.org/

  -- http://opendastamonitor.eu/

  -- http://quandl.com/

- 其他：

  -- [Wikipedia;s list of Machine Learning datasets](https://en.wikipedia.org/wiki/List_of_datasets_for_machine_learning_research)

  -- [Quora.com question](https://www.quora.com/Where-can-I-find-large-datasets-open-to-the-public)

  -- [Datatsets subreddit](https://www.reddit.com/r/datasets/)




## 数学公式

$$ RMSE(X, h)=\sqrt{\frac{1}{m}\sum_{i=1}^m(h(x^{(i)}-y^{(i)})^2} $$          //Root Mean Saure Error

$$MAE(X, h) = \frac{1}{m}\sum_{i=1}^m|h(x^{(i)}-y^{(i)})|$$                  //Mean Absolute Error



## 第二章

### Data Cleaning

- 数据清理可能用到的函数：dropna(), drop(),  fillna(), 适用于DataFrame。

- scikit-Learn提供了一个方面的类用于缺失值：Imputer。（只作用于数值属性）

- 特征值的缩放（两种常用的方法，一般调整到0-1）：

  - min-max scaling（normalization）
  - standardization

- 保存训练好的模型：

  ```python
  from sklearn.external import joblib

  joblib.dump(my_model, "my_model.pkl")
  # and later...
  my_model_loaded = joblib.load("my_model.pkl")
  ```

  ​

