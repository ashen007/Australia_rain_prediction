# Australia_rain_prediction

### Purpose
the purpose of this repository is to build, trrain and evalute different classification algorithms and compare prebuild models of scikit library with custom written algorithms from scrach. web app can use in [australian rain prediction](https://australian-rain-prediction.herokuapp.com/)

### Introduction
this models are binery class prediction models. most popular and widly used algorithms are used to build  this models and then use ensembeling methods to optimize baseline models to get correct predictions as much as possible. data used to train are feature engineered using imputainons, encoding, transformation. all algorithm are coded from scratch using numpy and then compare them with scikit learn models using same states. feature selection done using pearson correlation coefficent algorithm. full dataframe dividede into to by 80:20 ratio and 80% of data use to train the model and 20% of data used to testing. evaluations done throgh several methods which are amount of successful predictions done by each model, ROC curves and AUC.

### Models

- K Nearest Neighbors
- Logistic Clasification
- CART model
- Linear Regression based classification
- Ensemble models
- Stacking
- Boosting
- Bagging
- Random forest

### Performance
binary label prediction task, more than 100k data points used

<table>
  <tr>
    <th>Model</th>
    <th>Reg. based Classification</th>
    <th>Logistic Classification</th>
    <th>KNN</th>
    <th>CART tree</th>
    <th>Bagging</th>
    <th>Random forest</th>
    <th>ADABoosting</th>
    <th>Gradient Boosting</th>
    <th>Stacking</th>
  </tr>
  <tr>
    <td>score</td>
    <td>73.22</td>
    <td>83.42</td>
    <td>83.48</td>
    <td>83.25</td>
    <td>83.59</td>
    <td>84.07</td>
    <td>83.41</td>
    <td>83.94</td>
    <td>83.5</td>
  </tr>
</table>

- best baseline model: ***logistic classification*** achived 83.48 accurecy
- best ensemble model: ***random forest*** achived 84.07 accurecy

---

- feature enginerring & feature selection

![demo](https://github.com/ashen007/Australia_rain_prediction/blob/master/graphs/imputation.jpg)

![demo](https://github.com/ashen007/Australia_rain_prediction/blob/master/graphs/un-engineered_correlation.jpg)

![demo](https://github.com/ashen007/Australia_rain_prediction/blob/master/graphs/corr.jpg)

- CART Tree

![demo](https://github.com/ashen007/Australia_rain_prediction/blob/master/graphs/tree.jpg)
