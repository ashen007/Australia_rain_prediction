# Australia_rain_prediction

### purpose
the purpose of this repository is to build, trrain and evalute different classification algorithms and compare prebuild models of scikit library with custom written algorithms from scrach. web app can use in [australian rain prediction](https://australian-rain-prediction.herokuapp.com/)

### introduction
this models are binery class prediction models. most popular and widly used algorithms are used to build  this models and then use ensembeling methods to optimize baseline models to get correct predictions as much as possible. data used to train are feature engineered using imputainons, encoding, transformation. all algorithm are coded from scratch using numpy and then compare them with scikit learn models using same states. feature selection done using pearson correlation coefficent algorithm. full dataframe dividede into to by 80:20 ratio and 80% of data use to train the model and 20% of data used to testing. evaluations done throgh several methods which are amount of successful predictions done by each model, ROC curves and AUC.
