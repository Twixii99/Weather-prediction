# Weather Prediction Problem
## Problem description

This problem is about predicting the weather of the contenent of Austerilia.

The [data](https://www.kaggle.com/jsphyg/weather-dataset-rattle-package) contains about 23 columns, splitting between 6 categorical data and 16 quantative data columns.

## Data Description
### Categorical data :
* Date
* location
* windgustdir
* winddir9am
* winddir3pm
* raintoday
### Numerical data :
*  mintemp
*  maxtemp
*  rainfall
*  evaporation
*  sunshine
*  windgustspeed
*  windspeed9am
*  windspeed3pm
*  humidity9am
*  humidity3pm
*  pressure9am
*  pressure3pm
*  cloud9am
*  cloud3pm
*  temp9am
*  temp3pm

# Problem Thinking

* Removing the `Date` column as it's just Irrelevant to the problem
* Making EDA on the `Categorical data` and `Numerical data` independantly.
* Trying different ML calssification algorithms such as:
    * Logistic regression
    * Decision tree
    * Random forest
    * XGB random forest
* tunning the models parameters
* selecting the model with the best ROC_AUC score

# Building virtual environment 
```bash
# installing PIPENV virtual environment creator
pip install pipenv

# installing the dependancies
pipenv intall numpy pandas sklearn xgboost flask requests gunicorn

# for Running the virtual environment
pipenv shell
```
# How to build a container (Docker)
```bash
# Make sure you have docker first
# After building your own image use the following command to build the image
# the '.' assumes that you run the terminal from the same directory where the Dockerfile exists
sudo docker build . -t weather:1.0
# to run the docker image
sudo docker run --name <Container name> -it -p <continer port>:<forwarding port> -e <entry point if needed> <docker image name :version(latest by default)>  
```

