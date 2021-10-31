# Swimming Pool Attendance Prediction

This repository contain all code for my personal project - predicting the attendance of [Šutka](https://www.sutka.eu/en/) swimming pool.

Live web with predictions is running on [https://lukoucky.com](https://lukoucky.com/)

![Webpage image](predictor/report/imgs/webpage_new.png)

Project is completed from four main parts:

* [scraper](scraper) - scrapes current attendence, lines reservation, weather, holidays and other usefull data and saves them to database
* [predictor](predictor) - generates predictions of attendace
* [backend](backend) - flask backend providing REST API to data from scraper and predictions
* [frontend](frontend) - webpage with visualizations of attendence and predictions

Project is still work in progress. So far following is done:
* Gathering of attendance data, swimming lines usage data and weather data and storing them in database
* Visualization of data on web page
* Data processing for machine learning algorithms
* Several algorithms trained for prediction. Best performance so far have [Extra Trees Regressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesRegressor.html)

Still on TODO list:

* [ ] Finalize scipts for data exporting
* [ ] Automated generation of all prediction 
* [X] ~~Host website with predictions on github.io and get predictions through REST API (needs server with HTTPS)~~
* [ ] Move from MySQL to SQLite or Postgres
* [ ] Tune algorithms that are working now for better performance
* [ ] Refactor very slow `preprocessing_data.py`
* [ ] Implement Hidden Markov Model 
* [ ] Tune LSTM's 
* [ ] Make nicer web page and make it mobile friendly
* [ ] Move project to container
* [ ] Write short blog about this project
* [ ] Create dashboard with analysis of data and predictions 
