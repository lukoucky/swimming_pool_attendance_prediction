# Šutka Swimming Pool Attendance Prediction

This repository contain all code for my personal project that is predicting attendance of [Šutka](https://www.sutka.eu/en/) swimming pool.

Project is still work in progress. So far following is done:
* Gathering of attendance data, swimming lines usage data and weather data and storing them in database
* Visualization of data on web page
* Data processing for machine learning algorithms
* Several algorithms trained for prediction. Best performance so far have [Extra Trees Regressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesRegressor.html)

Still on TODO list:
* Finalize scipts for data exporting
* Automated generation of prediction 
* To host website with predictions on github.io and get predictions through REST API on my server is used jsonp approach. But since github pages only trust secured sources my server needs to have HTTPS. I generated self signed certificate but security exception needs to be granted in browser for everithing to work. I have to find a better way to do it without these security exceptions. Problem is that to have signed certificate like 
* Move from MySQL to SQLite
* Tune algorithms that are working now for better performance
* Refactor very slow `preprocessing_data.py`
* Implement Hidden Markov Model 
* Tune LSTM's 
* Make nicer web page and make it mobile friendly
* Move project to container
* Writte short blog about this project