# Swimming Pool Attendance Prediction

Predicting the occupation of [Šutka](https://www.sutka.eu/en/) swimming pool.

Live web with predictions is running on [https://lukoucky.com](https://lukoucky.com/)

![Webpage image](predictor/report/imgs/webpage_new.png)

## Main parts:

* [scraper](scraper) - scrapes current occupation, lines reservation, weather, holidays and other useful data and saves them to database
* [predictor](predictor) - generates predictions of attendance
* [backend](backend) - flask backend providing REST API to data from scraper and predictions
* [frontend](frontend) - webpage with visualizations of occupation and predictions

Project is still work in progress. So far following is done:
* Gathering of attendance data, swimming lines usage data and weather data and storing them in database
* Visualization of data on web page
* Data processing for machine learning algorithms
* Several algorithms trained for prediction. Best performance so far have [Extra Trees Regressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesRegressor.html)

## TODO list:

* [ ] Export and import data
    * [X] ~~Export DB once a day~~
    * [ ] Import data to database from export file
* [ ] Predictions
    * [ ] Generate prediction in airflow
    * [ ] Generate prediction to database
    * [ ] Predict from scikit learn model
    * [ ] Predict from NN model
* [ ] Algorithms
    * [ ] RNN
    * [ ] Hidden Markov Model 
    * [ ] LSTM
    * [ ] Transformer
* [ ] Refactoring
    * [ ] Refactor very slow `preprocessing_data.py`
    * [ ] Use SQLite
    * [ ] Take data from DB directly - not using the exported CSVs
* [ ] Web page update
* [ ] Write short blog about this project
* [ ] Create dashboard with analysis of data and predictions 

## History

* 2017 - I started swimming in [Šutka](https://www.sutka.eu/en/) swimming pool regularly but the random spikes in occupation and availability of realtime occupancy data gave me the idea to develop a tool that would tell me when is a good time to visit the pool. I implemented a simple scraper that stored occupancy and line reservation data every 5 minutes and store the data in the MySQL database. The next step was to use the data to create occupancy predictions and eventually the ultimate tool to determine when to go swimming. 
* 2019 - After several attempts to start the actual work on prediction tool I sign up to [Udacity](https://www.udacity.com/) Machine Learning Nanodegree (two 3 months courses from ML basics to production). The last big Capstone Project was to propose and solve problem using ML (gather data, analyze data, propose solution, implement and train ML solution, analyze results) and this was finally the push I need to get into this project.
* January 2020 - Many long nights later I had some working solution. With updated visualizations on web page, several working models and half working pipelines. I had enough to finish the Nanodegree but the work was far from done. The only ML classifier working better than simple Monthly Average was Extra Trees Regressor and Random Forest Regressor. There were some big errors in neural network models and unfinished work on Hidden Markov Model. 
* November 2021 - Last two years bring a lot of variance in to the data of course )show me the dataset not effected by COVID). After one month of completely missing data thanks to some error on the server I finally had some impulse to work on the project again. 

## Miscellaneous

### Dropbox refresh token

Dropbox allows API access to your folders through python SDK. You can visit [developer console](https://www.dropbox.com/developers/apps), create app, assign read and write privilages to specific folder and generate access token for OAuth2 access. The problem is that this token have only 4 hour expiration time. You need to generate refresh token to be able to access dropbox folders for longer time period. Here is how to obtain it form [this tutorial](https://www.codemzy.com/blog/dropbox-long-lived-access-refresh-token):

1) Create new access code

        https://www.dropbox.com/oauth2/authorize?client_id=<APP_KEY>&token_access_type=offline&response_type=code

2) Send post request with access code to get refresh token

        curl --location --request POST 'https://api.dropboxapi.com/oauth2/token' \
        -u '<APP_KEY>:<APP_SECRET>'
        -H 'Content-Type: application/x-www-form-urlencoded' \
        --data-urlencode 'code=<ACCESS_CODE>' \
        --data-urlencode 'grant_type=authorization_code'

3) Use refresh token together with app key and app secret to create insance of python dropbox client

        client = dropbox.Dropbox(app_key = dropbox_app_key,
                                 app_secret = dropbox_app_secret,
                                 oauth2_refresh_token = dropbox_refresh_token)
        client.files_upload(open(dump_file, "rb").read(), dropbox_path)
