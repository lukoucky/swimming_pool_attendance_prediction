# Backend server
Simple [flask](https://palletsprojects.com/p/flask/) server that provides REST API to attendance data and predictions. Run with `main.py --ssl_cert "path_to_ssl_certificate_file" --ssl_key "path_to_ssl_key_file"`

Server with real data and predictions runs on [https://lukoucky.com:5000](https://lukoucky.com:5000). So far it can only return real attendance with line reservations data or predictions from Monthly Average predictor and Extra Trees Regressor.

Valid requests for predictions and real attendance on January 12, 2020 are:
* Attendance with line usage for single day:
  * Request: `https://lukoucky.com:5000/attendance/2020/01/12`
  * Response: `{"attendance": "0,0,0,0,0,0,10,34,....", "lines_reserved": "0,0,0,0,0,0,1,1,...."}`
* Prediction from Monthly Average predictor for single day:
  * Request: `https://lukoucky.com:5000/prediction/average/2020/01/12`
  * Response: `{"prediction": "0,0,0,0,0,0,10,34,...."}`
* Prediction from Extra Trees Regressor for single day:
  * Request: `https://lukoucky.com:5000/prediction/extra_trees/2020/01/12`
  * Response: `{"prediction": "0,0,0,0,0,0,10,34,...."}`
* All above for single day:
  * Request: `https://lukoucky.com:5000/prediction/get_all_for/2020/01/12`
  * Response: `{"attendance": "0,0,0,0,0,0,10,34,....", "lines_reserved": "0,0,0,0,0,0,1,1,....", "prediction": {"monthly_average": "0,0,0,0,0,0,10,34,....", "extra_trees": "0,0,0,0,0,0,10,34,...."}}`

Response is json with `attendance` or `prediction` field that contains string with 288 numbers separated by comma. Each number represent attendance with 5 minute sampling time starting at 0:00 of given day and ending at 23:55. Real attendance also contains `lines_reserved` field that represents number of reserved lines with the same sampling as `attendance` field.

### TODO's:
* Extend API with requests for data for whole month or year.
 

### Notes 
* Server was developed with help of [this tutorial](https://blog.miguelgrinberg.com/post/the-flask-mega-tutorial-part-i-hello-world)
* Setup of service on linux server was done [like this](https://blog.miguelgrinberg.com/post/running-a-flask-application-as-a-service-with-systemd)
* SSL certificates on server were generated with help of [this tutorial](https://pythonprogramming.net/ssl-https-letsencrypt-flask-tutorial/)
* Certificates renewal done by cron once a day [like here](https://serverfault.com/a/825032)
