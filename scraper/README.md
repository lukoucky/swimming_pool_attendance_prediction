# Swimming Pool Attendance Prediction - Scraper

**TODOs**

* [ ] Update scraper from old Python2 scripts mess to something more organized
* [ ] Dockerize the scraper. Use [CRON jobs](https://stackoverflow.com/questions/37458287/how-to-run-a-cron-job-inside-a-docker-container) like until now or [move to Airflow](https://medium.com/devceldoret/moving-from-cron-to-apache-airflow-ac73007aa28e)
* [ ] Connect scraper docker with other services
* [ ] Update error reporting. So far there is [cronhub](https://cronhub.io/) but this may not be needed with the move towards Airflow. No meter what there needs to be better reporting. One project-wide TODO is to create dashboard. It would be great to have semaphore or something better for each scraper to visualize current state.
