#!/usr/bin/env python

import os
import json
import argparse
import pandas as pd
from flask import Flask, jsonify
from flask_cors import CORS

app = Flask(__name__)
cors = CORS(app)

@app.route('/attendance/<year>/<month>/<day>', methods=['GET'])
def get_attendace(year,month,day):
	filepath = '/var/www/html/data/%d-%02d-%02d.csv'%(int(year),int(month),int(day))
	attendance = get_data(filepath,'pool')
	lines = get_data(filepath,'lines_reserved')
	return jsonify({'attendance':attendance, 'lines_reserved':lines})

@app.route('/prediction/extra_trees/<year>/<month>/<day>', methods=['GET'])
def get_extra_trees_prediction(year,month,day):
	filepath = '/var/www/html/data/prediction_extra_tree/%d-%02d-%02d.csv'%(int(year),int(month),int(day))
	prediction = get_data(filepath,'pool')
	return jsonify({'prediction':prediction})

@app.route('/prediction/average/<year>/<month>/<day>', methods=['GET'])
def get_monthly_average_prediction(year,month,day):
	filepath = '/var/www/html/data/prediction_monthly_average/%d-%02d-%02d.csv'%(int(year),int(month),int(day))
	prediction = get_data(filepath,'pool')
	return jsonify({'prediction':prediction})

@app.route('/get_all_for/<year>/<month>/<day>', methods=['GET'])
def get_all_for(year,month,day):
	filepath = '/var/www/html/data/prediction_extra_tree/%d-%02d-%02d.csv'%(int(year),int(month),int(day))
	prediction_extra = get_data(filepath,'pool')

	filepath = '/var/www/html/data/prediction_monthly_average/%d-%02d-%02d.csv'%(int(year),int(month),int(day))
	prediction_avg = get_data(filepath,'pool')

	filepath = '/var/www/html/data/%d-%02d-%02d.csv'%(int(year),int(month),int(day))
	attendance = get_data(filepath,'pool')
	lines = get_data(filepath,'lines_reserved')
	return jsonify({'attendance':attendance, 'lines_reserved':lines, 'prediction':{'monthly_average':prediction_avg, 'extra_trees':prediction_extra}})

def get_data(filepath, column):
	values = 'nan,'*288
	if os.path.isfile(filepath):
		df = pd.read_csv(filepath)
		values = ''
		for i, row in df.iterrows():
			if pd.isna(row[column]):
				values += 'nan,'
			else:
				values += str(int(row[column]))+','
	return values[:-1]

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--ssl_cert', type=str, default=None, required=True, metavar='c', help='Path to ssl certificate file')
	parser.add_argument('--ssl_key', type=str, default=None, required=True, metavar='k', help='Path to ssl key file')

	args = parser.parse_args()
	if args.ssl_cert is None or args.ssl_key is None:
		parser.print_help()
	else:
		app.run(debug=True, host='0.0.0.0', ssl_context=(args.ssl_cert, args.ssl_key)) 	
