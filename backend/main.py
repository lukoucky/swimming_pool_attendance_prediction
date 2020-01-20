#!/usr/bin/env python

import json
import argparse
import pandas as pd
from functools import wraps
from flask import Flask, jsonify, send_from_directory, redirect, request, current_app
app = Flask(__name__)

def support_jsonp(f):
	"""Wraps JSONified output for JSONP"""
	@wraps(f)
	def decorated_function(*args, **kwargs):
		callback = request.args.get('callback', False)
		if callback:
			content = str(callback) + '(' + str(f(*args,**kwargs).data) + ')'
			return current_app.response_class(content, mimetype='application/javascript')
		else:
			return f(*args, **kwargs)
	return decorated_function

@app.route('/attendance/<year>/<month>/<day>', methods=['GET'])
@support_jsonp
def get_attendace(year,month,day):
	filepath = '/var/www/html/data/%d-%02d-%02d.csv'%(int(year),int(month),int(day))
	attendance = get_data(filepath,'pool')
	lines = get_data(filepath,'lines_reserved')
	return jsonify({'attendance':attendance, 'lines_reserved':lines})

@app.route('/prediction/extra_trees/<year>/<month>/<day>', methods=['GET'])
@support_jsonp
def get_extra_trees_prediction(year,month,day):
	filepath = '/var/www/html/data/prediction_extra_tree/%d-%02d-%02d.csv'%(int(year),int(month),int(day))
	prediction = get_data(filepath,'pool')
	return jsonify({'prediction':prediction})

@app.route('/prediction/average/<year>/<month>/<day>', methods=['GET'])
@support_jsonp
def get_monthly_average_prediction(year,month,day):
	filepath = '/var/www/html/data/prediction_monthly_average/%d-%02d-%02d.csv'%(int(year),int(month),int(day))
	prediction = get_data(filepath,'pool')
	return jsonify({'prediction':prediction})

def get_data(filepath, column):
	df = pd.read_csv(filepath)
	values = ''
	for i, row in df.iterrows():
		values += str(row[column])+','
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
