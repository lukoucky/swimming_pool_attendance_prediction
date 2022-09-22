#!/usr/bin/env python
import os
import argparse
import csv
from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS


app = Flask(__name__)
cors = CORS(app)

data_folder = '/web_data'

@app.route('/attendance/<year>/<month>/<day>', methods=['GET'])
def get_attendace(year,month,day):
	filepath = f'{data_folder}/{year}-{int(month):02d}-{int(day):02d}.csv'
	print(f'Getting data from filepath: {filepath}')
	attendance = get_data(filepath, 'pool')
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

	filepath = f'{data_folder}/{year}-{int(month):02d}-{int(day):02d}.csv'
	attendance = get_data(filepath,'pool')
	lines = get_data(filepath,'lines_reserved')
	return jsonify({'attendance':attendance, 'lines_reserved':lines, 'prediction':{'monthly_average':prediction_avg, 'extra_trees':prediction_extra}})


@app.route('/', methods=['GET'])
def index():
	return send_from_directory('/frontend', 'index.html')
	# return jsonify({'hello': 'world'})


@app.route('/<path>/<file>', methods=['GET'])
def get_file(path, file):
	return send_from_directory(f'/frontend/{path}', file)


def get_data(filepath, column):
	values = 'nan,'*288

	if os.path.isfile(filepath):
		with open(filepath, 'r') as file:
			my_reader = csv.reader(file, delimiter=',')
			is_header = True
			column_id = None
			for row in my_reader:
				if is_header:
					is_header = False
					for i, field in enumerate(row):
						if field == column:
							column_id = i
					if column_id == None:
						# TODO: Return some error to API
						print('Column not found')
						return values[:-1]
					values = ''
				else:
					values += row[column_id]+','
			values = values.replace('null', 'nan')
	return values[:-1]


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--ssl_cert', type=str, default=None, required=False, metavar='c', help='Path to ssl certificate file')
	parser.add_argument('--ssl_key', type=str, default=None, required=False, metavar='k', help='Path to ssl key file')

	args = parser.parse_args()
	if args.ssl_cert is None and args.ssl_key is None:
		print('Running without SSL!')
		app.run(debug=True, host='0.0.0.0', port='9878') 	
	else:
		if args.ssl_cert is None or args.ssl_key is None:
			parser.print_help()
		else:
			app.run(debug=False, host='0.0.0.0', ssl_context=(args.ssl_cert, args.ssl_key)) 	
