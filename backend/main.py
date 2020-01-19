import json
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

@app.route('/attendance/<filename>', methods=['GET'])
@support_jsonp
def get_attendace(filename):
	filepath = '/var/www/html/data/'+filename
	attendance = get_data(filepath,'pool')
	lines = get_data(filepath,'lines_reserved')
	return jsonify({'attendance':attendance, 'lines_reserved':lines})

@app.route('/prediction/extra_trees/<filename>', methods=['GET'])
@support_jsonp
def get_extra_trees_prediction(filename):
	filepath = '/var/www/html/data/prediction_extra_tree/'+filename
	prediction = get_data(filepath,'pool')
	return jsonify({'prediction':prediction})

@app.route('/prediction/average/<filename>', methods=['GET'])
@support_jsonp
def get_monthly_average_prediction(filename):
	filepath = '/var/www/html/data/prediction_monthly_average/'+filename
	prediction = get_data(filepath,'pool')
	return jsonify({'prediction':prediction})

def get_data(filepath, column):
	df = pd.read_csv(filepath)
	values = ''
	for i, row in df.iterrows():
		values += str(row[column])+','
	return values[:-1]

if __name__ == '__main__':
	app.run(debug=True, host='0.0.0.0', ssl_context=('/home/cert/server.crt', '/home/cert/server.key'))
