import json
import pandas as pd
from functools import wraps
from flask import Flask, send_from_directory, redirect, request, current_app
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


@app.route('/')
def index():
	return '<h1>Hello World!</h1>'


@app.route('/attendance/<filename>')
def data_regular(filename):
	print(filename)
	return send_from_directory('/var/www/html/data', filename)


@app.route('/prediction/<filename>')
def data_tree(filename):
	print(filename)
	return send_from_directory('/var/www/html/data/prediction_extra_tree', filename)


@default.route('/test/<filename>', methods=['GET'])
@support_jsonp
def test(filename):
	df = pd.read_csv('/var/www/html/data/prediction_extra_tree/'+filename)
	prediction = ''
	for i, row in df.iterrows():
	    prediction += str(row['pool'])+','
	return jsonify({"prediction":prediction[:-1]})


if __name__ == '__main__':
	app.run(debug=True, host='0.0.0.0')



