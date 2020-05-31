import flask
from flask import request, jsonify

import rulebased

app = flask.Flask(__name__)
app.config["DEBUG"] = True


@app.route('/api/all', methods=['GET'])
def api_all():
    return "<h1>Hier alle shit</h1><p>This site is a prototype API for distant reading of science fiction novels.</p>"

@app.route('/api', methods=['GET'])
def api_id():
    if 'name' in request.args:
        return "<h1>Hallo {}</h1><p>Je moeder.</p>".format(request.args['name'])
    else:
        return "<h1>Geef naam</h1><p>This site is a prototype API for distant reading of science fiction novels.</p>"


@app.route('api/intensify', methods=['POST', 'GET'])
def login():
    error = None
    if request.method == 'POST':
        if request.form['sentence']:
            sentence = request.form['sentence']
            print(sentence)
            return rulebased.main(sentence)
        else:
            error = 'Invalid input'
    # the code below is executed if the request method
    # was GET or the credentials were invalid
    return "GET or invalid credentials"

@app.route('api/analyse', methods=['POST', 'GET'])
def login():
    error = None
    if request.method == 'POST':
        if request.form['sentence']:
            sentence = request.form['sentence']
            print(sentence)
            return rulebased.main(sentence)
        else:
            error = 'Invalid input'
    # the code below is executed if the request method
    # was GET or the credentials were invalid
    return "GET or invalid credentials"

app.run()
