import json
import base64
from morph import match_born
from flask import Flask, jsonify, render_template, request


@app.route("/", methods=["POST"])
def morph():
    js = request.get_json()
    obj = json.loads(js)
    
    image = base64.b64decode(obj['clothes'])
    
    src_point = [[b[x], b[y]] for b in obj['user_bones']]
    dst_point = [[b[x], b[y]] for b in obj['clothes_bones']]

    return jsonify({ "data": obj['clothes']})
