import os
import pickle

import requests
from flask import Flask, request, render_template, url_for, make_response, send_file

from common.Get_Param import get_parameters
app = Flask(__name__)

url_reid = get_parameters('url_reid', 'http://feature-matcher.default.192.168.10.13.sslip.io/')

@app.route('/image/<user_id>/<image_id>')
def get_image(user_id, image_id):
    image = open(f'/home/data/images/{user_id}/{image_id}', 'rb')
    response = make_response(send_file(image, mimetype='image/jpeg'))
    response.headers.set('Content-Disposition', 'attachment', filename=f'{user_id}_{image_id}')
    return response

@app.route('/', methods=['GET', 'POST'])
def upload_images():
    if request.method == 'POST':
        user_id = request.form.get('user_id')
        query_images = request.files.getlist('query_image')
        process_images(user_id, query_images)
        files = [file for file in os.listdir(f'/home/data/images/{user_id}')]
        result_images = [url_for('get_image', image_id=file, user_id=user_id) for file in files]
        return render_template('result.html', images=result_images)
    return '''
    <!doctype html>
    <title>Upload Images</title>
    <h1>Upload Images</h1>
    <form method=post enctype=multipart/form-data>
      User ID: <input type=text name=user_id><br>
      Query Images: <input type=file name=query_image multiple><br>
      <input type=submit value=Upload>
    </form>
    '''


def process_images(user_id, query_images):
    directory = f'/home/data/query/{user_id}/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    for i, img in enumerate(query_images):
        with open(f'/home/data/query/{user_id}/{i}.jpg', 'wb') as f:
            f.write(img.read())
    data = {
        'user_id': user_id
    }
    requests.post(url_reid, json=data)



if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=8080)
