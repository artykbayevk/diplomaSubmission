from flask import Flask, send_from_directory, request, render_template
import logging, os, sys

app = Flask(__name__)
file_handler = logging.FileHandler('server.log')
app.logger.addHandler(file_handler)
app.logger.setLevel(logging.INFO)

PROJECT_HOME = os.path.dirname(os.path.realpath(__file__))
UPLOAD_FOLDER = '{}/static/uploads/'.format(PROJECT_HOME)
print(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def create_new_folder(local_dir):
    newpath = local_dir
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    return newpath


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def api_root():
    app.logger.info(PROJECT_HOME)
    if request.method == 'POST' and request.files['image']:

        ### IMAGE UPLOADING ###
        app.logger.info(app.config['UPLOAD_FOLDER'])
        img = request.files['image']
        img_name = img.filename
        print(img_name)
        create_new_folder(app.config['UPLOAD_FOLDER'])
        saved_path = os.path.join(app.config['UPLOAD_FOLDER'], 'main.jpg')
        app.logger.info("saving {}".format(saved_path))
        img.save(saved_path)


        ### IMAGE PASS THROUGH CNN AND SEGMENTATOR






        ### RESPONSE RESULT
        return render_template('second.html')
    else:
        return "Where is the image?"


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False)
