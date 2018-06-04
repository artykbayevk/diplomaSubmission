import os
from flask import Flask, request, render_template
from skimage import transform, io
import torch, glob

import operator


from vgg import vgg_my


from PIL import Image
from skimage import io,transform
from torchvision import transforms
from torch.autograd import Variable


pilTrans = transforms.ToPILImage()
import warnings
warnings.filterwarnings("ignore")
transformer = transforms.Compose([
        transforms.RandomCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406),
                         (0.229, 0.224, 0.225))])

LABELS = ['HP', 'ADIDAS', 'ADIDAS', 'ALDI', 'APPLE', 'becks_symbol', 'becks_text', 'BMW', 'carlsberg', 'carlsberg', 'chimay_symbol', 'chimay_text', 'COCACOLA', 'corona_symbol',
              'corona_text', 'DHL', 'erdinger_symbol', 'erdinger_text', 'ESSO', 'ESSO', 'FEDEX', 'FERRARI', 'FORD', 'FOSTERS', 'FOSTERS', 'GOOGLE', 'GUINESS', 'GUINESS', 'HEINEKEN', 'MILKA', 'NVIDIA', 'NVIDIA', 'paulaner_symbol', 'paulaner_text',
              'PEPSI', 'PEPSI', 'RITTERSPORT', 'SHELL', 'singha_symbol', 'singha_text', 'STARBUCKS', 'stellaartois_symbol', 'stellaartois_text', 'TEXACO', 'tsingtao_symbol', 'tsingtao_text', 'UPS','NO LOGO HERE']


def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)

def load_image(image, transform=None):
    if transform is not None:
        image = transform(image).unsqueeze(0)
    return image



app = Flask(__name__, static_folder='static')

PROJECT_HOME = os.path.dirname(os.path.realpath(__file__))
UPLOAD_FOLDER = '{}/static/uploads/'.format(PROJECT_HOME)
SEGMENT_FOLDER = '{}/static/segments/'.format(PROJECT_HOME)
MODELS_FOLDER = '{}/static/models/'.format(PROJECT_HOME)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SEGMENT_FOLDER'] = SEGMENT_FOLDER

def create_new_folder(local_dir):
    newpath = local_dir
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    return newpath

test_cnn = vgg_my.vgg19()
test_cnn.eval()
test_cnn.load_state_dict(torch.load(os.path.join(MODELS_FOLDER,'fl47-vgg19-0001-5000-ep-2200.pt'),map_location=lambda storage, loc: storage))


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def api_root():
    app.logger.info(PROJECT_HOME)
    if request.method == 'POST' and request.files['image']:

        ### IMAGE UPLOADING ###
        img = request.files['image']
        create_new_folder(app.config['UPLOAD_FOLDER'])
        saved_path = os.path.join(app.config['UPLOAD_FOLDER'], 'main.jpg')
        img.save(saved_path)

        input_image = io.imread(saved_path)
        h,w,c = input_image.shape
        diff = 0
        if(h >= w):
            diff = w/225.0
        else:
            diff = h/225.0

        new_shape = (int(h/diff),int(w/diff),c)
        changed_image = transform.resize(input_image, new_shape)
        os.remove(saved_path)
        io.imsave(saved_path, changed_image)
        SEGMENTS_DIR = app.config['SEGMENT_FOLDER']

        for i in range(10):
            new_f_name = saved_path.split('/')[-1].split('.')[0]+'-'+str(i)+'.jpg'
            new_f = os.path.join(SEGMENTS_DIR, new_f_name)
            io.imsave(new_f, changed_image)

        segment_images = glob.glob(os.path.join(SEGMENTS_DIR, '*.jpg'))
        predicted_labels = {}
        for f in segment_images:
            im = Image.open(f)
            image = load_image(im, transformer)
            image_tensor = to_var(image)
            output = test_cnn(image_tensor)
            _, predicted = torch.max(output.data, 1)
            label = LABELS[int(predicted)]
            if label in predicted_labels:
                predicted_labels[label]+=1
            else:
                predicted_labels[label] = 1
        sorted_x = sorted(predicted_labels.items(), key=operator.itemgetter(1))
        predicted_logo = sorted_x[-1][0]
        print(predicted_logo)




        return render_template('second.html', logo = predicted_logo)
    else:
        return "Where is the image?"


if __name__ == '__main__':
    app.run(    debug=False)
