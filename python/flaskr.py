#########
# routes
#########

import flask, os
import sqlite3
from flask import Flask, render_template,request, session, g, redirect, url_for, abort, flash
from werkzeug import secure_filename
from flask.ext.socketio import SocketIO, emit
from PIL import Image
import datetime, time
import scipy.misc

# From dog breeds
from src.common import consts
from src.data_preparation import dataset
from src.freezing import freeze
from src.common import paths
from src.inference import classify as classification
from src.common import my_methods

# For Inception
import pynder
import os
from urllib.request import urlopen
from skimage import io, transform

import numpy as np
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

sys.path.append("..")

from utils import label_map_util
from utils import visualization_utils as vis_util


app = Flask(__name__)      
app.secret_key = os.urandom(24)
session = {}
session['isFirst']=0
app.config.from_object(__name__) # load config from this file


# For Tinder Automate
user_limit = 1000
image_limit = 5

# From Inception class index
dog_index = 18

#Add parameters for image uploads:
UPLOAD_FOLDER = 'static/upload_folder/'
ALLOWED_EXTENSIONS = set(['png','jpg','jpeg'])
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# # What model to download.
MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# # List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')
NUM_CLASSES = 90

# Load a frozen TF model into Memory
print('')
detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

# Load Label Map
#label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
#categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
#category_index = label_map_util.create_category_index(categories)

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
    (im_height, im_width, 3)).astype(np.uint8)

# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)

# Definite input and output Tensors for detection_graph
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Each box represents a part of the image where a particular object was detected.
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represent how level of confidence for each of the objects.
# Score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

# Finish TensorFlow Initialization
##################################################################################################

# for demo page hack
auto = True

# Load default config and override config from an environment variable
app.config.update(dict(
    DATABASE=os.path.join(app.root_path, 'flaskr.db'),
    SECRET_KEY='development key',
    USERNAME='admin',
    PASSWORD='default'
))

app.config.from_envvar('FLASKR_SETTINGS', silent=True)


@app.route('/')
def home():
  return render_template('home.html')

@app.route('/about')
def about():
  return render_template('about.html')

@app.route('/slides')
def slides():
  return render_template('slides.html')

@app.route('/settings', methods=['GET', 'POST'])
def settings():
  
  if request.method == 'POST':
    
    session['iterations'] = request.form['iterations']
    session['my_dog_url'] = request.form['url']
    session['fb_name']    = request.form['name']
    session['fb_pass']    = request.form['password']
    session['breed']    = request.form['breed']
    session['fb_auth'] = my_methods.get_access_token(session['fb_name'], session['fb_pass'])

    # clean up image dir
    os.system('rm static/my_dog* static/matched*')

    # for demo page hack
    session['isFirst'] = 0
    
    # First classify my dog
    if os.path.exists('static/my_dog.jpg'): os.system('rm static/my_dog.jpg')
    probs = classification.classify('uri', session['my_dog_url'])
    session['my_dog_prob'] = probs.index[0]
    probs.index = probs.breed
    session['st'] = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H:%M:%S')
    _ = probs.take(range(5)).sort_values(['prob']).plot.barh().figure.savefig('static/my_dog_'+session['st']+'.jpg')

    # Dog size
    session['my_dog_breed'], session['my_dog_size'] = my_methods.get_size(session['my_dog_prob'])
    
    return redirect(url_for('autoswipe', request = request))
        
  return render_template('settings.html')




@app.route('/autoswipe', methods=['GET', 'POST'])
def autoswipe():

  # for first page demo
  session['isFirst'] +=1

  print(session['isFirst'])
  
  if session['isFirst'] == 1:
    return render_template('autoswipe.html', all_images = '../static/Norbios_match.jpg',
                           my_image = session['my_dog_url']+'?cs=tinysrgb',
                           my_probs='../static/saved_stat.jpg', matched_probs='../static/saved_matched_stat.jpg')
  
  # https://github.com/charliewolf/pynder/issues/136
  facebook_auth_token = session['fb_auth']
  
  try:
    tinder_session = pynder.Session(facebook_auth_token)
    #long_session = tinder_session.getLongLivedSession()
    total_users = tinder_session.nearby_users()
  except:
    print('No current Tinder Authorization. Please renew your key...')
    return render_template('autoswipe.html', request=request)
    
  print('\nStarting Tinder Automater...')

  # global vars for tracking
  isDog_gbl = False
  num_dog_match = 0
  
  # for HTML display
  dog_images    = []
  no_dog_images = []

  with detection_graph.as_default():
      with tf.Session(graph=detection_graph) as sess:

        # Loop over local Tinder users
        for i,user in enumerate(total_users):

            if num_dog_match >= int(session['iterations']): break
            if session['isFirst'] == 1: break
            print('\n---> Accessing User #',i)
            
	    # Image analysis
            image_count = 0
            isDog = False
            
            # Loop over user photos
            for image in user.get_photos(width='320'):
                if image_count >= image_limit: break
                if isDog: continue
                
                # get the image name 
                photo_url = str(image)
                
                # try to see if URL is still active(people turn off their accounts/switch photos often)
                try: image = io.imread(str(photo_url))
                except: continue
                image_count += 1

                # In numpy tensorflow form
                image_np_expanded = np.expand_dims(image, axis=0)

                # Actual detection.
                try:
                  (boxes, scores, classes, num) = sess.run(
                    [detection_boxes, detection_scores, detection_classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})
                except:
                  print('\nImage could not be processed :(')
              
                # Check top photos for Dog recognition
                for x in range(0,5):
                  if isDog: continue
                  if dog_index == classes[0][x] and scores[0][x] >= 0.51:

                    isDog, isDog_gbl = True, True

                    dog_images.append(photo_url+'?cs=tinysrgb')
                    
                    # Extract just the dog box
                    ymin = int(round(boxes[0,x,0]*np.shape(image)[0]))
                    ymax = int(round(boxes[0,x,2]*np.shape(image)[0]))
                    xmin = int(round(boxes[0,x,1]*np.shape(image)[1]))
                    xmax = int(round(boxes[0,x,3]*np.shape(image)[1]))
                    box_image = image[ymin:ymax,xmin:xmax]
                    raw_image = Image.fromarray(box_image)
                    st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H:%M:%S')
                    filepath = 'static/matched_dog_raw_'+st+'.jpeg'
                    if os.path.exists(filepath): os.system('rm '+filepath)
                    raw_image.save(filepath)
                    
                    # First classify matched dog
                    if os.path.exists('static/matched_dog_'+st+'.jpg'): os.system('rm static/matched_dog_'+st+'.jpg')
                    probs = classification.classify('file', filepath)
                    top_matched_prob = probs.index[0]
                    print('Matched Dog Breed:', top_matched_prob)
                    top_matched_size = 'medium'
                    matched_probs = probs.take(range(1)).sort_values(['prob'])
                    _ = probs.take(range(5)).sort_values(['prob']).plot.barh().figure.savefig('static/matched_dog_'+st+'.jpg')

                    # size of matched breed
                    session['match_dog_breed'], session['matched_dog_size'] = my_methods.get_size(top_matched_prob)
                    print('\nLooking up matched breed:', session['match_dog_breed'], session['matched_dog_size'])
                    
                    # Compare to Norbio(or input dog).  Options are none, same, size
                    if session['breed'] == 'same':
                      isDog, isDog_gbl = False, False
                      if session['my_dog_prob'] == top_matched_prob:
                        isDog, isDog_gbl = True, True
                        num_dog_match += 1
                        print('DOG MATCH!!!', scores[0][x],classes[0][x])
                    elif session['breed'] == 'size':
                      isDog, isDog_gbl = False, False
                      #if (session['my_dog_size'] == 'small' and top_matched_size != 'large') or (session['my_dog_size'] == 'large' and top_matched_size != 'small'):
                      if session['my_dog_size'] == session['matched_dog_size']:
                          isDog, isDog_gbl = True, True    
                          num_dog_match += 1
                          print('DOG MATCH!!!', scores[0][x],classes[0][x])
                    else:
                      num_dog_match += 1
                      print('DOG MATCH!!!', scores[0][x],classes[0][x])
                      
                if not isDog: no_dog_images.append(photo_url)
                all_images = dog_images
                
            if auto:
              if isDog:
                user.like()
                print('\nSwiping Right :)')
              else:
                print('\nSwiping Left :(')
                user.dislike()
                
  if isDog_gbl:
    print ('\n Sending URL:', all_images[0])
    print ('\n My Dog URL :', session['my_dog_url'])
    return render_template('autoswipe.html', all_images = all_images[0],
                           my_image = session['my_dog_url']+'?cs=tinysrgb',
                           my_probs='../static/my_dog_'+session['st']+'.jpg', matched_probs='../static/matched_dog_'+st+'.jpg')

  elif session['isFirst']==1:
    return render_template('autoswipe.html', all_images = '../static/Norbios_match.jpg',
                           my_image = session['my_dog_url']+'?cs=tinysrgb',
                           my_probs='../static/saved_stat.jpg', matched_probs='../static/saved_matched_stat.jpg')
  
  else:
    print('\nNo Dog Match Found...')
    return render_template('autoswipe.html',
                           all_images = 'http://d26qwpdz6hr740.cloudfront.net/doodles/54ff11f88dd49e164472f31c/image.png'+'?auto=compress&cs=tinysrgb',
                           my_image = session['my_dog_url']+'?cs=tinysrgb',
                           my_probs='../static/my_dog_'+session['st']+'.jpg', matched_probs='../static/my_dog_'+session['st']+'.jpg')

  
if __name__ == '__main__':
  app.run(debug=True)
