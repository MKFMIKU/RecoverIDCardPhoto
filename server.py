from flask import Flask, jsonify, request
import base64
import cv2
import tensorflow as tf
from model import DCGAN
from ops import *
from utils import *
import numpy as np
import matplotlib.pyplot as plt

sess = tf.Session()
run_config = tf.ConfigProto()
run_config.gpu_options.allow_growth=True
sess =  tf.Session(config=run_config)
dcgan = DCGAN(
    sess,
    batch_size = 1,
    checkpoint_dir='checkpoint',
    dataset_name='CCZJZ')
dcgan.load('checkpoint')

def complete(batch_masks,batch_images,demo,img_num):
    zhats = np.random.uniform(-1, 1, size=(dcgan.batch_size, dcgan.z_dim))
    t = 0
    lr = 0.01
    lrd = 0

    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8

    m = np.zeros_like(zhats)
    v = np.zeros_like(zhats)

    for i in xrange(300):
        fd = {
            dcgan.z: zhats,
            dcgan.mask: batch_masks,
            dcgan.inputs: batch_images
        }
        complete_loss,loss, G_imgs = dcgan.sess.run(
            [dcgan.complete_loss,dcgan.grad_complete_loss,dcgan.G],
            feed_dict=fd
        )
        clr = lr/(1+t*lrd)
        t = t+1
        m = np.multiply(m,beta1)+(1-beta1)*loss[0]
        v = np.multiply(v,beta2)+(1-beta2)*loss[0]*loss[0]

        biasCorrection1 = 1 - math.pow(beta1,t)
        biasCorrection2 = 1 - math.pow(beta2,t)
        stepSize = clr*np.sqrt(biasCorrection2)/biasCorrection1
        zhats = zhats-(stepSize*m/(np.sqrt(v)+epsilon))

    inv_masked_hat_images = np.multiply(G_imgs, batch_masks)
    completeed = np.multiply(batch_images, 1.0-batch_masks) + inv_masked_hat_images

    one = (completeed[0]+1.)/2.
    onec = one*255
    onec = onec.astype('uint8')
    onec = cv2.resize(onec,(178,220))

    out = cv2.addWeighted(onec,0.5,demo,0.5,0)

    plt.imsave('./static/%d.jpg'%img_num,out)

def mat2base64(mat):
    """Ecodes image array to Base64"""
    encoded = cv2.imencode(".jpg", mat)[1].tostring()
    b64 = base64.encodebytes(encoded)
    return b64

def base642mat(img_string):
    """Decodes Base64 string to an image array"""
    first_coma = img_string.find(',')
    img_str = img_string[first_coma:].encode('ascii')
    missing_padding = 4 - len(img_str) % 4
    if missing_padding:
        img_str += b'='* missing_padding
    img_bytes = base64.decodestring(img_str)
    image = np.asarray(bytearray(img_bytes), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return image

def pre_img(demo):
    mask = cv2.cvtColor(demo.copy(),cv2.COLOR_RGB2GRAY)
    ret,thresh = cv2.threshold(mask,225,255,cv2.THRESH_BINARY_INV)

    thresh, contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    thresh = cv2.drawContours(thresh, contours, -1, (0,255,0), 3)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(2, 2))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(10, 10))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    white_bg = np.ones_like(demo)*255
    ImageOne = cv2.bitwise_and(demo,demo,mask = thresh)
    white_bg = cv2.bitwise_not(white_bg,white_bg,mask = thresh)
    demo = ImageOne+white_bg
    mask = cv2.cvtColor(demo.copy(),cv2.COLOR_RGB2GRAY)

    r,g,b = cv2.split(demo)

    ret,thresh_g =  cv2.threshold(g,125,255,cv2.THRESH_BINARY_INV)
    ret,thresh_r = cv2.threshold(mask.copy(),100,255,cv2.THRESH_BINARY_INV)

    thresh_g_r = thresh_g-thresh_r

    batch_masks = cv2.cvtColor(thresh_g_r,cv2.COLOR_GRAY2RGB)
    batch_masks = cv2.normalize(batch_masks.astype('float'),None,0.,1.,cv2.NORM_MINMAX)
    batch_images = cv2.normalize(demo.astype('float'),None,-1.,1.,cv2.NORM_MINMAX)

    batch_masks = cv2.resize(batch_masks,(160,240))
    batch_images = cv2.resize(batch_images,(160,240))

    batch_masks = np.resize(batch_masks, [1] + [240,160,3])
    batch_images = np.resize(batch_images, [1] + [240,160,3])

    return batch_masks,batch_images,demo

app = Flask(__name__)

@app.route('/',methods=['POST'])
def index():
    key = request.form['key']
    img_data = request.form['img_data']
    img_num = request.form['img_num']

    auth = "false"
    result = "false"
    photo_data = None
    if key == "a686d856e64726e18495655acf8c4716":
        auth = "success"
        if img_data!=None:
            demo = base642mat(img_data)
            demo = cv2.resize(demo,(178,220))
            demo = cv2.cvtColor(demo,cv2.COLOR_BGR2RGB)
            plt.imsave('./static/demo.jpg',demo)
            batch_masks,batch_images,demo = pre_img(demo)
            complete(batch_masks,batch_images,demo,img_num)
            result = "success"

    return jsonify({'auth': auth,'result':result,'photo_data':'http://dcgan.hk1.mofasuidao.cn/static/complete.jpg'})

if __name__ == '__main__':
    app.run(port=8000)
