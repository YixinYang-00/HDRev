import os
import cv2
import numpy as np
import torch
from options.test_options import TestOptions
from data_processing import create_dataset
from models import create_model
from util.util import tensor2im, make_event_preview
import Imath, OpenEXR

class IOException(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)

def writeEXR(img, file):
    try:
        img = np.squeeze(img)
        sz = img.shape
        header = OpenEXR.Header(sz[1], sz[0])
        half_chan = Imath.Channel(Imath.PixelType(Imath.PixelType.HALF))
        header['channels'] = dict([(c, half_chan) for c in "RGB"])
        out = OpenEXR.OutputFile(file, header)
        B = (img[:,:,0]).astype(np.float16).tostring()
        G = (img[:,:,1]).astype(np.float16).tostring()
        R = (img[:,:,2]).astype(np.float16).tostring()
        out.writePixels({'R' : R, 'G' : G, 'B' : B})
        out.close()
    except Exception as e:
        raise IOException("Failed writing EXR: %s"%e)

def save(label, image, savedir, name):
    if(label in ['ldr', 'tm_gt', 'tm_images']):
        image_numpy = tensor2im(image)
        img_path = os.path.join(savedir, '%s_%s.jpg' % (name, label))
        cv2.imwrite(img_path, image_numpy) 
            
    if(label in ['gt', 'images']):
        image_numpy = tensor2im(image, imtype = np.float64)
        img_path = os.path.join(savedir, '%s_%s.exr' % (name, label))
        writeEXR(image_numpy, img_path)
    
    if(label == 'evs') :
        for i in range(0, image.shape[1], 3):
            image_numpy = make_event_preview(image[:, i:i+3, :, :])
            img_path = os.path.join(savedir, '%s_%s.jpg' % (name, label))
            cv2.imwrite(img_path, image_numpy)

def save_image(visuals, savedir, file_path):
    (filepath, filename) = os.path.split(file_path[0])
    (name, extension) = os.path.splitext(filename)
    for label, image in visuals.items():
        for i, im in enumerate(image): 
                save(label, im, savedir, name + '_%.3d' % i)
        
if __name__ == '__main__':
    opt = TestOptions().parse()
    
    opt.phase = 'test'
    opt.num_threads = 0 
    opt.batch_size = 1
    opt.serial_batches = True
    opt.no_flip = True 
    dataset = create_dataset(opt) 
    model = create_model(opt)     
    save_dir = os.path.join(opt.results_dir, opt.name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if opt.eval:
        model.eval()
        
    for i, data in enumerate(dataset):
        if i >= opt.num_test: 
            break
        model.set_input(data) 
        model.test() 
        visuals = model.get_current_visuals()
        img_path = model.get_image_paths()
        if i % 5 == 0:  
            print('processing (%04d)-th image... %s' % (i, img_path))
        save_image(visuals, save_dir, data['ldr_path'])