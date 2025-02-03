import os
import sys
import ntpath
import time
import cv2
from . import util

if sys.version_info[0] == 2:
    VisdomExceptionBase = Exception
else:
    VisdomExceptionBase = ConnectionError

def save_images(webpage, visuals, image_path, aspect_ratio=1.0, width=256):
    """Save images to the disk.

    Parameters:
        webpage (the HTML class) -- the HTML webpage class that stores these imaegs (see html.py for more details)
        visuals (OrderedDict)    -- an ordered dictionary that stores (name, images (either tensor or numpy) ) pairs
        image_path (str)         -- the string is used to create image paths
        aspect_ratio (float)     -- the aspect ratio of saved images
        width (int)              -- the images will be resized to width x width

    This function will save images stored in 'visuals' to the HTML file specified by 'webpage'.
    """
    image_dir = webpage.get_image_dir()
    short_path = ntpath.basename(image_path[0])
    name = os.path.splitext(short_path)[0]

    webpage.add_header(name)
    ims, txts, links = [], [], []

    for label, im_data in visuals.items():
        im = util.tensor2im(im_data)
        image_name = '%s_%s.png' % (name, label)
        save_path = os.path.join(image_dir, image_name)
        h, w, _ = im.shape
        if aspect_ratio > 1.0:
            im = cv2.resize(im, (h, int(w * aspect_ratio)), interpolation=cv2.INTER_CUBIC)
        if aspect_ratio < 1.0:
            im = cv2.resize(im, (int(h / aspect_ratio), w), interpolation=cv2.INTER_CUBIC)
        util.save_image(im, save_path)

        ims.append(image_name)
        txts.append(label)
        links.append(image_name)
    webpage.add_images(ims, txts, links, width=width)

class Visualizer():
    """This class includes several functions that can display/save images and print/save logging information.

    It uses a Python library 'visdom' for display, and a Python library 'dominate' (wrapped in 'HTML') for creating HTML files with images.
    """

    def __init__(self, opt):
        """Initialize the Visualizer class

        Parameters:
            opt -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        self.opt = opt  # cache the option
        
        # create a logging file to store training losses
        self.loss_log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
        self.metric_log_name = os.path.join(opt.checkpoints_dir, opt.name, 'metric_log.txt')
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        self.img_dir = os.path.join(self.save_dir, 'images')
        util.mkdirs([self.save_dir, self.img_dir])
        with open(self.loss_log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)
        with open(self.metric_log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Metrics (%s) ================\n' % now)

    # losses: same format as |losses| of plot_current_losses
    def print_current_losses(self, epoch, iters, losses, metrics, t_comp, t_data):
        """print current losses on console; also save the losses to the disk

        Parameters:
            epoch (int) -- current epoch
            iters (int) -- current training iteration during this epoch (reset to 0 at the end of every epoch)
            losses (OrderedDict) -- training losses stored in the format of (name, float) pairs
            t_comp (float) -- computational time per data point (normalized by batch_size)
            t_data (float) -- data loading time per data point (normalized by batch_size)
        """
        message = '(epoch: %d, iters: %d, time: %.3f, data: %.3f) ' % (epoch, iters, t_comp, t_data)
        message_loss = '(epoch: %d, iters: %d, time: %.3f, data: %.3f) ' % (epoch, iters, t_comp, t_data)
        message_metrics = '(epoch: %d, iters: %d, time: %.3f, data: %.3f) ' % (epoch, iters, t_comp, t_data)
        for k, v in losses.items():
            message_loss += '%s: %.3f ' % (k, v)
            message += '%s: %.3f ' % (k, v)
        with open(self.loss_log_name, "a") as log_file:
            log_file.write('%s\n' % message_loss)  # save the message
        for k, v in metrics.items():
            message_metrics += '%s: %.3f ' % (k, v)
            message += '%s: %.3f ' % (k, v)
        with open(self.metric_log_name, "a") as log_file:
            log_file.write('%s\n' % message_metrics)  # save the message
            
        print(message)  # print the message
            
    # Save training samples to disk
    def save_image_to_disk(self, visuals, iteration, epoch):
        for label, image in visuals.items():
            if(label in ['ldr', 'gt', 'images']):
                for i in range(0, image.shape[1], 3):
                    image_numpy = util.tensor2im(image[:, i:i+3, :, :])
                    img_path = os.path.join(self.img_dir, 'epoch%.3d_iter%.4d_%s_%d.jpg' % (epoch, iteration, label, i))
                    cv2.imwrite(img_path, image_numpy) 
            if(label == 'evs') :
                for i in range(0,image.shape[1],3):
                    image_numpy = util.make_event_preview(image[:, i:i+3, :, :])
                    img_path = os.path.join(self.img_dir, 'epoch%.3d_iter%.4d_%s_%d.jpg' % (epoch, iteration, label, i))
                    cv2.imwrite(img_path, image_numpy)
            else:
                continue

