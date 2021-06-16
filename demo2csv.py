#!python3
import argparse
import csv
import os
from numpy.core.arrayprint import BoolFormat
import torch
import cv2
import numpy as np
from experiment import Structure, Experiment
from concern.config import Configurable, Config
import math
import seaborn as sns
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description='Text Recognition Training')
    parser.add_argument('exp', type=str)
    parser.add_argument('--resume', type=str, help='Resume from checkpoint')
    parser.add_argument('--image_path', type=str,default='./datasets/icdar2015/try_image', help='image path')
    parser.add_argument('--result_dir', type=str, default='./demo_results/', help='path to save results')
    parser.add_argument('--data', type=str,
                        help='The name of dataloader which will be evaluated on.')
    parser.add_argument('--image_short_side', type=int, default=320,
                        help='The threshold to replace it in the representers')
    parser.add_argument('--thresh', type=float,
                        help='The threshold to replace it in the representers')
    parser.add_argument('--box_thresh', type=float, default=0.45,
                        help='The threshold to replace it in the representers')
    parser.add_argument('--visualize', action='store_true',
                        help='visualize maps in tensorboard', default=True)
    parser.add_argument('--resize', action='store_true',
                        help='resize')
    parser.add_argument('--polygon', action='store_true',default=False,
                        help='output polygons if true')
    parser.add_argument('--eager', '--eager_show', action='store_true', dest='eager_show',
                        help='Show iamges eagerly')
    
    args = parser.parse_args()
    args = vars(args)
    args = {k: v for k, v in args.items() if v is not None}

    conf = Config()
    experiment_args = conf.compile(conf.load(args['exp']))['Experiment']
    experiment_args.update(cmd=args)
    experiment = Configurable.construct_class_from_config(experiment_args)
    
    image_paths = get_images(args['image_path'])

    # Start demo here
    Demo(experiment, experiment_args, cmd=args).inference(image_paths, args['visualize'])


# Get all image path to img_files array
def get_images(in_path):
    img_files = []

    files = os.listdir(in_path)
    files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
    #print(files)
    for path in files:
        img_files.append(os.path.join(in_path, path))
        #print(path)
    return img_files

'''
    for (dirpath, dirnames, filenames) in os.walk(in_path):
        for file in filenames:
            filename, ext = os.path.splitext(file)
            ext = str.lower(ext)
            if ext == '.jpg' or ext == '.jpeg' or ext == '.gif' or ext == '.png' or ext == '.pgm':
                img_files.append(os.path.join(dirpath, file))
                print(img_files+"\n")
'''

class Demo:
    def __init__(self, experiment, args, cmd=dict()):
        self.RGB_MEAN = np.array([122.67891434, 116.66876762, 104.00698793])
        self.experiment = experiment
        experiment.load('evaluation', **args)
        self.args = cmd
        self.logger = experiment.logger
        model_saver = experiment.train.model_saver
        self.structure = experiment.structure
        #self.model_path = self.args['resume']
        self.model_path = cmd.get(
            'resume', os.path.join(
                self.logger.save_dir(model_saver.dir_path),
                'final'))

    def init_torch_tensor(self):
        # Use gpu or not
        torch.set_default_tensor_type('torch.FloatTensor')
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            self.device = torch.device('cpu')

    def init_model(self):
        model = self.structure.builder.build(self.device)
        return model

    def resume(self, model, path):
        if not os.path.exists(path):
            print("Checkpoint not found: " + path)
            return
        print("Resuming from " + path)
        states = torch.load(
            path, map_location=self.device)
        model.load_state_dict(states, strict=False)
        print("Resumed from " + path)

    def resize_image(self, img):
        height, width, _ = img.shape
        if height < width:
            new_height = self.args['image_short_side']
            new_width = int(math.ceil(new_height / height * width / 32) * 32)
        else:
            new_width = self.args['image_short_side']
            new_height = int(math.ceil(new_width / width * height / 32) * 32)
        #resized_img = cv2.resize(img, (new_width, new_height))
        resized_img = cv2.resize(img, (480, 480)) # image size
        return resized_img
        
    def load_image(self, image_path):
        img = cv2.imread(image_path, cv2.IMREAD_COLOR).astype('float32')
        original_shape = img.shape[:2]
        img = self.resize_image(img)
        img -= self.RGB_MEAN
        img /= 255.
        img = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0)
        return img, original_shape
        
    def format_output(self, batch, output, id):
        batch_boxes, batch_scores = output
        output_data = []


        for index in range(batch['image'].size(0)):
            original_shape = batch['shape'][index]
            filename = batch['filename'][index]
            result_file_name = 'res_' + filename.split('/')[-1].split('.')[0] + '.txt'
            result_file_path = os.path.join(self.args['result_dir'], result_file_name)
            boxes = batch_boxes[index]
            scores = batch_scores[index]
            if self.args['polygon']:
                print("HELLO")
                '''
                with open(result_file_path, 'wt') as res:
                    for i, box in enumerate(boxes):
                        box = np.array(box).reshape(-1).tolist()
                        result = ",".join([str(int(x)) for x in box])
                        score = scores[i]
                        res.write(result + ',' + str(score) + "\n")
                '''
            else:
                for i in range(boxes.shape[0]):
                    # if got higher confident score
                    score = scores[i]
                    if score < self.args['box_thresh']:
                        continue
                    box = boxes[i,:,:].reshape(-1).tolist()
                    box.insert(0, id)
                    box.append(score)
                    result = ",".join([str(int(x)) for x in box])
                    #res.write(result + ',' + str(score) + "\n")
                    output_data.append(box)
            
        with open('test.csv','a+',newline='') as csv_file:
            csv_write = csv.writer(csv_file)
            #csv_write.writerow(output_data)
            csv_write.writerows(output_data)

        csv_file.close()
        
    def inference(self, image_path, visualize=True):

        self.init_torch_tensor()
        model = self.init_model()
        self.resume(model, self.model_path)
        all_matircs = {}
        model.eval()
        batch = dict()
        id = 1
        height, width = [480, 480] # image size
        print_Heatmap = False

        for img_sample in image_path:
            batch['filename'] = [img_sample]
            img, original_shape = self.load_image(img_sample)
            #height, width = original_shape
            #print(width)
            batch['shape'] = [original_shape]
            with torch.no_grad():
                batch['image'] = img
                #print(img.size())

                # Model Output
                pred = model.forward(batch, training=False)
                #print(pred) #[1, 1, 736, 992]

                #resized_pred = cv2.resize((pred_array*255).astype(int).astype('float32'), (height, width), interpolation=cv2.INTER_AREA)
                
                # Print Heatmap
                if print_Heatmap:

                    pred_binary = pred['binary'][0].permute(1,2,0).detach().cpu().numpy()
                    pred_thresh = pred['thresh'][0].permute(1,2,0).detach().cpu().numpy()
                    pred_thresh_binary = pred['thresh_binary'][0].permute(1,2,0).detach().cpu().numpy()

                    fig, (ax1, ax2, ax3) = plt.subplots(1,3)
                
                    sns.heatmap(data=pred_binary.reshape((height, width)),square=True, ax=ax1) 
                    sns.heatmap(data=pred_thresh.reshape((height, width)),square=True, ax=ax2)
                    sns.heatmap(data=pred_thresh_binary.reshape((height, width)),square=True, ax=ax3) 
                    #plt.close('all')
                    #print(-img[0].permute(1,2,0).detach().cpu().numpy())
                    #plt.imshow(-img[0].permute(1,2,0).detach().cpu().numpy())
                    plt.show()
                    
                
                # binary: text region segmentation map, with shape (N, 1, H, W)
                output = self.structure.representer.represent(batch, pred['thresh_binary'], is_output_polygon=self.args['polygon']) 

                #if not os.path.isdir(self.args['result_dir']):
                    #os.mkdir(self.args['result_dir'])

                self.format_output(batch, output, id)
                id += 1
                if visualize and self.structure.visualizer:
                    #print('image saved')
                    vis_image = self.structure.visualizer.demo_visualize(img_sample, output)
                    cv2.imwrite(os.path.join(self.args['result_dir'], img_sample.split('\\')[-1].split('.')[0]+'.jpg'), vis_image)


if __name__ == '__main__':
    main()
