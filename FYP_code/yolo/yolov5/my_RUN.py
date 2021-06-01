import argparse
import os
import platform
import shutil
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords,
    xyxy2xywh, plot_one_box, strip_optimizer, set_logging)
from utils.torch_utils import select_device, load_classifier, time_synchronized

import matplotlib.pyplot as plt 
import numpy as np

# IMPORTANT NOTE - all important code written by Xian Ren Ng is 'enveloped' with a ### start ### and ### end ### "tag". Others are written by ultralytics which are the developers of yolov5.
# there might be some comments all around which can is for debugging purposes and can be ignored. 

###################################################### START ############################################################################
#lidaudioverse imports 
import libaudioverse as lib
import time
lib.initialize() 


#This section involves audio rendering with libaudioverse
def start_play_audio(no_audios):
    print('AUDIO CALLED')
    song = 'audio/Three Little Pigs.wav' #class 0 - depending on which audio do you want to attach your class too. 
    song1 = 'audio/Treasure.wav'#class 1

    server = lib.Server()

    #song 1 
    buffer_song = lib.Buffer(server) 
    buffer_song.load_from_file(song)

    buffer_node = lib.BufferNode(server)
    buffer_node.buffer = buffer_song

    buffer_node.looping = True 

    if(no_audios == 2):
        print("PLAYING TWO AUDIO FILES")
        server1 = lib.Server() 
        #song 2
        buffer_song1 = lib.Buffer(server1)
        buffer_song1.load_from_file(song1)

        buffer_node1 = lib.BufferNode(server1)
        buffer_node1.buffer = buffer_song1
        buffer_node1.buffer = buffer_song1

        buffer_node1.looping = True
    else: 
        print("PLAYING ONE AUDIO FILE")

    #create an environment which represents the listener  
    env = lib.EnvironmentNode(server, "default")

    env.panning_strategy = lib.PanningStrategies.hrtf
    env.output_channels = 2 #set number of output channels
    env.position = [0,0,0] #x,y,z coordinate of listener/environment
    
    env.connect(0, server) #connect channel zero of environment 'env' to server
    server.set_output_device('default') #same as system

    #create a source
    global source_audio 

    source_audio = lib.SourceNode(server, env)
    buffer_node.connect(0, source_audio, 0) #connect output of buffer node to input of source 

    if(no_audios == 2): 
        print("CONNECTED AUDIO 2 to Environment")
        env.connect(0,server1)
        server1.set_output_device('default')

        global source_audio1
        source_audio1 = lib.SourceNode(server1, env)
        buffer_node1.connect(0, source_audio1, 0) #connect output of buffer node to input of source 

#sources are ALWAYS connected to the environment with which they were created. 
######################################################### END #########################################################################


def detect(save_img=False,check= 1):

    ###################################################### START ###########################################################################
    #initiliase the rendering location as out of range so that there is no audio output
    mid_x_0 = 160
    mid_y_0 = 160
    size_z_0 = 160
    mid_x_1 = 160
    mid_y_1 = 160
    size_z_1 = 160
    ###################################################### START ###########################################################################

    print('DETECT CALLED')

    out, source, weights, view_img, save_txt, imgsz = \
        opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    webcam = source.isnumeric() or source.startswith(('rtsp://', 'rtmp://', 'http://')) or source.endswith('.txt')

    # Initialize
    set_logging()
    device = select_device(opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
        modelc.to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    print("names: ",names)
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]


    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once

    #NOTE: Prediction and Processing Loop will ALWAYS RUN 
    #NOTE: If detected, if statement == TRUE  

    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # Apply Classifier
        if classify:
            print("RUN CLASSIFIER")
            pred = apply_classifier(pred, modelc, img, im0s) 

            
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy() #NOTE: string s = 0, the source number
            #else:
            #    p, s, im0 = path, '', im0s

            save_path = str(Path(out) / Path(p).name)
            txt_path = str(Path(out) / Path(p).stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
            s += '%gx%g ' % img.shape[2:]  # print string - NOTE: image size
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

            #NOTE: If detected, if statement == TRUE

            #classes_liss = list() #refresh list 

            #if not detected or none within camera FOV, do not play sound
            reset_audio(check)

            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string NOTE: example "1 0s 1 1s", 1 class 0 and 1 class 1

                    #classes_liss.append(names[int(c)])

                # Write results
                for *xyxy, conf, cls in reversed(det): #process each class in each loop, 0 -> 1 -> 2 -> 3 -> 4 -> 0......
                    #if save_txt:  # Write to file
                    #    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    #    with open(txt_path + '.txt', 'a') as f:
                    #        f.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format
                    
                    if save_img or view_img:  # Add bbox to image
                        label = '%s %.2f' % (names[int(cls)], conf) #NOTE: class name , confidence/probability
                        #print('label: ',label)
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
                    
                    #print('names[int(cls)]: ',names[int(cls)])
                    ############# EXTRACT Coordinates ##########
                    #xyxy upper left and lower right corner 
                    #print("left_x: ",xyxy[0].cpu().numpy()," right_x: ",xyxy[2].cpu().numpy()," upper_y: ",xyxy[1].cpu().numpy()," lower_y: ",xyxy[3].cpu().numpy())


##################################################### START #############################################################################
                    left_x = xyxy[0].cpu().numpy()
                    upper_y = xyxy[1].cpu().numpy()
                    right_x = xyxy[2].cpu().numpy()
                    lower_y = xyxy[3].cpu().numpy()

                    mid_x = (right_x - left_x)/2 + left_x
                    mid_y = (lower_y - upper_y)/2 + upper_y
                    size_z = (right_x - left_x)*(lower_y - upper_y)

                    #update_audio_position(mid_x,mid_y,size_z,check,names[int(cls)])
                    
                    #save each class coordinates 
                    if(names[int(cls)] == '0'): 
                        mid_x_0 = mid_x
                        mid_y_0 = mid_y
                        size_z_0 = size_z

                    if(names[int(cls)] == '1'): 
                        mid_x_1 = mid_x
                        mid_y_1 = mid_y
                        size_z_1 = size_z
                    
                        # 2 - straight | 3 - left | 4 -right #
                    #update audio position based on face direction
                    if(names[int(cls)] == '2'): #look straight
                        print("straight")
                        update_audio_position(mid_x,mid_y,size_z,check,names[int(cls)])

                    elif(names[int(cls)] == '4'): #look right but facing left in image, so find min image value
                        val = np.min([mid_x_0,mid_x_1])
                        print("right")
                        if(val == mid_x_0): 
                            print("0 on right")
                            update_audio_position(mid_x_0,mid_y_0,size_z_0,check,names[int(cls)])
                        elif(val == mid_x_1): 
                            print("1 on right")
                            update_audio_position(mid_x_1,mid_y_1,size_z_1,check,names[int(cls)])

                    elif(names[int(cls)] == '3'): #look left but facing right in image, so find max image value 
                        val = np.max([mid_x_0,mid_x_1])
                        print("left")
                        if(val == mid_x_0): 
                            print("0 on left")
                            update_audio_position(mid_x_0,mid_y_0,size_z_0,check,names[int(cls)])
                        elif(val == mid_x_1): 
                            print("1 on left")
                            update_audio_position(mid_x_1,mid_y_1,size_z_1,check,names[int(cls)])
################################################### END ###############################################################################

            # Print time (inference + NMS)
            #print('%sDone. (%.3fs)' % (s, t
            # Stream results
            if view_img:
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit

                    lib.shutdown() #NOTE: shutdown to prevent "playing only one file" error later

                    raise stopiteration

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)
                else:
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fourcc = 'mp4v'  # output video codec
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                    vid_writer.write(im0)

    #if save_txt or save_img:
    #    print('Results saved to %s' % Path(out))
    
    #print('Done. (%.3fs)' % (time.time() - t0))

###################################################### START ############################################################################
#def update_audio_position(x,y,z,check,check2): 
def update_audio_position(x,y,z,check,which_class): 
    #keep it within the range of -150 to 150 
    print("image coordinates: x: ",x," y: ",y," z: ",z)

    x_scaled = (x-319.996)/2.6667
    y_scaled = (y-240)/-1.6
    z_scaled = (z-153600)/-1024



    #print('audio coordinates: x: ',x_scaled,' y: ',y_scaled,' z: ',z_scaled)
    #print("\n")
    #print("check2: ",check2)

    #if(check == 1): #one source
    source_audio.position = [x_scaled,y_scaled,z_scaled]

    #if(check == 2): #two sources
    #    if(which_class == '0'):
    #        print("update source ONE and which_class = ",which_class)
    #        source_audio.position = [x_scaled,y_scaled,z_scaled]
    #    elif(which_class == '1'): 
    #        print("update source TWO and which_class = ",which_class)
    #        source_audio1.position = [x_scaled,y_scaled,z_scaled]

def reset_audio(check): 
    source_audio.position = [160,160,160]
    if(check == 2): 
        source_audio1.position = [160,160,160]
################################################### END ###############################################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--numberOf', type=int, default= 1 , help='number of classes or audio') #written by Xian Ren. For checking number of audio files used. 2 is for split audio example
    opt = parser.parse_args()
    #print(opt)

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            #arguement = 1 for one audio, 2 for two audio sources
            number = opt.numberOf
            print("number: ",number)
            start_play_audio(number)
            detect(check = number)
