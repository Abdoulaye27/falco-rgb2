# -*- coding: utf-8 -*-
"""
Class definition of YOLO_v3 style detection model on image and video
"""

import colorsys
import os
from timeit import default_timer as timer
import time

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from PIL import Image, ImageFont, ImageDraw

from myopic_filter import bayes_filter
from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from yolo3.utils import letterbox_image
import os
#from keras.utils import multi_gpu_model

from julia.api import Julia
jl = Julia(compiled_modules=False)

# Load your Julia functions
jl.eval('include("falco_function.jl")')

# Initialize belief using Julia's initialize_belief()
belief = jl.eval("reset_belief()")

class YOLO(object):
    _defaults = {
        "model_path": 'yolo_heridal.h5',
        "anchors_path": 'model_data/yolo_anchors.txt',
        "classes_path": 'model_data/_classes.txt',
        "score" : 0.1,
        "iou" : 0.45,
        "model_image_size" : (416, 416),
        "gpu_num" : 1,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults) # set up default values
        self.__dict__.update(kwargs) # and update with user overrides
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.boxes, self.scores, self.classes = self.generate()

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors==6 # default setting
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = tiny_yolo_body(Input(shape=(None,None,3)), num_anchors//2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes)
            self.yolo_model.load_weights(self.model_path) # make sure model, anchors and classes match
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                num_anchors/len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2, ))
        if self.gpu_num>=2:
            self.yolo_model = multi_gpu_model(self.yolo_model, gpus=self.gpu_num)
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                len(self.class_names), self.input_image_shape,
                score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def detect_image(self, image):
        start = timer()
        score = 0

        if self.model_image_size != (None, None):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        print(image_data.shape)
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })

        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

        font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                    size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 300

        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            print(label, (left, top), (right, bottom))
            #print(f'label_size = {label_size}, type = {type(label_size)}')
            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            # My kingdom for a good redistributable image drawing library.
            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[c])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[c])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw

        end = timer()
        print(end - start)
        return image, score

    def close_session(self):
        self.sess.close()

def detect_video(yolo, video_path, output_path=""):
    import cv2
    vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")
    video_FourCC    = int(vid.get(cv2.CAP_PROP_FOURCC))
    video_fps       = vid.get(cv2.CAP_PROP_FPS)
    video_size      = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    isOutput = True if output_path != "" else False
    if isOutput:
        print("!!! TYPE:", type(output_path), type(video_FourCC), type(video_fps), type(video_size))
        out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video_size)
    accum_time = 0
    count_frame = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = timer()
    frames = []
    yolo_alerts = []
    pomdp_alerts = []
    pomdp_gathers = []
    neoCount = 0
    neoCount_list = []
    cs_list = []
    action_list = []
    comp_time = []
    while True:
        return_value, frame = vid.read()
        if return_value:
            count_frame += 1
            if count_frame %5 == 0:
                start_frame = time.time()
                neoCount += 1
                neoCount_list.append(neoCount)
                image = Image.fromarray(frame)
                image, score = yolo.detect_image(image)
                print('Confidence score is: ', score)
                if score is not 0:
                    yolo_alerts.append(1)
                else:
                    yolo_alerts.append(0)
                frames.append(image)
                cs_list.append(score)
                #action = bayes_filter(score)
                action, belief = jl.eval(f"generate_action({score})")
                del belief
                action_list.append(action)
                result = np.asarray(image)
                curr_time = timer()
                exec_time = curr_time - prev_time
                prev_time = curr_time
                accum_time = accum_time + exec_time
                curr_fps = curr_fps + 1
                if accum_time > 1:
                    accum_time = accum_time - 1
                    fps = "FPS: " + str(curr_fps)
                    curr_fps = 0
                cv2.putText(result, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.50, color=(255, 0, 0), thickness=2)
                cv2.namedWindow("result", cv2.WINDOW_NORMAL)
                if action == 1:
                    print('ALERT OPERATOR!')
                    pomdp_alerts.append(1)
                    pomdp_gathers.append(0)
                    #cv2.imshow('Frame', frameResult)
                if action == 2:
                    print('GATHER INFORMATION!')
                    pomdp_alerts.append(0)
                    pomdp_gathers.append(1)
                if action == 3:
                    print('CONTINUE MISSION!')
                    pomdp_alerts.append(0)
                    pomdp_gathers.append(0)
                cv2.imshow("result", result)
                end_frame = time.time() - start_frame
                print(f'{end_frame*1000:.2f} [ms]')
                comp_time.append(end_frame*1000)
                print('--------------------------------------------------------------')
                if isOutput:
                    out.write(result)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    yolo.close_session()

    # plot cs_list versus action_list
    df = pd.DataFrame({'Confidence scores over mission': cs_list, 
                    'Actions taken': action_list, 'Frames': neoCount_list})
    action_dict = {1: 'alert operator', 2: 'gather information', 3: 'continue mission'}
    df['Actions taken'] = df['Actions taken'].map(action_dict)
    df['Actions taken'] = pd.Categorical(df['Actions taken'], categories=action_dict.values())
    df = pd.concat([df, pd.DataFrame({'Confidence scores over mission': [np.nan], 'Actions taken': ['alert operator']})], ignore_index=True)
    plt.figure(1)
    '''
    plt.scatter(cs_list, action_list)
    '''
    sns.catplot(data=df, x='Confidence scores over mission', y='Actions taken', kind='swarm')
    plt.xlabel('Confidence scores over mission')
    plt.ylabel('Actions taken')
    # title of the graph
    plt.title('Decisions taken during L3 maneuver - YOLO RGB')
    # save the figure before calling show
    plt.tight_layout()
    plt.savefig('yolo_rgbL3_csac.png', bbox_inches='tight')
    plt.show()

    # plot frames versus cs_list
    plt.figure(2)
    plt.scatter(neoCount_list, cs_list)
    plt.xlabel('Frames')
    plt.ylabel('Confidence score')
    # title of the graph
    plt.title('Confidence scores evolution during L3 maneuver - YOLO RGB')
    # save the figure before calling show
    plt.tight_layout()
    plt.savefig('yolo_rgbL3_neocs.png')
    plt.show()

    # plot frames versus action_list
    plt.figure(3)
    '''
    plt.scatter(neoCount_list, action_list)
    '''
    sns.catplot(data=df, x='Frames', y='Actions taken', kind='swarm')
    plt.xlabel('Frames')
    plt.ylabel('Actions taken')
    # title of the graph
    plt.title('Decisions evolution during L3 maneuver - YOLO RGB')
    # save the figure before calling show
    plt.tight_layout()
    plt.savefig('yolo_rgbL3_neoac.png', bbox_inches='tight')
    plt.show()

    
    # plot yolo detections 
    detections_complete_yolo = [-1] * len(frames)
    detections_complete_pomdp = [-1] * len(frames)
    detections_complete_pomdp_gathers = [-1] * len(frames)
    for i in range(len(yolo_alerts)):
        detections_complete_yolo[i] = yolo_alerts[i]
    for i in range(len(pomdp_alerts)):
        detections_complete_pomdp[i] = pomdp_alerts[i]
    for i in range(len(pomdp_gathers)):
        detections_complete_pomdp_gathers[i] = pomdp_gathers[i]
    plt.figure(4)
    frame_num = list(range(len(frames)))
    ground_truth = np.zeros(len(frames))
    #ground_truth[:] = 1
    #ground_truth[57:123] = 1
    #LeveL2
    '''
    ground_truth[1:583] = 1
    ground_truth[602:636] = 1
    ground_truth[708:761] = 1
    ground_truth[805:884] = 1
    ground_truth[917:1044] = 1
    ground_truth[1079:1119] = 1
    ground_truth[1145:1180] = 1
    ground_truth[1235:1619] = 1
    '''
    #Level3
    ground_truth[227:515] = 1
    ground_truth[570:747] = 1
    plt.scatter(frame_num, ground_truth+3)
    detection_array_yolo = np.array(detections_complete_yolo)
    plt.scatter(frame_num, detection_array_yolo+2.75)
    detection_array_pomdp = np.array(detections_complete_pomdp)
    plt.scatter(frame_num, detection_array_pomdp+2.5)
    detection_array_pomdp_gathers = np.array(detections_complete_pomdp_gathers)
    plt.scatter(frame_num, detection_array_pomdp_gathers+2.25)
    plt.ylim(3.05,4.05)
    plt.xlabel('Frame Number')
    plt.ylabel('Algorithms & Ground truth alerts')
    plt.title('Alerts during L3 maneuver - YOLO RGB') 
    plt.yticks([4,3.75,3.5,3.25],['person present','yolo','filter alerts','filter gathers info'])
    plt.tight_layout()
    plt.savefig('yolo_rgbL3_final.png')
    plt.show()

    plt.figure(5)
    plt.scatter(neoCount_list, comp_time)
    plt.xlabel('Frames')
    plt.ylabel('Time [ms]')
    # title of the graph
    plt.title('Frame processing time evolution during L3 maneuver - YOLO RGB')
    # save the figure before calling show
    plt.tight_layout()
    plt.savefig('yolo_rgbL3_time.png')
    plt.show()
    
    # Performance metrics calculations
    yolo_tp=0
    yolo_fp=0
    yolo_tn=0
    yolo_fn=0
    yolo_true_positives = []
    yolo_false_positives = []
    yolo_true_negatives = []
    yolo_false_negatives = []
    for i in range(len(ground_truth)):
        if ground_truth[i] == 1 and detection_array_yolo[i] == 1:
            yolo_tp += 1
            #yolo_tp.append(yolo_true_positives)
        if ground_truth[i] == 0 and detection_array_yolo[i] == 1:
            yolo_fp += 1
            #yolo_fp.append(yolo_false_positives)
        if ground_truth[i] == 0 and detection_array_yolo[i] == 0:
            yolo_tn += 1
            #yolo_tn.append(yolo_true_negatives)
        if ground_truth[i] == 1 and detection_array_yolo[i] == 0:
            yolo_fn += 1
            #yolo_fn.append(yolo_false_negatives)

    pomdp_tp=0
    pomdp_fp=0
    pomdp_tn=0
    pomdp_fn=0
    pomdp_true_positives = []
    pomdp_false_positives = []
    pomdp_true_negatives = []
    pomdp_false_negatives = []
    for i in range(len(ground_truth)):
        if ground_truth[i] == 1 and detection_array_pomdp[i] == 1:
            pomdp_tp += 1
            #pomdp_tp.append(pomdp_true_positives)
        if ground_truth[i] == 0 and detection_array_pomdp[i] == 1:
            pomdp_fp += 1
            #pomdp_fp.append(pomdp_false_positives)
        if ground_truth[i] == 0 and detection_array_pomdp[i] == 0:
            pomdp_tn += 1
            #pomdp_tn.append(pomdp_true_negatives)
        if ground_truth[i] == 1 and detection_array_pomdp[i] == 0:
            pomdp_fn += 1
            #pomdp_fn.append(pomdp_false_negatives)

    print('YOLO true positives: ', yolo_tp)
    print('YOLO false positives: ', yolo_fp)
    print('YOLO true negatives: ', yolo_tn)
    print('YOLO false negatives: ', yolo_fn)

    print('POMDP true positives: ', pomdp_tp)
    print('POMDP false positives: ', pomdp_fp)
    print('POMDP true negatives: ', pomdp_tn)
    print('POMDP false negatives: ', pomdp_fn)

    print('---------------------------------------')
    
    # Initialize metrics to 0 or some default value
    pomdp_precision = 0
    pomdp_recall = 0
    pomdp_f1 = 0

    if pomdp_tp + pomdp_fp > 0:
        pomdp_precision = pomdp_tp / (pomdp_tp + pomdp_fp)

    if pomdp_tp + pomdp_fn > 0:
        pomdp_recall = pomdp_tp / (pomdp_tp + pomdp_fn)

    if pomdp_precision + pomdp_recall > 0: # Ensure the denominator in F1 calculation isn't 0
        pomdp_f1 = 2 * (pomdp_precision * pomdp_recall) / (pomdp_precision + pomdp_recall)
    
    yolo_precision = yolo_tp/(yolo_tp+yolo_fp)
    yolo_recall = yolo_tp/(yolo_tp+yolo_fn)
    yolo_f1 = 2/((1/yolo_precision)+(1/yolo_recall))
    
    print('POMDP precision= ', pomdp_precision)
    print('POMDP recall= ', pomdp_recall)
    print('POMDP f1_score= ', pomdp_f1)
    
    print('YOLO precision= ', yolo_precision)
    print('YOLO recall= ', yolo_recall)
    print('YOLO f1_score= ', yolo_f1)

        # Create a dictionary
    data = {
        'Metric': ['True Positives', 'False Positives', 'True Negatives', 'False Negatives', 'Precision', 'Recall', 'F1 Score'],
        'YOLO': [yolo_tp, yolo_fp, yolo_tn, yolo_fn, yolo_precision, yolo_recall, yolo_f1],
        'POMDP': [pomdp_tp, pomdp_fp, pomdp_tn, pomdp_fn, pomdp_precision, pomdp_recall, pomdp_f1]
    }

    # Create DataFrame
    df = pd.DataFrame(data)

    # Print the output
    print(df)

    # Save the DataFrame as a csv
    df.to_csv('perf_rgb_L3_pomdp.csv', index=False)
    
