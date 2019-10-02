# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 15:36:28 2019

@author: 林霖--------------------- 
作者：alphaTao 
来源：CSDN 
原文：https://blog.csdn.net/zkt286468541/article/details/81238708 
版权声明：本文为博主原创文章，转载请附上博文链接！
"""

from imageai.Detection import VideoObjectDetection
import os
import time
#计时
start = time.time()


 #当前文件目录
execution_path = os.getcwd()

'''
___EXAMPLE___
Processing Frame :  16
FOR FRAME  16
Output for each object :  [{'name': 'vase', 'percentage_probability': 21.869677305221558, 'box_points': (352, 310, 523, 535)}, {'name': 'cup', 'percentage_probability': 15.058965981006622, 'box_points': (352, 310, 523, 535)}, {'name': 'person', 'percentage_probability': 25.99698007106781, 'box_points': (3, 17, 210, 544)}]
Output count for unique objects :  {'vase': 1, 'cup': 1, 'person': 1}
------------END OF A FRAME --------------
'''


'''
def forFrame(frame_number, output_array, output_count):
    print("FOR FRAME " , frame_number)
    print("Output for each object : ", output_array)
    print("Output count for unique objects : ", output_count)
    print("------------END OF A FRAME --------------")

def forSeconds(second_number, output_arrays, count_arrays, average_output_count):
    print("SECOND : ", second_number)
    print("Array for the outputs of each frame ", output_arrays)
    print("Array for output count for unique objects in each frame : ", count_arrays)
    print("Output average count for unique objects in the last second: ", average_output_count)
    print("------------END OF A SECOND --------------")

def forMinute(minute_number, output_arrays, count_arrays, average_output_count):
    print("MINUTE : ", minute_number)
    print("Array for the outputs of each frame ", output_arrays)
    print("Array for output count for unique objects in each frame : ", count_arrays)
    print("Output average count for unique objects in the last minute: ", average_output_count)
    print("------------END OF A MINUTE --------------")
'''

detector = VideoObjectDetection()
detector.setModelTypeAsYOLOv3()
#detector.setModelTypeAsRetinaNet()
#detector.setModelTypeAsTinyYOLOv3() #设置需要使用的模型
detector.setModelPath( "D:\Documents\ml_learning\models\yolo242m.h5") #execution_path, "yolo-tiny.h5",加载已经训练好的模型数据
detector.loadModel(detection_speed="flash")


#Video Object Detection and Tracking
#这里它提供了三种不同的模型供我们选择：（ImageAI至少需要更新到2.0.2 后两个模型是新加的） 
#RetinaNet (Size = 145 mb, high performance and accuracy, with longer detection time) 
#YOLOv3 (Size = 237 mb, moderate performance and accuracy, with a moderate detection time) 
#TinyYOLOv3 (Size = 34 mb, optimized for speed and moderate performance, with fast detection time)

#设置输入视频地址 输出地址 每秒帧数等
#video_path = detector.detectObjectsFromVideo(input_file_path=os.path.join(execution_path, "1.mp4"), output_file_path="D:\Documents\ml_learning\1_detected", frames_per_second=30, log_progress=True)
video_path = detector.detectObjectsFromVideo(input_file_path="2.mp4", 
                                             output_file_path="D:\Documents\ml_learning\pred2",
#                                             per_second_function=forSeconds, 
#                                             per_frame_function = forFrame, 
#                                             per_minute_function= forMinute,
                                             frames_per_second=30, frame_detection_interval=10, 
                                             log_progress=True,minimum_percentage_probability=10)

print(video_path)
#结束计时
end = time.time()
print ("\ncost time:",end-start)

'''
execution_path = os.getcwd()

detector = VideoObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath( os.path.join(execution_path , "resnet50_coco_best_v2.0.1.h5"))
detector.loadModel()

custom_objects = detector.CustomObjects(person=True, bicycle=True, motorcycle=True)

video_path = detector.detectCustomObjectsFromVideo(custom_objects=custom_objects, input_file_path=os.path.join(execution_path, "traffic.mp4"), output_file_path=os.path.join(execution_path, "traffic_custom_detected"), frames_per_second=20, log_progress=True)
print(video_path)
'''
