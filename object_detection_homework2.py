import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from socket import *
import lane_detect
import cv2
import threading
import asyncio
import pymysql
import datetime, time
from socket import error as SocketError

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
from utils import label_map_util
from utils import visualization_utils as vis_util

prior_left_line = None
prior_right_line = None
left_fit_line = None
right_fit_line = None
# cnt = 0
i = 0
frame_count=0

username = None
video_index = None

latitude=0
longitude=0
weather=""

title = 'C:/Django/myblackbox/myapp/static/videos/' + time.strftime('%y%m%d_%H%M%S') + '.mp4'
codec = cv2.VideoWriter_fourcc(*'H264')  # fourcc stands for four character code
out = cv2.VideoWriter(title, codec, 10.0, (1920, 1080))

#서버 부분 Fixme
class SocketInfo():
    HOST=''
    PORT=80
    BUFSIZE=1024
    ADDR=(HOST,PORT)

async def save_video(out, title, mysqlcursor, mysqlconn):
    global video_index

    out.release()
    print('비디오를 release 했음.')
    # sql = "INSERT INTO videos (id, path, username,thumbnailpath) VALUES (%s, %s, %s,%s)"
    # val = (video_index, title, username,"dd")
    # mysqlcursor.execute(sql, val)
    # mysqlconn.commit()
    video_index += 1

def lane_detection(conn, feed):
    global prior_right_line, prior_left_line, left_fit_line, right_fit_line, i
    result,left,right,left_fit,right_fit,j = lane_detect.detect(feed, prior_right_line, prior_left_line, left_fit_line, right_fit_line, i)

    if result is False:
        pass

    elif result == "change":
        prior_left_line = left
        prior_right_line = right
        left_fit_line = left_fit
        right_fit_line = right_fit
        i = j
        msg = "lane\n"
        msg += "change\n"
        msg += str(left_fit[0]) + "\n"
        msg += str(left_fit[1]) + "\n"
        msg += str(left_fit[2]) + "\n"
        msg += str(left_fit[3]) + "\n"
        msg += str(right_fit[0]) + "\n"
        msg += str(right_fit[1]) + "\n"
        msg += str(right_fit[2]) + "\n"
        msg += str(right_fit[3]) + "\n"
        conn.send(msg.encode())
    elif result == "pass":
        prior_left_line = left
        prior_right_line = right
        left_fit_line = left_fit
        right_fit_line = right_fit
        i = j
        msg = "lane\n"
        msg += "pass\n"
        msg += str(left_fit[0]) + "\n"
        msg += str(left_fit[1]) + "\n"
        msg += str(left_fit[2]) + "\n"
        msg += str(left_fit[3]) + "\n"
        msg += str(right_fit[0]) + "\n"
        msg += str(right_fit[1]) + "\n"
        msg += str(right_fit[2]) + "\n"
        msg += str(right_fit[3]) + "\n"
        conn.send(msg.encode())

    # return (prior_right_line, prior_left_line, left_fit_line, right_fit_line, i)



def main():
    # global prior_left_line, prior_right_line, left_fit_line, right_fit_line, i
    global username, video_index , title, codec, out, frame_count, latitude, longitude, weather
    # What model to download.
    MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
    MODEL_FILE = MODEL_NAME + '.tar.gz'
    DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

    # Path to frozen detection graph. This is the actual model that is used for the object detection.
    PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

    # List of the strings that is used to add correct label for each box.
    PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

    NUM_CLASSES = 90

    opener = urllib.request.URLopener()
    opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
    tar_file = tarfile.open(MODEL_FILE)
    for file in tar_file.getmembers():
      file_name = os.path.basename(file.name)
      if 'frozen_inference_graph.pb' in file_name:
        tar_file.extract(file, os.getcwd())


    detection_graph = tf.Graph()
    with detection_graph.as_default():
      od_graph_def = tf.GraphDef()
      with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')


    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    lane_flag = False
    ssock = socket(AF_INET, SOCK_STREAM)
    ssock.bind(SocketInfo.ADDR)
    print('try')
    ssock.listen(5)
    conn, addr = ssock.accept()
    print('connected by', addr)
    print('qq')

    # cap=cv2.VideoCapture(0) # 0 stands for very first webcam attach
    cv2.namedWindow("image", cv2.WINDOW_AUTOSIZE)

    save_count=0
    save_title_num= 0

    # title = 'C:/tf_object_detection_api/object_detection/videos/' + time.strftime('%y%m%d_%H%M%S') + '.avi'
    # codec = cv2.VideoWriter_fourcc(*'DIVX')  # fourcc stands for four character code
    # out = cv2.VideoWriter(title, codec, 20.0, (1920, 1080))

    imgData = b''

    while(True):
        data = conn.recv(1024)
        if data.find(b'user=') != -1:
            username = data[data.find(b'user=') + 5:data.find(b'=userend')].decode("utf-8")
            print("username : ", username + "\n")
            break

    # DB 커넥션
    mysqlconn = pymysql.connect(host='127.0.0.1', user='admin', password='rla0204rb!', db='blackbox')
    mysqlcursor = mysqlconn.cursor()

    mysqlcursor.execute("SELECT COUNT(*) from videos")
    result = mysqlcursor.fetchone()
    video_index = result[0]+1
    print("videos테이블에 몇개 튜플 있는지 확인했어 : ", video_index)

    sql = "INSERT INTO videos (id, path, username,thumbnailpath) VALUES (%s, %s, %s,%s)"
    val = (video_index, title, username, 'C:/Django/myblackbox/myapp/static/thumbnails/'+str(video_index)+'.jpg')
    mysqlcursor.execute(sql, val)
    mysqlconn.commit()
    loop = asyncio.get_event_loop()

    with detection_graph.as_default():
      with tf.Session(graph=detection_graph) as sess:
        ret=True
        while (ret):
            data = conn.recv(1024)
            if data.find(b'\x00\x04true') != -1:
                print('true')
                lane_flag = True
                data.strip(b'\x00\x04true')
            elif data.find(b'\x00\x05false') != -1:
                print('false')
                lane_flag = False
                data.strip(b'\x00\x05false')

            if data.find(b'address=') != -1:
               try:
                   latitude, longitude = data[data.find(b'address=') + 8:data.find(b'=addressend')].decode("utf-8").split(',')
                   weather = data[data.find(b'weather=') + 8:data.find(b'=weatherend')].decode("utf-8")
               except:
                   data += conn.recv(100)
                   latitude, longitude = data[data.find(b'address=') + 8:data.find(b'=addressend')].decode("utf-8").split(',')
                   weather = data[data.find(b'weather=') + 8:data.find(b'=weatherend')].decode("utf-8")

            imgData += data
            a = imgData.find(b'\xff\xd8')
            b = imgData.find(b'\xff\xd9')

            if a != -1 and b != -1:
                frame_count+=1
                feed = cv2.imdecode(np.frombuffer(imgData[a:b + 2], dtype=np.uint8), 1)
                if frame_count == 3:
                    frame_count=0
                    if lane_flag == True:
                        threading.Thread(target=lane_detection, args=(conn, feed, )).start()

                    # Definite input and output Tensors for detection_graph
                    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                    # Each box represents a part of the image where a particular object was detected.
                    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                    # Each score represent how level of confidence for each of the objects.
                    # Score is shown on the result image, together with the class label.
                    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
                    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
                    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

                      # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                    image_np_expanded = np.expand_dims(feed, axis=0)
                      # Actual detection.
                    (boxes, scores, classes, num) = sess.run(
                          [detection_boxes, detection_scores, detection_classes, num_detections],
                          feed_dict={image_tensor: image_np_expanded})
                      # Visualization of the results of a detection.


                    print("함수 넘기기 전이야. video_index랑 save_count는 ", video_index, save_count)
                    _, box_points = vis_util.visualize_boxes_and_labels_on_image_array(
                          feed,
                          np.squeeze(boxes),
                          np.squeeze(classes).astype(np.int32),
                          np.squeeze(scores),
                          category_index,
                          video_index,
                          save_count,
                          title,
                          mysqlcursor,
                          mysqlconn,
                          conn,
                          latitude,
                          longitude,
                          weather,
                          use_normalized_coordinates=True,
                          line_thickness=8,
                          )

                    # h, w = feed.shape[:2]
                    # print("---------------box")
                    # print(box_points)
                    # if len(box_points) == 0:
                    #     box_msg = "noobject\n"
                    #     conn.send(box_msg.encode())
                    # elif len(box_points) > 0:
                    #     box_msg = str(len(box_points)) + "\n"
                    #     for i in box_points:
                    #         box_msg += str(i[0] * h) + "\n"
                    #         box_msg += str(i[1] * w) + "\n"
                    #         box_msg += str(i[2] * h) + "\n"
                    #         box_msg += str(i[3] * w) + "\n"
                    #     conn.send(box_msg.encode())

                save_count += 1
                # VideoFileOutput.write(feed)
                if save_count == 1:
                    print("thumbnail start")
                    cv2.imwrite('C:/Django/myblackbox/myapp/static/thumbnails/'+str(video_index)+'.jpg', feed)
                    print("thumbnail finish")
                out.write(feed)
                if save_count == 300:
                    out.write(feed)
                    save_count = 0
                    save_title_num += 1
                    #threading.Thread(target=save_video, args=(out, title, mysqlcursor, mysqlconn)).start()
                    # save_video(out,title,mysqlcursor,mysqlconn)

                    loop.run_until_complete(save_video(out,title,mysqlcursor,mysqlconn))

                    title = 'C:/Django/myblackbox/myapp/static/videos/' + time.strftime('%y%m%d_%H%M%S') + '.mp4'
                    out = cv2.VideoWriter(title, codec, 10.0, (1920, 1080))

                    sql = "INSERT INTO videos (id, path, username,thumbnailpath) VALUES (%s, %s, %s,%s)"
                    val = (video_index, title, username, "C:/Django/myblackbox/myapp/static/thumbnails/"+str(video_index)+".jpg")
                    mysqlcursor.execute(sql, val)
                    mysqlconn.commit()

                cv2.imshow('image', feed)
                imgData = imgData[b + 2:]

                # msg = "finish\n"
                #
                # if lane_flag == True:
                #     conn.send(msg.encode())

                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
                    loop.close()
                    cv2.destroyAllWindows()
                    cap.release()
                    mysqlconn.close()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        #save_video(out, title, mysqlcursor, mysqlconn)
        #mysqlconn.close()
        sys.exit()
    except SocketError:
        #save_video(out, title, mysqlcursor, mysqlconn)
        # mysqlconn.close()
        sys.exit()
    except ConnectionResetError:
        #save_video(out, title, mysqlcursor, mysqlconn)
        # mysqlconn.close()
        sys.exit()

