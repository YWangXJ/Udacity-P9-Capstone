import datetime
from styx_msgs.msg import TrafficLight
import tensorflow as tf
import numpy as np


class TLClassifier(object):
    def __init__(self, is_site):
        # load classifier according to is_site
        if is_site:
            path = 'light_classification/model/Real/frozen_inference_graph.pb'
        else:
            path = 'light_classification/model/Sim/frozen_inference_graph.pb'

        # initialize tf
        self.graph = tf.Graph()
        self.threshold = .5

        # load tensors
        with self.graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(path, 'rb') as fid:
                od_graph_def.ParseFromString(fid.read())
                tf.import_graph_def(od_graph_def, name='')

            self.image_tensor = self.graph.get_tensor_by_name('image_tensor:0')
            self.boxes = self.graph.get_tensor_by_name('detection_boxes:0')
            self.scores = self.graph.get_tensor_by_name('detection_scores:0')
            self.classes = self.graph.get_tensor_by_name('detection_classes:0')
            self.num_detections = self.graph.get_tensor_by_name('num_detections:0')

        self.sess = tf.Session(graph=self.graph)

    def final_state(self, class_id):
        if class_id == 1:
            return TrafficLight.GREEN
        elif class_id == 2:
            return TrafficLight.RED
        elif class_id == 3:
            return TrafficLight.YELLOW
        else:
            return TrafficLight.UNKNOWN

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        # TODO implement light color prediction
        with self.graph.as_default():
            img_expand = np.expand_dims(image, axis=0)

        boxes, scores, classes, num_detections = self.sess.run(
            [self.boxes, self.scores, self.classes, self.num_detections],
            feed_dict={self.image_tensor: img_expand})

        boxes = np.squeeze(boxes)
        scores = np.squeeze(scores)
        classes = np.squeeze(classes).astype(np.int32)

        # return corresponding light states
        if scores[0] > self.threshold and num_detections > 0:
            return self.final_state(classes[0])

        # return unknown if nothing detected
        return TrafficLight.UNKNOWN
