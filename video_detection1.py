"""
Sections of this code were taken from:
https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb
"""
import numpy as np
import tensorflow as tf
import cv2
import Tracker

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# Path to frozen detection graph. This is the actual model that is used
# for the object detection.
PATH_TO_CKPT = '''D:\\trained_models\custom\\faster_rcnn_inception_v2\\frozen_inference_graph.pb'''

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = '''D:\\trained_models\\custom\\faster_rcnn_inception_v2\\label_map.pbtxt'''

NUM_CLASSES = 1


def load_graph():
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    return detection_graph


def get_output_dict(output_dict):
    # all outputs are float32 numpy arrays, so convert types as appropriate
    output_dict['num_detections'] = int(output_dict['num_detections'][0])
    output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
    output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
    output_dict['detection_scores'] = output_dict['detection_scores'][0]

    return output_dict


def get_image_tensor():
    ops = tf.get_default_graph().get_operations()
    all_tensor_names = {output.name for op in ops for output in op.outputs}
    tensor_dict = {}
    for key in [
        'num_detections', 'detection_boxes', 'detection_scores',
        'detection_classes', 'detection_masks'
    ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
            tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                tensor_name)

    image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
    return image_tensor, tensor_dict


def get_eligible_boxes_scores(output_dict, min_score_thresh=.5):
    boxes = output_dict['detection_boxes']
    scores = output_dict['detection_scores']
    eligible_boxes = []
    eligible_scores = []
    for i in range(boxes.shape[0]):
        if scores is None or scores[i] > min_score_thresh:
            eligible_boxes.append(boxes[i])
            eligible_scores.append(scores[i])

    return np.array(eligible_boxes), np.array(eligible_scores)


def detect_in_image(image_path):
    detection_graph = load_graph()
    category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
    trackers = {}
    MIN_SCORE_THRESH = .5

    with detection_graph.as_default():
        with tf.Session() as sess:
            # Get handles to input and output tensors
            image_tensor, tensor_dict = get_image_tensor()

            frame = cv2.imread(image_path)
            frame2 = np.copy(frame)

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # if frame_count % detection_rate == 0 or boxes is None:
            # Run inference
            output_dict1 = sess.run(tensor_dict, feed_dict={image_tensor: np.expand_dims(image, 0)})
            output_dict = get_output_dict(output_dict1)

            new_boxes, new_scores = get_eligible_boxes_scores(output_dict, MIN_SCORE_THRESH)

            # print('output_dict: ', output_dict)
            # print('\nnew_boxes: ', new_boxes)
            # print('\nnew_scores: ', new_scores)

            # use tracker
            # if trackers.count == 0:
            #     # no tracker yet, to initialize them
            #     for n in output_dict['num_detections']:
            #         pass
            # Visualization of the results of a detection.
            # note: perform the detections using a higher threshold
            vis_util.visualize_boxes_and_labels_on_image_array(
                frame,
                output_dict['detection_boxes'],
                output_dict['detection_classes'],
                output_dict['detection_scores'],
                category_index,
                line_thickness=6,
                skip_labels=True,
                skip_scores=True,
                use_normalized_coordinates=True,
                min_score_thresh=MIN_SCORE_THRESH)

            vis_util.visualize_boxes_and_labels_on_image_array(
                frame2,
                new_boxes,
                output_dict['detection_classes'],
                new_scores,
                category_index,
                line_thickness=6,
                skip_labels=True,
                skip_scores=True,
                use_normalized_coordinates=True,
                min_score_thresh=MIN_SCORE_THRESH)

            cv2.imwrite("D:\\temp\\frame1.jpg", frame)
            cv2.imwrite("D:\\temp\\frame2.jpg", frame2)

            print('image saved..')

    cv2.destroyAllWindows()


def detect_in_video(input_video_path, output_video_path, detection_rate=5):
    # VideoWriter is the responsible of creating a copy of the video
    # used for the detections but with the detections overlays. Keep in
    # mind the frame size has to be the same as original video.
    # out = cv2.VideoWriter('output_video_path', cv2.VideoWriter_fourcc('F', 'M', 'P', '4'), 10, (1280, 720))

    detection_graph = load_graph()
    category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
    cap = cv2.VideoCapture(input_video_path)
    frame_count = 0
    MIN_SCORE_THRESH = .5
    trackers = []
    boxes = None
    scores = None

    with detection_graph.as_default():
        with tf.Session() as sess:
            # Get handles to input and output tensors
            image_tensor, tensor_dict = get_image_tensor()

            while cap.isOpened():
                # Read the frame
                ret, frame = cap.read()
                if ret is False:
                    break

                frame_count += 1
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                if frame_count % detection_rate == 0 or boxes is None:
                    # Run inference
                    output_dict = sess.run(tensor_dict, feed_dict={image_tensor: np.expand_dims(image, 0)})
                    output_dict = get_output_dict(output_dict)
                    trackers.clear()
                    boxes.clear()
                    scores.clear()
                    boxes, scores = get_eligible_boxes_scores(output_dict, MIN_SCORE_THRESH)
                else:
                    # use tracker
                    if trackers.count == 0 and boxes.count > 0:
                        # no tracker but boxes are present, so initialize the trackers
                        for i in range(scores.count):
                            trackers.append(Tracker(cv2.TrackerCSRT_create, frame, boxes[i]))
                    else:
                        # we have trackers, so update them
                        for i in range(trackers.count):
                            success, box = trackers[i].update(frame)
                            boxes[i] = box
                # Visualization of the results of a detection.
                # note: perform the detections using a higher threshold
                vis_util.visualize_boxes_and_labels_on_image_array(
                    frame,
                    boxes,
                    output_dict['detection_classes'],
                    scores,
                    category_index,
                    line_thickness=6,
                    skip_labels=True,
                    skip_scores=True,
                    use_normalized_coordinates=True,
                    min_score_thresh=MIN_SCORE_THRESH)

                cv2.imshow('frame', frame)
                # output_rgb = cv2.cvtColor(color_frame, cv2.COLOR_RGB2BGR)
                # out.write(output_rgb)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    # out.release()
    cap.release()
    cv2.destroyAllWindows()


def main():
    input_video = '/mnt/65539378-7b4e-48e9-a5a8-a51c140029e1/retail/data/videos/VID_20181016_170708.mp4'
    output_video = '/home/muon/Code/retail/code/models-master/research/object_detection/neeraj/output_video/processed.avi'
    detect_in_video(input_video, 5)

    # detect_in_image("D:\\headcount\\11-20\\12\\fc1_2-5626.jpg")


if __name__ == '__main__':
    print('inside main')
    main()
