class Tracker:
    # available trackers are:
    # cv2.TrackerCSRT_create,
    # cv2.TrackerKCF_create,
    # cv2.TrackerBoosting_create,
    # cv2.TrackerMIL_create,
    # cv2.TrackerTLD_create,
    # cv2.TrackerMedianFlow_create,
    # cv2.TrackerMOSSE_create   
    def __init__(self, tracker_type, frame, box):
        # grab the appropriate object tracker using our dictionary of
        # OpenCV object tracker objects
        self.tracker = tracker_type()

        # initialize the bounding box coordinates of the object we are going
        # to track
        self.box = box

        # start OpenCV object tracker using the supplied bounding box coordinates
        self.tracker.init(frame, box)

    def update(self, frame):
        # grab the new bounding box coordinates of the object
        success, box = self.tracker.update(frame)
        return success, box
