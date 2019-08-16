from __future__ import print_function
import torch
from torch.autograd import Variable
import cv2
import time
# from imutils.video import FPS, WebcamVideoStream
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Single Shot MultiBox Detection')
parser.add_argument('--weights', default='weights/blaze_face_with_land_50000_loss_0.115669921041.pth',
                    type=str, help='Trained state_dict file path')
parser.add_argument('--cuda', default=False, type=bool,
                    help='Use cuda in live demo')
args = parser.parse_args()

COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
FONT = cv2.FONT_HERSHEY_SIMPLEX


def cv2_demo(net):
    def predict(frame):

        tran_frame = frame.astype(np.float32)
        height, width = tran_frame.shape[:2]
        x = torch.from_numpy(tran_frame).permute(2, 0, 1)

        print (x.shape)
        print (width, height)
  
        x = Variable(x.unsqueeze(0))
        y = net(x)  # forward pass
        detections = y.data

        print (detections)
        # scale each detection back up to the image
        scale = torch.Tensor([width, height, width, height])
        for i in range(detections.size(1)):
            j = 0
            while detections[0, i, j, 0] >= 0.6:
                pt = (detections[0, i, j, 1:] * scale).cpu().numpy()
                cv2.rectangle(frame,
                              (int(pt[0]), int(pt[1])),
                              (int(pt[2]), int(pt[3])),
                              COLORS[i % 3], 2)
                # cv2.putText(frame, labelmap[i - 1], (int(pt[0]), int(pt[1])),
                #             FONT, 2, (255, 255, 255), 2, cv2.LINE_AA)
                j += 1
        cv2.imshow('frame', frame)
        cv2.waitKey (0)
        cv2.destroyAllWindows()

        return frame

    # start video stream thread, allow buffer to fill
    print("[INFO] starting threaded video stream...")
    # stream = WebcamVideoStream(src=0).start()  # default camera
    # time.sleep(1.0)
    # start fps timer
    # loop over frames from the video file stream
    # while True:
        # grab next frame
        # frame = stream.read()
    frame = cv2.imread("/media/nv/7174c323-375e-4334-b15e-019bd2c8af08/celeba_align/pre_img_align_celeba/celebA/000027.jpg")
    print (type(frame))

    key = cv2.waitKey(1) & 0xFF

    
    # update FPS counter
    # fps.update()
    frame = predict(frame)


        # # keybindings for display
        # if key == ord('p'):  # pause
        #     while True:
        #         key2 = cv2.waitKey(1) or 0xff
        #         cv2.imshow('frame', frame)
        #         if key2 == ord('p'):  # resume
        #             break
        # cv2.imshow('frame', frame)
        # if key == 27:  # exit
        #     break


if __name__ == '__main__':
    import sys
    from os import path
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

    from data.__init__ import BaseTransform as labelmap
    from blazeface import build_blazeface

    net = build_blazeface('test')    # initialize SSD
    net.load_state_dict(torch.load(args.weights))
    # transform = BaseTransform(net.size, (104/256.0, 117/256.0, 123/256.0))

    # fps = FPS().start()
    cv2_demo(net.eval())
    # stop the timer and display FPS information
    # fps.stop()

    # print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    # print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

    # cleanup
    # stream.stop()
