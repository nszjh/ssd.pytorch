import torch
from torch.autograd import Function
from ..box_utils import decode, nms, decode_landmark
from data.config import celeba as cfg


class Detect(Function):
    """At test time, Detect is the final layer of SSD.  Decode location preds,
    apply non-maximum suppression to location predictions based on conf
    scores and threshold to a top_k number of output predictions for both
    confidence score and locations.
    """
    def __init__(self, num_classes, bkg_label, top_k, conf_thresh, nms_thresh):
        self.num_classes = num_classes
        self.background_label = bkg_label
        self.top_k = top_k
        # Parameters used in nms.
        self.nms_thresh = nms_thresh
        if nms_thresh <= 0:
            raise ValueError('nms_threshold must be non negative.')
        self.conf_thresh = conf_thresh
        self.variance = cfg['variance']

    def forward(self, all_data, conf_data, prior_data):
        """
        Args:
            loc_data: (tensor) Loc preds from loc layers
                Shape: [batch,num_priors*4]
            conf_data: (tensor) Shape: Conf preds from conf layers
                Shape: [batch*num_priors,num_classes]
            prior_data: (tensor) Prior boxes and variances from priorbox layers
                Shape: [1,num_priors,4]
        """
        loc_data = all_data[:,:, 12:]
        landmark_data = all_data[:,:, :10]

        num = loc_data.size(0)  # batch size
        num_priors = prior_data.size(0)
        output = torch.zeros(num, self.num_classes, self.top_k, 5)
        conf_preds = conf_data.view(num, num_priors,
                                    self.num_classes).transpose(2, 1)

        print ("loc_data: ", loc_data)
        print ("conf_data", conf_data)

        # Decode predictions into bboxes.
        for i in range(num):
            print ("loc:", loc_data[i].shape)
            print ("prior_data:", prior_data.shape)
            decoded_boxes = decode(loc_data[i], prior_data, self.variance)
            print ("decoded_boxes", decoded_boxes)


            decoded_landmarks = decode_landmark(landmark_data[i], prior_data, self.variance)

            # For each class, perform nms
            conf_scores = conf_preds[i].clone()

            print ("conf_scores", conf_scores)

            for cl in range(1, self.num_classes):
                print ("start ....")
                c_mask = conf_scores[cl].gt(self.conf_thresh)
                scores = conf_scores[cl][c_mask]
                if scores.size(0) == 0:
                    continue

                print ("c_mask: ", c_mask)
                l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
                boxes = decoded_boxes[l_mask].view(-1, 4)
                # idx of highest scoring and non-overlapping boxes per class
                ids, count = nms(boxes, scores, self.nms_thresh, self.top_k)
                output[i, cl, :count] = \
                    torch.cat((scores[ids[:count]].unsqueeze(1),
                               boxes[ids[:count]]), 1)
                
                land_mask = c_mask.unsqueeze(1).expand_as(decoded_landmarks)
                lanmarks = decoded_landmarks[land_mask].view(-1, 10)
                print (lanmarks)

        flt = output.contiguous().view(num, -1, 5)
        _, idx = flt[:, :, 0].sort(1, descending=True)
        _, rank = idx.sort(1)
        flt[(rank < self.top_k).unsqueeze(-1).expand_as(flt)].fill_(0)
        return output
