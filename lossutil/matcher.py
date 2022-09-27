# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
import numpy as np
from lossutil.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
def gaussian_heatmap(center = (2, 2), axis = 0, sig_x = 1,sig_y = 1):
    """
    It produces single gaussian at expected center
    :param center:  the mean position (X, Y) - where high value expected
    :param image_size: The total image size (width, height)
    :param sig: The sigma value
    :return:
    """
    x_axis = axis - center[0]
    y_axis = axis - center[1]
    xx, yy = np.meshgrid(x_axis, y_axis)
    kernel = np.exp(-0.5 * (np.square(xx)/(np.square(sig_x)+1e-6) + np.square(yy)/ (np.square(sig_y)+1e-6)))
    kernel = torch.from_numpy(kernel).unsqueeze(0)
    return kernel

class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 10, cost_bbox: float = 1, cost_giou: float = 1):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        #print(out_prob[:,tgt_ids])
        cost_class = -out_prob[:, tgt_ids]

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # Compute the giou cost betwen boxes
        cost_giou = - generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))[0]

        # Final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

class TrackingMatcher(nn.Module):
    """This class computes an assignment between the ground-truth and the predictions of the network.
    The corresponding feature vectors within the ground-truth box are matched as positive samples.
    """

    def __init__(self):
        super().__init__()
        self.sum = 0

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Always tensor([0]) represents the foreground,
                           since single target tracking has only one foreground category
                 "boxes": Tensor of dim [1, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order),
                  and it is always 0, because single target tracking has only one target per image
            For each batch element, it holds:
                len(index_i) = len(index_j)
        """
        indices = []
        bs, num_queries = outputs["pred_logits"].shape[:2]
        len_feature = int(np.sqrt(num_queries))
        a = np.arange(0, num_queries, 1)
        b = a.reshape([len_feature, len_feature])
        for i in range(bs):
            cx, cy, w, h = targets[i]['boxes'][0]
            cx = cx.item(); cy = cy.item(); w = w.item(); h = h.item()
            xmin = cx-w/2; ymin = cy-h/2; xmax = cx+w/2; ymax = cy+h/2
            Xmin = int(np.around(xmin*len_feature))
            Ymin = int(np.around(ymin*len_feature))
            Xmax = int(np.around(xmax*len_feature)) 
            Ymax = int(np.around(ymax*len_feature))
            if Xmin == Xmax:
                Xmax = Xmax+1
            if Ymin == Ymax:
                Ymax = Ymax+1
            Xmin = Xmin * (Xmin>0)
            Ymin = Ymin * (Ymin>0)

            #0,1,2,3 
            #4,5,6,7
            #8,9,10,11
            #12,13,14,15
            c = b[Ymin:Ymax,Xmin:Xmax].flatten()#[10,11,14,15]
#             if 'drop_index' in outputs:
#                 c = np.setdiff1d(c,outputs['drop_index'][i])

            d = np.zeros(len(c), dtype=int)#[0,0,0,0]
            indice = (c,d)
            indices.append(indice)

        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices],0


class AllMatcher(nn.Module):
    """This class computes an assignment between the ground-truth and the predictions of the network.
    The corresponding feature vectors within the ground-truth box are matched as positive samples.
    """

    def __init__(self):
        super().__init__()
        self.sum = 0

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Always tensor([0]) represents the foreground,
                           since single target tracking has only one foreground category
                 "boxes": Tensor of dim [1, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order),
                  and it is always 0, because single target tracking has only one target per image
            For each batch element, it holds:
                len(index_i) = len(index_j)
        """
        indices = []
        bs, num_queries = outputs["pred_logits"].shape[:2]
        len_feature = int(np.sqrt(num_queries))
        gaussian_maps = torch.zeros([1,len_feature,len_feature])
        a = np.arange(0, num_queries, 1)
        b = a.reshape([len_feature, len_feature])
        c = b[:].flatten()#[10,11,14,15]
        d = np.zeros(len(c), dtype=int)#[0,0,0,0]
        indice = (c,d)
        for i in range(bs):
            indices.append(indice)
            cx, cy, w, h = targets[i]['boxes'][0]
            cx = cx.item(); cy = cy.item(); w = w.item(); h = h.item()
            gaussian_map = gaussian_heatmap((int(cx*len_feature),int(cy*len_feature)),axis=a[0:len_feature],sig_x=w*len_feature/2,sig_y=h*len_feature/2)
            gaussian_maps = torch.cat([gaussian_maps,gaussian_map],dim = 0)
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices],gaussian_maps[1:,:,:].flatten(1,2).cuda()

def get_gaussian_maps(outputs, targets):
    bs, num_queries = outputs["pred_logits"].shape[:2]
    len_feature = int(np.sqrt(num_queries))
    gaussian_maps = torch.zeros([1,len_feature,len_feature])
    a = np.arange(0, num_queries, 1)
    for i in range(bs):
            cx, cy, w, h = targets[i]['boxes'][0]
            cx = cx.item(); cy = cy.item(); w = w.item(); h = h.item()
            gaussian_map = gaussian_heatmap((int(cx*len_feature),int(cy*len_feature)),axis=a[0:len_feature],sig_x=w*len_feature/2,sig_y=h*len_feature/2)
            gaussian_maps = torch.cat([gaussian_maps,gaussian_map],dim = 0)
    return gaussian_maps[1:,:,:].flatten(1,2).cuda()



class MapMatcher(nn.Module):
    """This class computes an assignment between the ground-truth and the predictions of the network.
    The corresponding feature vectors within the ground-truth box are matched as positive samples.
    """

    def __init__(self):
        super().__init__()
        self.sum = 0

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Always tensor([0]) represents the foreground,
                           since single target tracking has only one foreground category
                 "boxes": Tensor of dim [1, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order),
                  and it is always 0, because single target tracking has only one target per image
            For each batch element, it holds:
                len(index_i) = len(index_j)
        """
        indices = []
        bs, num_queries = outputs["pred_boxes"].shape[:2]
        len_feature = int(np.sqrt(num_queries))
        for i in range(bs):
            cx, cy, w, h = targets[i]['boxes'][0]
            cx = cx.item(); cy = cy.item(); w = w.item(); h = h.item()
            xmin = cx-w/2; ymin = cy-h/2; xmax = cx+w/2; ymax = cy+h/2
            Xmin = int(np.ceil(xmin*len_feature))
            Ymin = int(np.ceil(ymin*len_feature))
            Xmax = int(np.ceil(xmax*len_feature)) 
            Ymax = int(np.ceil(ymax*len_feature))
            if Xmin == Xmax:
                Xmax = Xmax+1
            if Ymin == Ymax:
                Ymax = Ymax+1
            Xmin = Xmin * (Xmin>0)
            Ymin = Ymin * (Ymin>0)

            a = np.arange(0, num_queries, 1)
            b = a.reshape([len_feature, len_feature])
            #0,1,2,3 
            #4,5,6,7
            #8,9,10,11
            #12,13,14,15
            c = b[Ymin:Ymax,Xmin:Xmax].flatten()#[10,11,14,15]

            d = np.zeros(len(c), dtype=int)#[0,0,0,0]
            indice = (c,d)
            indices.append(indice)
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices],0

def build_matcher():
    #return HungarianMatcher()
    return AllMatcher()
def build_trackingmatcher():
    return TrackingMatcher()

#def build_matcher(args):
#    return HungarianMatcher(cost_class=args.set_cost_class, cost_bbox=args.set_cost_bbox, cost_giou=args.set_cost_giou)
