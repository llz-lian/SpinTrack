from inspect import getgeneratorlocals
from turtle import forward
from typing import Set
import torch
import torch.nn as nn
import torch.nn.functional as F
from lossutil.misc import accuracy
import lossutil.box_ops as box_ops
from lossutil.matcher import build_matcher,gaussian_heatmap,build_trackingmatcher,get_gaussian_maps
import numpy as np
def cxcyToxyxy(bbox):
    """
        bbox:[B,num_box,(cx,cy,w,h)] => [B,num_box,(x,y,x+w,x+h)]
    """
    bbox[:,:,0] = bbox[:,:,0] - bbox[:,:,2]/2
    bbox[:,:,1] = bbox[:,:,1] - bbox[:,:,3]/2
    bbox[:,:,2] = bbox[:,:,0] + bbox[:,:,2]
    bbox[:,:,3] = bbox[:,:,1] + bbox[:,:,3]
    return bbox 

def caliou(bboxA,bboxB):
    """
    

    Args:
        bboxA ([type]): [x1,y1,x2,y2] pred
        bboxB ([type]): [x1,y1,x2,y2] target

    Returns:
        [type]: [description]
    """
    B,box_num,_ = bboxA.shape
    #bboxB = bboxB.unsqueeze(1)
    # bboxB = bboxB.expand(B,box_num,_)
    upleft = torch.max(bboxA[:,:,:2],bboxB[:,:,:2])
    downright = torch.min(bboxA[:,:,2:],bboxB[:,:,2:])

    inter_wh = downright - upleft
    
    inter_wh = torch.clamp(inter_wh,min = 0.0)
    
    inter = inter_wh[:,:,0] * inter_wh[:,:,1]
    
    area_gt = (bboxA[:,:,2]-bboxA[:,:,0]) * (bboxA[:,:,3] - bboxA[:,:,1])
    area_pred = (bboxB[:,:,2]-bboxB[:,:,0]) * (bboxB[:,:,3] - bboxB[:,:,1])
    
    union = area_gt + area_pred - inter
    
    return inter/union

def qfocal_loss(pred,target,beta = 2,use_sigmoid = True):
    assert pred.size() == target.size()
    pred = pred.float()
    target = target.type_as(pred)
    pred_sigmoid = pred
    if use_sigmoid:
        pred_sigmoid = pred.sigmoid()
    focal_weight = torch.abs(torch.pow((pred_sigmoid-target),beta))
    loss = F.binary_cross_entropy_with_logits(
        pred, target, reduction='none') * focal_weight
    return loss

def varifocal_loss(pred,
                   target,
                   alpha=0.85,
                   gamma=2.0,
                   iou_weighted=True,
                   use_sigmoid=True):
    """`Varifocal Loss <https://arxiv.org/abs/2008.13367>`_
    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the
            number of classes
        target (torch.Tensor): The learning target of the iou-aware
            classification score with shape (N, C), C is the number of classes.
        alpha (float, optional): A balance factor for the negative part of
            Varifocal Loss, which is different from the alpha of Focal Loss.
            Defaults to 0.75.
        gamma (float, optional): The gamma for calculating the modulating
            factor. Defaults to 2.0.
        iou_weighted (bool, optional): Whether to weight the loss of the
            positive example with the iou target. Defaults to True.
        use_sigmoid (bool, optional): Whether the prediction is
            used for sigmoid or softmax. Defaults to True.
    """
    # pred and target should be of the same size
    assert pred.size() == target.size()
    pred = pred.float()
    target = target.type_as(pred)
    if use_sigmoid:
        pred_sigmoid = pred.sigmoid()
    else:
        pred_sigmoid = pred
    
    if iou_weighted:
        focal_weight = target * (target > 0.0).float() + \
            alpha * (pred_sigmoid - target).abs().pow(gamma) * \
            (target <= 0.0).float()
        #iou>0: q + a*|p-q|^γ*0
        #iou=0: 0 + a*|p-q|^γ*1
    else:
        focal_weight = (target > 0.0).float() + \
            alpha * (pred_sigmoid - target).abs().pow(gamma) * \
            (target <= 0.0).float()
        #iou>0: 1 + a*|p-q|^γ*0
        #iou=0: 0 + a*|p-q|^γ*1
    loss = F.binary_cross_entropy_with_logits(
        pred, target, reduction='none') * focal_weight
    #iou>0: binary_cross_entropy(p,q) ，q * binary_cross_entropy(p,q)
    #iou=0: binary_cross_entropy(p,q) * a * |p-q|^γ
    #p=0 q=0 ==> loss = 0
    #p=1 q=0 ==> loss = binary_cross_entropy(p,q) * a * p^γ
    #between ==> loss = binary_cross_entropy(p,q) * a *|p-q|^γ
    return loss


class VarifocalLoss(nn.Module):
    def __init__(self,
                 use_sigmoid=True,
                 alpha=0.75,
                 gamma=2.0,
                 iou_weighted=True):
        """`Varifocal Loss <https://arxiv.org/abs/2008.13367>`_
        Args:
            use_sigmoid (bool, optional): Whether the prediction is
                used for sigmoid or softmax. Defaults to True.
            alpha (float, optional): A balance factor for the negative part of
                Varifocal Loss, which is different from the alpha of Focal
                Loss. Defaults to 0.75.
            gamma (float, optional): The gamma for calculating the modulating
                factor. Defaults to 2.0.
            iou_weighted (bool, optional): Whether to weight the loss of the
                positive examples with the iou target. Defaults to True.
        """
        super(VarifocalLoss, self).__init__()
        assert alpha >= 0.0
        self.use_sigmoid = use_sigmoid
        self.alpha = alpha
        self.gamma = gamma
        self.iou_weighted = iou_weighted

    def forward(self,
                pred,
                target):
        """Forward function.
        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction.
        Returns:
            torch.Tensor: The calculated loss
        """
        return varifocal_loss(
            pred,
            target,
            alpha=self.alpha,
            gamma=self.gamma,
            iou_weighted=self.iou_weighted,
            use_sigmoid=self.use_sigmoid)


def gaussian_map_loss(preds,maps,beta=4 ,alpha=2 ,sigmoid = True):
    # preds:[B,N,1]
    # maps[B,N]
    preds = preds.squeeze(-1)
    if sigmoid:
        preds = preds.sigmoid()
    preds = preds.float()
    maps = maps.float()
    one_idx = maps.eq(1.)
    preds = preds.clamp(min=0.000001,max=1)
    #not ones
    preds_1 = (1-preds).clamp(min=0.000001,max=1)
    loss = torch.pow((1-maps),beta) * torch.pow(preds,alpha) * torch.log(preds_1)
    loss[one_idx] = 0
    #ones
    loss_ones = torch.pow(1-preds[one_idx],alpha) * torch.log(preds[one_idx])
    loss_ones = loss_ones.mean()
    loss = loss.mean()
    return -(loss+loss_ones)


class SetCriterion(nn.Module):
    """ This class computes the loss for TransT.
    The process happens in two steps:
        1) we compute assignment between ground truth box and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, always be 1 for single object tracking.
            matcher: module able to compute a matching between target and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        #self.ious  = None

        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

    def loss_labels(self, outputs, targets, indices, num_boxes, maps=None,bboxs = None):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])

        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'varifocal': loss_ce}

        #if log:
            # TODO this should probably be a separate loss, not hacked in this one here
        #    losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    def loss_gaussian(self,outputs, targets, indices, num_boxes, maps=None,bboxs = None):
        losses = {}
        pred = outputs['pred_logits']
        loss = gaussian_map_loss(pred,maps)
        losses['varifocal'] = loss
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes,maps,bboxs):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """

        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]

        # src_logits = maps[idx].unsqueeze(-1)#[num_boxes,1]
        # src_logits = torch.clamp(src_logits,min = 0.01,max=1.)

        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')#[num_boxes,4]
        losses = {}

        # loss_bbox = loss_bbox * src_logits

        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        # giou, iou = box_ops.generalized_box_iou(
        #     box_ops.box_cxcywh_to_xyxy(src_boxes),
        #     box_ops.box_cxcywh_to_xyxy(target_boxes))

        # giou = torch.diag(giou)
        # iou = torch.diag(iou)
        # loss_giou = 1 - giou
        # iou = iou

        giou,iou = box_ops.bbox_alpha_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes),
            GIoU = True,
            alpha = 1
        )
        loss_giou = 1 - giou
        iou = iou

        # loss_giou = loss_giou * src_logits.squeeze(-1)

        losses['loss_giou'] = loss_giou.sum() / num_boxes
        losses['iou'] = iou.sum() / num_boxes
        #losses['min_iou'] = iou.min()
        return losses

    def loss_vari(self,outputs, targets, indices, num_boxes,maps,bboxs = None):
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        # num_queries = src_boxes.shape[1]
        src_boxes = src_boxes.unsqueeze(0)

        # target_boxes = bboxs[:, None, :].repeat((1, num_queries, 1)).view(-1, 4)
        # src_boxes = src_boxes.flatten(1,2)
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        target_boxes = target_boxes.unsqueeze(0)
        boxes_q = caliou(cxcyToxyxy(src_boxes),cxcyToxyxy(target_boxes)).squeeze(0)


        src_logits = outputs['pred_logits']
        B,_,_ = src_logits.shape
        target_classes = torch.full(src_logits.shape[:2], 0,
                                    dtype=torch.float32, device=src_logits.device)
        target_classes[idx] = boxes_q#[B,576]
        # target_classes = boxes_q.reshape(B,-1)
        vari_loss_sum = varifocal_loss(src_logits[:,:,0],target_classes,use_sigmoid=True).mean()
        losses = {}
        losses['varifocal'] = vari_loss_sum
        losses['weight'] = 0
        if 'pred_weights' in outputs:
            weight = outputs['pred_weights'].float().softmax(dim = -1)
            B,shape = weight.shape
            maps = maps.type_as(outputs['pred_weights'][0])
            maps = maps.softmax(dim = -1)
            # s = maps.sum(dim = -1)
            # s_ = weight.sum(dim = -1)
            # weight_loss = (1 - F.normalize(maps)@F.normalize(weight).T)#.mean(dim = -1)
            # weight_loss = torch.diag(weight_loss)

            weight_loss = F.l1_loss(weight,maps,reduction='none')
            losses['weight'] = weight_loss.sum(dim = -1).mean(dim = -1)
        return losses
        
    def loss_qfl(self,outputs, targets, indices, num_boxes,maps):
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        #boxes_p = outputs['pred_logits'][idx][:,0]#.sigmoid()

        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        src_boxes = src_boxes.unsqueeze(0)
        target_boxes = target_boxes.unsqueeze(0)
        boxes_q = caliou(cxcyToxyxy(src_boxes),cxcyToxyxy(target_boxes)).squeeze(0)
        #vari_loss = boxes_q * F.binary_cross_entropy_with_logits(boxes_p,boxes_q,reduction='none')
        #vari_loss_sum = vari_loss.sum()


        src_logits = outputs['pred_logits']
        target_classes = torch.full(src_logits.shape[:2], 0,
                                    dtype=torch.float32, device=src_logits.device)
        target_classes[idx] = boxes_q#[B,576]

        vari_loss_sum = qfocal_loss(src_logits[:,:,0],target_classes).mean()
        losses = {}
        losses['qfl'] = vari_loss_sum
        return losses
    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        #[([11,12,14,16],[0,0,0,0])]
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])#i:idx src:[11,12,14,16] _:[0...]
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx
 
    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes,maps,bboxs):
        loss_map = {
            'labels': self.loss_labels,
            'boxes': self.loss_boxes,
            'vari':self.loss_vari,
            'gloss':self.loss_gaussian,
            'qfl':self.loss_qfl
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes,maps,bboxs)

    def set_vari(self):
        self.losses = ['boxes','vari']
    def set_ce(self):
        self.losses = ['gloss']
        
    def getIndices(self,outputs,targets,get_maps = True):
        if len(outputs) == 0:
            return 0,0
        # outputs = outputs[0]
        # outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        # Retrieve the matching between the outputs of the last layer and the target
        indices,gassian_maps = self.matcher(outputs, targets)
        if get_maps:
            gassian_maps = get_gaussian_maps(outputs,targets)
        return indices,gassian_maps

    def forward(self, outputs, targets,bboxs=None,indices = None,gassian_maps = None):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        # outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # # Retrieve the matching between the outputs of the last layer and the target
        # indices,gassian_maps = self.matcher(outputs_without_aux, targets)
        # gassian_maps = get_gaussian_maps(outputs,targets)
        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes_pos = sum(len(t[0]) for t in indices)

        num_boxes_pos = torch.as_tensor([num_boxes_pos], dtype=torch.float, device=next(iter(outputs.values())).device)

        num_boxes_pos = torch.clamp(num_boxes_pos, min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes_pos,gassian_maps,bboxs))

        return losses


class Vitloss(nn.Module):
    def __init__(self,matcher) -> None:
        super().__init__()
        self.matcher = matcher
        self.weight = torch.nn.CosineEmbeddingLoss(reduction='mean')
    def loss_gaussian(self,outputs,maps):
        pred = outputs['pred_logits'].flatten(1)
        loss = gaussian_map_loss(pred,maps)
        return loss

    def get_gaussian_maps(self,bs,len_feature,targets):
        gaussian_maps = torch.zeros([1,len_feature,len_feature])
        a = np.arange(0, len_feature**2, 1)

        for i in range(bs):
            cx, cy, w, h = targets[i]['boxes'][0]
            cx = cx.item(); cy = cy.item(); w = w.item(); h = h.item()
            gaussian_map = gaussian_heatmap((int(cx*len_feature),int(cy*len_feature)),axis=a[0:len_feature],sig_x=w*len_feature/2,sig_y=h*len_feature/2)
            gaussian_maps = torch.cat([gaussian_maps,gaussian_map],dim = 0)

        return gaussian_maps[1:,:,:].flatten(1,2).cuda()

    def weight_loss(self,pred,maps):
        # maps[maps<0.1]
        maps = maps.softmax(dim = -1)
        maps = maps.type_as(pred['pred_weights'][0])
        loss = 0
        for weight in pred['pred_weights']:
            loss = loss + (1 - F.normalize(maps)@F.normalize(weight).T).mean(dim = -1)
        loss = loss / 3
        return loss.mean(dim = -1)


    def forward(self,pred,bbox_indices,bbox):
        losses = {}
        src_boxes = pred['pred_boxes']
        src_cls = pred['pred_logits']
        B,_,L,L = src_cls.shape
        gassian_maps = self.get_gaussian_maps(B,L,bbox_indices)
        losses['gloss'] = self.loss_gaussian(pred,gassian_maps)
        losses['weight'] = self.weight_loss(pred,gassian_maps)
        loss_bbox = F.l1_loss(src_boxes, bbox, reduction='none')
        losses['loss_bbox'] = loss_bbox.sum() / B
        giou,iou = box_ops.bbox_alpha_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(bbox),
            CIoU = True,
            alpha = 3
        )
        loss_giou = 1 - giou

        losses['loss_giou'] = loss_giou.sum()/B
        losses['iou'] = iou.sum()/B
        return losses


def getCriterionvit():
    matcher = build_matcher()
    return Vitloss(matcher)

def getCriterion(): 
    weight_dict = {'loss_ce': 8.5, 'loss_bbox': 5}
    weight_dict['loss_giou'] = 2
    losses = ['boxes','vari']
    matcher = build_trackingmatcher()

    return SetCriterion(1, matcher=matcher, weight_dict=weight_dict,eos_coef=0.08, losses=losses)

def getCriterion2(): 
    weight_dict = {'loss_ce': 8.5, 'loss_bbox': 5}
    losses = ['labels']
    matcher = build_matcher()
    return SetCriterion(1, matcher=matcher, weight_dict=weight_dict,eos_coef=0.08, losses=losses)


def convertLabel(cls_label_v,bbox,bs):
    cls_label = ['labels']*bs
    boxes = ['boxes']*bs
    zip1 = zip(cls_label,cls_label_v)
    zip2 = zip(boxes,bbox)
    dic1 = [{k:v} for k,v in zip1]
    dic2 = [{k:v} for k,v in zip2]
    zip3 = zip(dic1,dic2)
    [v.update(k) for k,v in zip3]
    return dic2


"""
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self):
        super(LabelSmoothingCrossEntropy, self).__init__()
    def forward(self, x, target, smoothing=0.1):
        confidence = 1. - smoothing
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + smoothing * smooth_loss
        return loss.mean()


def spread1D(vector:torch.Tensor,index:int,direction:int,rate:float):
    shape = vector.shape
    limit = 1 if direction<0 else shape[-1]-1
    while index!=limit:
        num = rate * vector[:,:,index]
        vector[:,:,index] -= num
        index += direction
        vector[:,:,index] = num
    return vector

def allspread1D(vector:torch.Tensor,index:int,rate:float):
    vector = spread1D(vector,index,-1,rate)
    vector = spread1D(vector,index,1,rate)
    return vector


def spread2D(grid:torch.Tensor,center:list,ep:float,water:float):
    x,y = center
    H,W = grid.shape
    up,down,left,right = y,(H-y),x,(W-x)
    max_r = max(up,down,left,right)

    temp_grid = torch.zeros([H,W])
    reg_num = []
    for i in range(max_r):
        _,num = getRegNum(temp_grid,center,i)
        reg_num.append(num)
    reg_num[0] = 1
    print(reg_num)
    for i in range(1,max_r):
        grid , _ = getRegNum(grid,center,i-1,(1-ep) * water/reg_num[i-1])
        water = ep * water
        grid , _ = getRegNum(grid,center,i,water/reg_num[i])

    return grid



def getRegNum(grid:torch.Tensor,center:list,r:float,num:float = 0):
    W,H = grid.shape
    x,y = center
    corner = [0,1,2,4]
    arrow = 0
    ret = 0
    if y+r < H:
        #down
        arrow += 1
        left = x if x<r else r
        right = W-x if W-x<r else r
        grid[x-left:x+right+1,y+r] = num
        ret += (left + right+1)
    if y-r >=0:
        #up
        arrow += 1
        left = x if x<r else r
        right = W-x if W-x<r else r
        grid[x-left:x+right,y-r] = num
        ret += (left + right+1)        
    if x-r >=0:
        #left
        arrow += 1
        up = y if y<r else r
        down = H-y if H-y<r else r
        grid[x-r,y-up:y+down] = num
        ret += (up+down+1)
    if x+r < W:
        #right
        arrow += 1
        up = y if y<r else r
        down = H-y if H-y<r else r
        grid[x+r,y-up:y+down] = num
        ret += (up+down+1)
    if arrow == 0:
        return None
    return grid,ret - corner[arrow-1]


def convertToPoint(vector:torch.Tensor)->list:
    shape = vector.shape
    L = shape[-1]
    line = torch.linspace(1,L,L) - 1
    pos_1d = torch.sum(line * vector) * 64
    pos_1d = pos_1d.detach().numpy()
    y = pos_1d % 256
    x = (pos_1d - y)/256
    return [x,y]


def convertToOrigin(point:list,x_scale:float,y_scale:float,img_size:list = None)->list:
    x,y = point
    x = x * x_scale
    y = y * y_scale
    return [x.astype(int),y.astype(int)]

def getBbox(left_p,right_p,x_scale,y_scale):
    left_p = convertToPoint(left_p)
    right_p = convertToPoint(right_p)
    left_p =  convertToOrigin(left_p,x_scale,y_scale)
    right_p = convertToOrigin(right_p,x_scale,y_scale)
    return [left_p,right_p]
"""

if __name__ == "__main__":
    boxa = torch.Tensor(
       [[[150,100,100,100],
         [150,100,100,100],
         [100,150,100,100],
         [100,150,100,100]],
        [[100,150,50,50],
         [100,150,50,50],
         [100,150,50,50],
         [100,150,50,50]]]
    )/200
    classa = torch.Tensor(
        [[[1,0],
          [1,0],
          [1,0],
          [1,0]],
         [[0,1],
          [0,1],
          [1,0],
          [1,0]]]
    )
    target = []
    index1 = {'labels':
        torch.Tensor([0,0]).long(),'boxes':
        torch.Tensor([[150,100,100,100],
                      [150,100,100,100]])/200}
    index2 = {'labels':
        torch.Tensor([0,0]).long(),'boxes':
        torch.Tensor([[100,150,50,50],
                      [100,150,50,50]])/200}
    target.append(index1)
    target.append(index2)
    
    output = []
    out = {'pred_boxes':boxa,'pred_logits':classa}
    
    
    weight_dict = {'loss_ce': 8.334, 'loss_bbox': 5}
    weight_dict['loss_giou'] = 2
    losses = ['labels', 'boxes']
    matcher = build_matcher()
    
    cit = SetCriterion(1, matcher=matcher, weight_dict=weight_dict,
                             eos_coef=0.0625, losses=losses)
    losses = cit(out,target)
    print(losses)
    
    
