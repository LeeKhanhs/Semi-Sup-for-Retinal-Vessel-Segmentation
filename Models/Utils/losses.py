import torch
from torch.nn import functional as F
import numpy as np
from scipy.ndimage import distance_transform_edt as distance
from skimage import segmentation as skimage_seg


def dice_loss(score, target):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target * target)
    z_sum = torch.sum(score * score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss



def dice_loss1(score, target):
    # non-square
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target)
    z_sum = torch.sum(score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss


def iou_loss(score, target):
    target = target.float()
    smooth = 1e-5
    tp_sum = torch.sum(score * target)
    fp_sum = torch.sum(score * (1 - target))
    fn_sum = torch.sum((1 - score) * target)
    loss = (tp_sum + smooth) / (tp_sum + fp_sum + fn_sum + smooth)
    loss = 1 - loss
    return loss


def entropy_loss(p, C=2):
    ## p N*C*W*H*D
    y1 = -1 * torch.sum(p * torch.log(p + 1e-6), dim=1) / torch.tensor(
        np.log(C)).cuda()
    ent = torch.mean(y1)

    return ent


def softmax_dice_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss
    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    n = input_logits.shape[1]
    dice = 0
    for i in range(0, n):
        dice += dice_loss1(input_softmax[:, i], target_softmax[:, i])
    mean_dice = dice / n

    return mean_dice


def entropy_loss_map(p, C=2):
    ent = -1 * torch.sum(p * torch.log(p + 1e-6), dim=1,
                         keepdim=True) / torch.tensor(np.log(C)).cuda()
    return ent


def softmax_mse_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss
    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)

    mse_loss = (input_softmax - target_softmax)**2
    return mse_loss


def softmax_kl_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns KL divergence
    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_log_softmax = F.log_softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)

    # return F.kl_div(input_log_softmax, target_softmax)
    kl_div = F.kl_div(input_log_softmax, target_softmax, reduction='none')
    # mean_kl_div = torch.mean(0.2*kl_div[:,0,...]+0.8*kl_div[:,1,...])
    return kl_div


def symmetric_mse_loss(input1, input2):
    """Like F.mse_loss but sends gradients to both directions
    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to both input1 and input2.
    """
    assert input1.size() == input2.size()
    return torch.mean((input1 - input2)**2)


def compute_sdf01(segmentation):
    """
    compute the signed distance map of binary mask
    input: segmentation, shape = (batch_size, class, x, y, z)
    output: the Signed Distance Map (SDM) 
    sdm(x) = 0; x in segmentation boundary
             -inf|x-y|; x in segmentation
             +inf|x-y|; x out of segmentation
    """
    # print(type(segmentation), segmentation.shape)

    segmentation = segmentation.astype(np.uint8)

    if len(segmentation.shape) == 4:  # 3D image
        segmentation = np.expand_dims(segmentation, 1)
    normalized_sdf = np.zeros(segmentation.shape)
    if segmentation.shape[1] == 1:
        dis_id = 0
    else:
        dis_id = 1
    for b in range(segmentation.shape[0]):  # batch size
        for c in range(dis_id, segmentation.shape[1]):  # class_num
            # ignore background
            posmask = segmentation[b][c]
            if np.max(posmask) == 0:
                continue
            negmask = ~posmask
            posdis = distance(posmask)
            negdis = distance(negmask)
            boundary = skimage_seg.find_boundaries(
                posmask, mode='inner').astype(np.uint8)
            sdf = negdis / np.max(negdis) / 2 - posdis / np.max(
                posdis) / 2 + 0.5
            sdf[boundary > 0] = 0.5
            normalized_sdf[b][c] = sdf
    return normalized_sdf


def compute_sdf1_1(segmentation):
    """
    compute the signed distance map of binary mask
    input: segmentation, shape = (batch_size, class, x, y, z)
    output: the Signed Distance Map (SDM) 
    sdm(x) = 0; x in segmentation boundary
             -inf|x-y|; x in segmentation
             +inf|x-y|; x out of segmentation
    """
    # print(type(segmentation), segmentation.shape)

    segmentation = segmentation.astype(np.uint8)
    if len(segmentation.shape) == 4:  # 3D image
        segmentation = np.expand_dims(segmentation, 1)
    normalized_sdf = np.zeros(segmentation.shape)
    if segmentation.shape[1] == 1:
        dis_id = 0
    else:
        dis_id = 1
    for b in range(segmentation.shape[0]):  # batch size
        for c in range(dis_id, segmentation.shape[1]):  # class_num
            # ignore background
            posmask = segmentation[b][c]
            if np.max(posmask) == 0:
                continue
            negmask = ~posmask
            posdis = distance(posmask)
            negdis = distance(negmask)
            boundary = skimage_seg.find_boundaries(
                posmask, mode='inner').astype(np.uint8)
            sdf = negdis / np.max(negdis) - posdis / np.max(posdis)
            sdf[boundary > 0] = 0
            normalized_sdf[b][c] = sdf
    return normalized_sdf


def compute_fore_dist(segmentation):
    """
    compute the foreground of binary mask
    input: segmentation, shape = (batch_size, class, x, y, z)
    output: the Signed Distance Map (SDM) 
    sdm(x) = 0; x in segmentation boundary
             -inf|x-y|; x in segmentation
             +inf|x-y|; x out of segmentation
    """
    # print(type(segmentation), segmentation.shape)

    segmentation = segmentation.astype(np.uint8)
    if len(segmentation.shape) == 4:  # 3D image
        segmentation = np.expand_dims(segmentation, 1)
    normalized_sdf = np.zeros(segmentation.shape)
    if segmentation.shape[1] == 1:
        dis_id = 0
    else:
        dis_id = 1
    for b in range(segmentation.shape[0]):  # batch size
        for c in range(dis_id, segmentation.shape[1]):  # class_num
            # ignore background
            posmask = segmentation[b][c]
            posdis = distance(posmask)
            normalized_sdf[b][c] = posdis / np.max(posdis)
    return normalized_sdf


def sum_tensor(inp, axes, keepdim=False):
    axes = np.unique(axes).astype(int)
    if keepdim:
        for ax in axes:
            inp = inp.sum(int(ax), keepdim=True)
    else:
        for ax in sorted(axes, reverse=True):
            inp = inp.sum(int(ax))
    return inp


def AAAI_sdf_loss(net_output, gt):
    """
    net_output: net logits; shape=(batch_size, class, x, y, z)
    gt: ground truth; (shape (batch_size, 1, x, y, z) OR (batch_size, x, y, z))
    """
    smooth = 1e-5
    axes = tuple(range(2, len(net_output.size())))
    shp_x = net_output.shape
    shp_y = gt.shape

    with torch.no_grad():
        if len(shp_x) != len(shp_y):
            gt = gt.view((shp_y[0], 1, *shp_y[1:]))

        if all([i == j for i, j in zip(net_output.shape, gt.shape)]):
            # if this is the case then gt is probably already a one hot encoding
            y_onehot = gt
        else:
            gt = gt.long()
            y_onehot = torch.zeros(shp_x)
            if net_output.device.type == "cuda":
                y_onehot = y_onehot.cuda(net_output.device.index)
            y_onehot.scatter_(1, gt, 1)
        gt_sdm_npy = compute_sdf1_1(y_onehot.cpu().numpy())
        if net_output.device.type == "cuda":
            gt_sdm = torch.from_numpy(gt_sdm_npy).float().cuda(
                net_output.device.index)
        else:
            gt_sdm = torch.from_numpy(gt_sdm_npy).float()
    intersect = sum_tensor(net_output * gt_sdm, axes, keepdim=False)
    pd_sum = sum_tensor(net_output**2, axes, keepdim=False)
    gt_sum = sum_tensor(gt_sdm**2, axes, keepdim=False)
    L_product = (intersect + smooth) / (intersect + pd_sum + gt_sum)
    # print('L_product.shape', L_product.shape) (4,2)
    L_SDF_AAAI = -L_product.mean() + torch.norm(net_output - gt_sdm,
                                                1) / torch.numel(net_output)

    return L_SDF_AAAI


def sdf_kl_loss(net_output, gt):
    """
    net_output: net logits; shape=(batch_size, class, x, y, z)
    gt: ground truth; (shape (batch_size, 1, x, y, z) OR (batch_size, x, y, z))
    """
    smooth = 1e-5
    axes = tuple(range(2, len(net_output.size())))
    shp_x = net_output.shape
    shp_y = gt.shape

    with torch.no_grad():
        if len(shp_x) != len(shp_y):
            gt = gt.view((shp_y[0], 1, *shp_y[1:]))

        if all([i == j for i, j in zip(net_output.shape, gt.shape)]):
            # if this is the case then gt is probably already a one hot encoding
            y_onehot = gt
        else:
            gt = gt.long()
            y_onehot = torch.zeros(shp_x)
            if net_output.device.type == "cuda":
                y_onehot = y_onehot.cuda(net_output.device.index)
            y_onehot.scatter_(1, gt, 1)
        # print('y_onehot.shape', y_onehot.shape)
        gt_sdf_npy = compute_sdf(y_onehot.cpu().numpy())
        gt_sdf = torch.from_numpy(gt_sdf_npy + smooth).float().cuda(
            net_output.device.index)
    # print('net_output, gt_sdf', net_output.shape, gt_sdf.shape)
    # exit()
    sdf_kl_loss = F.kl_div(net_output,
                           gt_sdf[:, 1:2, ...],
                           reduction='batchmean')

    return sdf_kl_loss


# don't put the sample itself into the Positive set
class Supervised_Contrastive_Loss(torch.nn.Module):
    '''
    from https://github.com/GuillaumeErhard/Supervised_contrastive_loss_pytorch/blob/main/loss/spc.py
    https://blog.csdn.net/wf19971210/article/details/116715880
    Treat samples in the same labels as the positive samples, others as negative samples
    '''
    def __init__(self, temperature=0.1, device='cpu'):
        super(Supervised_Contrastive_Loss, self).__init__()
        self.temperature = temperature
        self.device = device
    
    def forward(self, projections, targets, attribute=None):
        # projections (bs, dim), targets (bs)
        # similarity matrix/T
        # dot_product_tempered = torch.mm(projections, projections.T) / self.temperature
        dot_product_tempered = F.cosine_similarity(projections.unsqueeze(1), projections.unsqueeze(0),dim=2)/self.temperature
        # print(dot_product_tempered)
        # Minus max for numerical stability with exponential. Same done in cross entropy. Epsilon added to avoid log(0)
        # exp_dot_tempered = (
        #     torch.exp(dot_product_tempered - torch.max(dot_product_tempered, dim=1, keepdim=True)[0]) + 1e-5
        # )
        exp_dot_tempered = torch.exp(dot_product_tempered- torch.max(dot_product_tempered, dim=1, keepdim=True)[0])+ 1e-5
        # a matrix, same labels are true, others are false
        mask_similar_class = (targets.unsqueeze(1).repeat(1, targets.shape[0]) == targets).to(self.device)
        # a matrix, diagonal are zeros, others are ones
        mask_anchor_out = (1 - torch.eye(exp_dot_tempered.shape[0])).to(self.device)
        mask_nonsimilar_class = ~mask_similar_class
        # mask_nonsimilar_attr = ~mask_similar_attr
        # a matrix, same labels are 1, others are 0, and diagonal are zeros
        mask_combined = mask_similar_class * mask_anchor_out
        # num of similar samples for sample
        cardinality_per_samples = torch.sum(mask_combined, dim=1)
        # print(exp_dot_tempered * mask_nonsimilar_class* mask_similar_attr)
        # print(torch.sum(exp_dot_tempered * mask_nonsimilar_class* mask_similar_attr, dim=1, keepdim=True)+exp_dot_tempered)
        if attribute != None:
            mask_similar_attr = (attribute.unsqueeze(1).repeat(1, attribute.shape[0]) == attribute).to(self.device)
            log_prob = -torch.log(exp_dot_tempered / (torch.sum(exp_dot_tempered * mask_nonsimilar_class * mask_similar_attr, dim=1, keepdim=True)+exp_dot_tempered+1e-5))
       
        else:
            log_prob = -torch.log(exp_dot_tempered / (torch.sum(exp_dot_tempered * mask_nonsimilar_class, dim=1, keepdim=True)+exp_dot_tempered+1e-5))
        supervised_contrastive_loss = torch.sum(log_prob * mask_combined)/(torch.sum(cardinality_per_samples)+1e-5)

        
        return supervised_contrastive_loss
import torch.distributed as dist
@torch.no_grad()
def gather_together(data):
    dist.barrier()

    world_size = dist.get_world_size()
    gather_data = [None for _ in range(world_size)]
    dist.all_gather_object(gather_data, data)

    return gather_data
@torch.no_grad()

def dequeue_and_enqueue(keys, queue, queue_ptr, queue_size):
    # gather keys before updating queue
    keys = keys.detach().clone().cpu()
    gathered_list = gather_together(keys)
    keys = torch.cat(gathered_list, dim=0).cuda()

    batch_size = keys.shape[0]

    ptr = int(queue_ptr)

    queue[0] = torch.cat((queue[0], keys.cpu()), dim=0)
    if queue[0].shape[0] >= queue_size:
        queue[0] = queue[0][-queue_size:, :]
        ptr = queue_size
    else:
        ptr = (ptr + batch_size) % queue_size  # move pointer

    queue_ptr[0] = ptr

    return batch_size

def compute_contra_memobank_loss(
    rep,
    label_l,
    label_u,
    prob_l,
    prob_u,
    low_mask,
    high_mask,
    cfg,
    memobank,
    queue_prtlis,
    queue_size,
    rep_teacher,
    momentum_prototype=None,
    i_iter=0,
):
    # current_class_threshold: delta_p (0.3)
    # current_class_negative_threshold: delta_n (1)
    current_class_threshold = cfg["current_class_threshold"]
    current_class_negative_threshold = cfg["current_class_negative_threshold"]
    low_rank, high_rank = cfg["low_rank"], cfg["high_rank"]
    temp = cfg["temperature"]
    num_queries = cfg["num_queries"]
    num_negatives = cfg["num_negatives"]

    num_feat = rep.shape[1]
    num_labeled = label_l.shape[0]
    num_segments = 2  # 2 classes: cancer and non-cancer

    low_valid_pixel = torch.cat((label_l, label_u), dim=0) * low_mask
    high_valid_pixel = torch.cat((label_l, label_u), dim=0) * high_mask

    rep = rep.permute(0, 2, 3, 1)
    rep_teacher = rep_teacher.permute(0, 2, 3, 1)

    seg_feat_all_list = []
    seg_feat_low_entropy_list = []  # candidate anchor pixels
    seg_num_list = []  # the number of low_valid pixels in each class
    seg_proto_list = []  # the center of each class

    _, prob_indices_l = torch.sort(prob_l, 1, True)
    prob_indices_l = prob_indices_l.permute(0, 2, 3, 1)  # (num_labeled, h, w, num_cls)

    _, prob_indices_u = torch.sort(prob_u, 1, True)
    prob_indices_u = prob_indices_u.permute(0, 2, 3, 1)  # (num_unlabeled, h, w, num_cls)

    prob = torch.cat((prob_l, prob_u), dim=0)  # (batch_size, num_cls, h, w)

    valid_classes = []
    new_keys = []
    for i in range(num_segments):  # Only 2 segments (cancer and non-cancer)
        low_valid_pixel_seg = low_valid_pixel[:, i]  # select binary mask for i-th class
        high_valid_pixel_seg = high_valid_pixel[:, i]

        prob_seg = prob[:, i, :, :]
        rep_mask_low_entropy = (prob_seg > current_class_threshold) * low_valid_pixel_seg.bool()
        rep_mask_high_entropy = (prob_seg < current_class_negative_threshold) * high_valid_pixel_seg.bool()

        seg_feat_all_list.append(rep[low_valid_pixel_seg.bool()])
        seg_feat_low_entropy_list.append(rep[rep_mask_low_entropy])

        # positive sample: center of the class
        seg_proto_list.append(
            torch.mean(
                rep_teacher[low_valid_pixel_seg.bool()].detach(), dim=0, keepdim=True
            )
        )

        # generate class mask for unlabeled data
        class_mask_u = torch.sum(
            prob_indices_u[:, :, :, low_rank:high_rank].eq(i), dim=3
        ).bool()

        # generate class mask for labeled data
        class_mask_l = torch.sum(prob_indices_l[:, :, :, :low_rank].eq(i), dim=3).bool()

        class_mask = torch.cat((class_mask_l * (label_l[:, i] == 0), class_mask_u), dim=0)

        negative_mask = rep_mask_high_entropy * class_mask

        keys = rep_teacher[negative_mask].detach()
        new_keys.append(
            dequeue_and_enqueue(
                keys=keys,
                queue=memobank[i],
                queue_ptr=queue_prtlis[i],
                queue_size=queue_size[i],
            )
        )

        if low_valid_pixel_seg.sum() > 0:
            seg_num_list.append(int(low_valid_pixel_seg.sum().item()))
            valid_classes.append(i)

    if len(seg_num_list) <= 1:  # in rare cases with small mini-batches
        if momentum_prototype is None:
            return new_keys, torch.tensor(0.0) * rep.sum()
        else:
            return momentum_prototype, new_keys, torch.tensor(0.0) * rep.sum()

    else:
        reco_loss = torch.tensor(0.0).cuda()
        seg_proto = torch.cat(seg_proto_list)  # shape: [valid_seg, 256]
        valid_seg = len(seg_num_list)  # number of valid classes

        prototype = torch.zeros((prob_indices_l.shape[-1], num_queries, 1, num_feat)).cuda()

        for i in range(valid_seg):
            if len(seg_feat_low_entropy_list[i]) > 0 and memobank[valid_classes[i]][0].shape[0] > 0:
                seg_low_entropy_idx = torch.randint(len(seg_feat_low_entropy_list[i]), size=(num_queries,))
                anchor_feat = seg_feat_low_entropy_list[i][seg_low_entropy_idx].clone().cuda()
            else:
                reco_loss = reco_loss + 0 * rep.sum()
                continue

            with torch.no_grad():
                negative_feat = memobank[valid_classes[i]][0].clone().cuda()

                high_entropy_idx = torch.randint(len(negative_feat), size=(num_queries * num_negatives,))
                negative_feat = negative_feat[high_entropy_idx]
                negative_feat = negative_feat.reshape(num_queries, num_negatives, num_feat)
                positive_feat = seg_proto[i].unsqueeze(0).unsqueeze(0).repeat(num_queries, 1, 1).cuda()

                if momentum_prototype is not None:
                    if not (momentum_prototype == 0).all():
                        ema_decay = min(1 - 1 / i_iter, 0.999)
                        positive_feat = (1 - ema_decay) * positive_feat + ema_decay * momentum_prototype[valid_classes[i]]

                    prototype[valid_classes[i]] = positive_feat.clone()

                all_feat = torch.cat((positive_feat, negative_feat), dim=1)  # (num_queries, 1 + num_negative, num_feat)

            # Sử dụng BCE Loss thay vì Cross Entropy
            # Assume `anchor_feat` is the prediction logits (before sigmoid), and targets are all 0s (positive class)
            bce_loss = F.binary_cross_entropy_with_logits(
                torch.cosine_similarity(anchor_feat.unsqueeze(1), all_feat, dim=2), torch.zeros(num_queries).cuda()
            )

            reco_loss = reco_loss + bce_loss

        if momentum_prototype is None:
            return new_keys, reco_loss / valid_seg
        else:
            return prototype, new_keys, reco_loss / valid_seg

# class Supervised_Contrastive_Loss(torch.nn.Module):
#     '''
#     from https://github.com/GuillaumeErhard/Supervised_contrastive_loss_pytorch/blob/main/loss/spc.py
#     https://blog.csdn.net/wf19971210/article/details/116715880
#     Treat samples in the same labels as the positive samples (including itself), others as negative samples
#     '''
#     def __init__(self, temperature=0.1, device='cpu'):
#         super(Supervised_Contrastive_Loss, self).__init__()
#         self.temperature = temperature
#         self.device = device
    
#     def forward(self, projections, targets, attribute=None):
#         # projections (bs, dim), targets (bs)
#         # similarity matrix/T
#         # dot_product_tempered = torch.mm(projections, projections.T) / self.temperature
#         dot_product_tempered = F.cosine_similarity(projections.unsqueeze(1), projections.unsqueeze(0),dim=2)/self.temperature
#         # print(dot_product_tempered)
#         # Minus max for numerical stability with exponential. Same done in cross entropy. Epsilon added to avoid log(0)
#         # exp_dot_tempered = (
#         #     torch.exp(dot_product_tempered - torch.max(dot_product_tempered, dim=1, keepdim=True)[0]) + 1e-5
#         # )
#         exp_dot_tempered = torch.exp(dot_product_tempered- torch.max(dot_product_tempered, dim=1, keepdim=True)[0])+ 1e-6
#         # a matrix, same labels are true, others are false
#         mask_similar_class = (targets.unsqueeze(1).repeat(1, targets.shape[0]) == targets).to(self.device)
#         # a matrix, diagonal are zeros, others are ones
#         mask_anchor_out = (1 - torch.eye(exp_dot_tempered.shape[0])).to(self.device)
#         mask_nonsimilar_class = ~mask_similar_class
#         # mask_nonsimilar_attr = ~mask_similar_attr
#         # a matrix, same labels are 1, others are 0, and diagonal are zeros
#         mask_combined = mask_similar_class * mask_anchor_out
#         # num of similar samples for sample
#         cardinality_per_samples = torch.sum(mask_similar_class, dim=1)
#         # print(exp_dot_tempered * mask_nonsimilar_class)
#         # print(torch.sum(exp_dot_tempered * mask_nonsimilar_class* mask_similar_attr, dim=1, keepdim=True)+exp_dot_tempered)
#         if attribute != None:
#             mask_similar_attr = (attribute.unsqueeze(1).repeat(1, attribute.shape[0]) == attribute).to(self.device)
#             log_prob = -torch.log(exp_dot_tempered / (torch.sum(exp_dot_tempered * mask_nonsimilar_class * mask_similar_attr, dim=1, keepdim=True)+exp_dot_tempered+1e-6))
       
#         else:
#             log_prob = -torch.log(exp_dot_tempered / (torch.sum(exp_dot_tempered * mask_nonsimilar_class, dim=1, keepdim=True)+exp_dot_tempered+1e-6))
#         supervised_contrastive_loss = torch.sum(log_prob * mask_similar_class)/(torch.sum(cardinality_per_samples)+1e-6)

        
#         return supervised_contrastive_loss




if __name__ == '__main__':

    # # check supervised contrastive loss
    # loss_func = Supervised_Contrastive_Loss()
    # # a,b = torch.tensor([[0.,0,0,0,1,1,1,1,1,1]]), torch.tensor([[1.,1,1,1,0,0,0,0,0,0]])
    # a,b  = torch.ones((3,7)), torch.ones(3,7)
    # # a,b = a.repeat((3,1)), b.repeat((3,1))
    # # a = torch.tensor([[0.,0,1,1]])
    # # a= a.repeat((6,1))
    # # a = torch.randn(3,10)
    # # b = torch.tensor([[1.,1,1,1,0,0,0,0,0,0]])
    # # x = torch.cat((a,b),dim=0)
    # x = torch.randn(6,10)

    # y = torch.tensor([1,2,3,4,5,6])
    # # z = torch.tensor([2,3,3,2,3,3])
    # loss = loss_func(x, y)
    # print(loss)

    a = torch.tensor([0.0,1.0,0.0,1.0])
    b = torch.tensor([0.0,0.0,0.0,1.0])
    # print(a)
    # print(b)
    dice = dice_per_img(a,b)
    dice_all = dice_loss(a,b)
    print(dice.shape)
    print(dice)
    print(dice_all)

    

class DiceCoeff(Function):
    """Dice coeff for individual examples"""

    def forward(self, input, target):
        # target = _make_one_hot(target, 2)
        self.save_for_backward(input, target)
        eps = 0.0001
        # dot是返回两个矩阵的点集
        # inter,uniun:两个值的大小分别是10506.6,164867.2
        self.inter = torch.dot(input.view(-1), target.view(-1))
        self.union = torch.sum(input) + torch.sum(target) + eps
        # print("inter,uniun:",self.inter,self.union)

        t = (2 * self.inter.float() + eps) / self.union.float()
        return t

    # This function has only a single output, so it gets only one gradient
    def backward(self, grad_output):

        input, target = self.saved_variables
        grad_input = grad_target = None

        if self.needs_input_grad[0]:
            grad_input = grad_output * 2 * (target * self.union - self.inter) \
                         / (self.union * self.union)
        if self.needs_input_grad[1]:
            grad_target = None

        # 这里没有打印出来，难道没有执行到这里吗
        # print("grad_input, grad_target:",grad_input, grad_target)

        return grad_input, grad_target


def dice_coeff(input, target):
    """Dice coeff for batches"""
    if input.is_cuda:
        s = torch.FloatTensor(1).cuda().zero_()
    else:
        s = torch.FloatTensor(1).zero_()

    # print("size of input, target:", input.shape, target.shape)

    for i, c in enumerate(zip(input, target)):
        # c[0],c[1]的大小都是原图大小torch.Size([1, 576, 544])
        # print("size of c0 c1:", c[0].shape,c[1].shape)
        s = s + DiceCoeff().forward(c[0], c[1])

    return s / (i + 1)


def dice_coeff_loss(input, target):
    return 1 - dice_coeff(input, target)
