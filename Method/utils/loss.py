from skimage import measure
import numpy as np
import torch
import torch.nn.functional as Func
from torch import nn


def dice_loss(output1, output2, output_PL):
    output_PL = torch.where(output_PL == -1, 0, output_PL).float()
    smooth = 1e-5
    intersect = torch.sum(output1 * output_PL)
    y_sum = torch.sum(output_PL * output_PL)
    z_sum = torch.sum(output1 * output1)
    loss_1 = 1 - (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    intersect = torch.sum(output2 * output_PL)
    z_sum = torch.sum(output2 * output2)
    loss_2 = 1 - (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    return loss_1 + loss_2

def intergrity_loss(pred):
    predictions_original_list = []
    pred = nn.Sigmoid()(pred)
    pred = torch.concat([1-pred, pred], dim=1)
    for i in range(pred.shape[0]):
        prediction = np.uint8(np.argmax(pred[i, :, :, :].detach().cpu(), axis=0))
        prediction = keep_largest_connected_components(prediction)
        prediction = torch.from_numpy(prediction).to(pred.device)
        predictions_original_list.append(prediction)
    predictions = torch.stack(predictions_original_list)
    pred_keep_largest_connected = torch.unsqueeze(predictions, 1)

    loss_integrity = 1 - Func.cosine_similarity(pred[:, 1, :, :], pred_keep_largest_connected, dim=1).mean()
    return loss_integrity, pred_keep_largest_connected


def gaze_graph_supervision_loss(feature, fixation):
    # ----Construct graph for feature
    f_node = feature.reshape(feature.shape[0], feature.shape[1], -1, 1).contiguous()
    f_node = Func.normalize(f_node, p=2.0, dim=1).transpose(2, 1).squeeze(-1)
    distance = pairwise_distance(f_node.detach())

    # ----Construct graph for fixation
    N = fixation.shape[0]
    fixation_remapped = torch.full_like(fixation, -1)
    for i in range(N):
        mat = fixation[i, 0]
        unique = torch.unique(mat)
        valid = unique[unique != -1]
        sorted_valid, _ = torch.sort(valid)
        mapping = {v.item(): j for j, v in enumerate(sorted_valid)}
        mask = (mat != -1)
        remapped_vals = mat.clone()
        for old, new in mapping.items():
            remapped_vals[mat == old] = new
        fixation_remapped[i, 0] = torch.where(mask, remapped_vals, -1)
    fixation_node = fixation_remapped.reshape(fixation.shape[0], fixation.shape[1], -1, 1).contiguous().transpose(2, 1).squeeze(-1)
    distance_fixation = pairwise_distance(fixation_node.detach())

    # Compute loss
    gt = (distance_fixation != 1).float()
    distance_norm = nn.Sigmoid()(distance)
    loss_map = - (1 - gt) * torch.log(1 - distance_norm)
    valid_count = (1 - gt).sum()
    if valid_count == 0:
        return torch.tensor(0.0, device=distance.device, requires_grad=True)
    loss = (loss_map).sum() / valid_count
    return loss

def graph_consistency_loss(feature, feature_aug):
    # Node consistency loss
    node_consistency_loss = 1 - Func.cosine_similarity(feature, feature_aug, dim=1).mean()

    # Edge consistency loss
    # ----Construct graph for feature
    f_node = feature.reshape(feature.shape[0], feature.shape[1], -1, 1).contiguous()
    f_node = Func.normalize(f_node, p=2.0, dim=1).transpose(2, 1).squeeze(-1)
    distance = pairwise_distance(f_node.detach())
    # ----Construct graph for feature_aug
    f_node_aug = feature_aug.reshape(feature_aug.shape[0], feature_aug.shape[1], -1, 1).contiguous()
    f_node_aug = Func.normalize(f_node_aug, p=2.0, dim=1).transpose(2, 1).squeeze(-1)
    distance_aug = pairwise_distance(f_node_aug.detach())
    # ----MSE Loss
    edge_consistency_loss = Func.l1_loss(distance, distance_aug)

    # Graph consistency loss
    graph_loss = node_consistency_loss + edge_consistency_loss
    return graph_loss


def compute_stable_region_cosine(f1, f2, gaze):
    cons = Func.cosine_similarity(f1, f2, dim=1)
    certain_mask = torch.where(gaze != 0, 1, 0).squeeze(1)
    num = torch.sum(certain_mask, dim=(1, 2))
    cons = cons * certain_mask
    cons = torch.sum(cons, dim=(1, 2))
    return torch.sum(cons / num)


def deep_supervision_loss(output, gaze):
    certain_mask = torch.where(gaze == 0, 0, 1)
    certain_num = torch.sum(certain_mask, dim=(1, 2, 3))
    gaze_m = torch.where(gaze == -1, 0, gaze)
    output_m = output * certain_mask
    criterion = torch.nn.BCEWithLogitsLoss(reduction="none")
    ce_loss = criterion(output_m, gaze_m.float())
    ce_loss = (torch.sum(ce_loss, dim=(1, 2, 3)) + 1e-5) / (certain_num + 1e-5)
    return ce_loss.mean()


def cross_entropy_loss(output1, output2, gaze, t1, t2):
    gaze_binary = torch.where(gaze < t1, 0, gaze)
    gaze_binary = torch.where(gaze_binary > t2, 1, gaze_binary)
    mask = torch.where((gaze_binary >= t1) & (gaze_binary <= t2), 0, 1)
    gaze_binary_m, output1_m, output2_m = gaze_binary * mask, output1 * mask, output2 * mask
    certain_num = torch.sum(mask, dim=(1, 2, 3))

    criterion = torch.nn.BCEWithLogitsLoss(reduction="none")
    ce_loss_1 = criterion(output1_m, gaze_binary_m.float())
    ce_loss_1 = (torch.sum(ce_loss_1, dim=(1, 2, 3)) + 1e-5) / (certain_num + 1e-5)
    ce_loss_2 = criterion(output2_m, gaze_binary_m.float())
    ce_loss_2 = (torch.sum(ce_loss_2, dim=(1, 2, 3)) + 1e-5) / (certain_num + 1e-5)
    ce_loss = ce_loss_1.mean() + ce_loss_2.mean()
    return ce_loss


def uncertain_consistency_loss(feature1, feature2, gaze_binary):
    uncertain_mask = torch.where(gaze_binary == 0, 1, 0)
    uncertain_num = torch.sum(uncertain_mask, dim=(1, 2, 3))
    feature1 = feature1 * uncertain_mask
    feature2 = feature2 * uncertain_mask
    cons = Func.cosine_similarity(feature1, feature2, dim=1)
    cons = (torch.sum(cons, dim=(1, 2)) + 1e-5) / (uncertain_num + 1e-5)
    cons_loss = 1 - cons.mean()
    return cons_loss


def batched_index_select(x, idx):
    batch_size, num_dims, num_vertices_reduced = x.shape[:3]
    _, num_vertices, k = idx.shape
    idx_base = torch.arange(0, batch_size, device=idx.device).view(-1, 1, 1) * num_vertices_reduced
    idx = idx + idx_base
    idx = idx.contiguous().view(-1)
    x = x.transpose(2, 1)
    feature = x.contiguous().view(batch_size * num_vertices_reduced, -1)[idx, :]
    feature = feature.view(batch_size, num_vertices, k, num_dims).permute(0, 3, 1, 2).contiguous()
    return feature


def pairwise_distance(x):
    with torch.no_grad():
        x_inner = -2 * torch.matmul(x, x.transpose(2, 1))
        x_square = torch.sum(torch.mul(x, x), dim=-1, keepdim=True)
        return x_square + x_inner + x_square.transpose(2, 1)


def refine_label(f1_b, f2_b, gaze_binary, stable_threshold):
    f1 = f1_b.reshape(f1_b.shape[0], f1_b.shape[1], -1, 1).contiguous()
    f2 = f2_b.reshape(f2_b.shape[0], f2_b.shape[1], -1, 1).contiguous()
    gaze = gaze_binary.reshape(gaze_binary.shape[0], gaze_binary.shape[1], -1).contiguous()
    # ------------------get stable area-----------------------
    stable_cos_score = Func.cosine_similarity(f1, f2, dim=1).squeeze(-1)
    stable_mask = torch.where(stable_cos_score > stable_threshold, 1, 0)
    # ------------------construct graph-----------------------
    feature = Func.normalize(f1, p=2.0, dim=1)
    feature = feature.transpose(2, 1).squeeze(-1)
    batch_size, n_points, n_dims = feature.shape
    dist = pairwise_distance(feature.detach())
    _, nn_idx = torch.topk(-dist, k=24)
    center_idx = torch.arange(0, n_points, device=feature.device).repeat(batch_size, 24, 1).transpose(2, 1)
    edge_index = torch.stack((nn_idx, center_idx), dim=0)
    # ------------------get neighbor nodes fj-----------------------
    f1_i = batched_index_select(f1, edge_index[1])
    f1_j = batched_index_select(f1, edge_index[0])
    # ------------------compute cosine-----------------------
    f_i_j_cos = Func.cosine_similarity(f1_i, f1_j, dim=1)
    # ------------------estimate label use cos-----------------------
    gaze_label = batched_index_select(gaze, edge_index[0])[:,0,:]
    pseudo_label = torch.sum(gaze_label * f_i_j_cos, 2)
    pseudo_label = torch.where(pseudo_label > 0, 1, -1)
    # ------------------update stable label-----------------------
    update_mask = torch.where((gaze[:,0,:] == 0) & (stable_mask == 1), 1, 0)
    pseudo_label = pseudo_label * update_mask
    pseudo_label = pseudo_label + gaze[:, 0, :]
    pseudo_label = pseudo_label.reshape(f1_b.shape[0], f1_b.shape[2], f1_b.shape[3]).contiguous().unsqueeze(1)
    return pseudo_label


def pairwise_cos_distance(x):
    x = x.reshape(x.shape[0], x.shape[1], -1, 1).contiguous()
    x = x.transpose(2, 1).squeeze(-1)
    # cosine similarity
    x_t = x.transpose(2, 1)
    x_matmul =  torch.matmul(x, x_t)
    norm_a = torch.sqrt(torch.sum(x * x + 1e-5, dim=2, keepdim=True))
    norm_b = torch.sqrt(torch.sum(x_t * x_t + 1e-5, dim=1, keepdim=True))
    cos_sim = (x_matmul) / (norm_a * norm_b + 1e-5)
    return cos_sim


def pairwise_pos_mask(x):
    x = x.reshape(x.shape[0], 1, -1, 1).contiguous()
    x = x.transpose(2, 1).squeeze(-1)
    x, x_t = x.repeat(1, 1, x.size(1)), x.transpose(2, 1).repeat(1, x.size(1), 1)
    x_add = x + x_t
    pos_mask = torch.where((x_add == 2) | (x_add == -2), 1, 0)
    uncertain_mask = torch.where((x_add == 1) | (x_add == -1), 0, 1)
    return pos_mask, uncertain_mask

def inner_contrastive_loss(feature, gaze, temperature):
    # ------------------uncertain mask-----------------------
    mask = torch.where(gaze == 0, 0, 1)
    feature = feature * mask
    gaze = gaze * mask
    # ------------------cosine distance of each certain feature-----------------------
    feature_distance = pairwise_cos_distance(feature)
    pos_mask, uncertain_mask = pairwise_pos_mask(gaze)
    # ------------------inforNCE loss-----------------------
    feature_distance = torch.exp(feature_distance / temperature) * uncertain_mask
    pos_feature_distance = feature_distance * pos_mask
    softmax = torch.sum(pos_feature_distance, dim=1) / (torch.sum(feature_distance, dim=1) + 1e-5)
    loss = -torch.log(torch.where(softmax <= 1e-7, torch.tensor(1.0), softmax))
    loss = torch.mean(loss, dim=1)
    loss = torch.mean(loss)
    return loss


def keep_largest_connected_components(mask):
    '''
    Keeps only the largest connected components of each label for a segmentation mask.
    '''
    # keep a heart connectivity
    mask_shape = mask.shape

    heart_slice = np.where((mask > 0), 1, 0)
    out_heart = np.zeros(heart_slice.shape, dtype=np.uint8)
    for struc_id in [1]:
        binary_img = heart_slice == struc_id
        blobs = measure.label(binary_img, connectivity=1)
        props = measure.regionprops(blobs)
        if not props:
            continue
        area = [ele.area for ele in props]
        largest_blob_ind = np.argmax(area)
        largest_blob_label = props[largest_blob_ind].label
        out_heart[blobs == largest_blob_label] = struc_id

    # keep LV/RV/MYO connectivity
    out_img = np.zeros(mask.shape, dtype=np.uint8)
    for struc_id in [1, 2, 3]:
        binary_img = mask == struc_id
        blobs = measure.label(binary_img, connectivity=1)
        props = measure.regionprops(blobs)
        if not props:
            continue
        area = [ele.area for ele in props]
        largest_blob_ind = np.argmax(area)
        largest_blob_label = props[largest_blob_ind].label
        out_img[blobs == largest_blob_label] = struc_id

    final_img = out_heart * out_img
    return final_img


