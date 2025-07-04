import datetime
import time
import numpy as np
import torch
import torchvision
from torch import nn
import utils.misc as utils
from Method.utils.loss import inner_contrastive_loss, cross_entropy_loss, uncertain_consistency_loss, refine_label, \
    deep_supervision_loss, compute_stable_region_cosine, graph_consistency_loss, gaze_graph_supervision_loss, \
    intergrity_loss, dice_loss
from Method.utils.strong_aug import Jigsaw, StrongAugmentations
from medpy.metric.binary import dc


def train_one_epoch(model, criterion, train_loader, optimizer, device, epoch, args, writer, t1, t2, LOSS_WEIGHTS):
    start_time = time.time()
    model.train()
    print('-' * 40)
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 50
    total_steps = len(train_loader)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    img_list, img_aug_list, label_list, gaze_list, fixation_list, gaze_binary_list = [], [], [], [], [], []
    output_list, output_aug_list, output_aug_restore_list = [], [], []
    pred_connected_list = []
    output_d3_list, gaze_d3_list, PL_d3_list, feature_d3_list = [], [], [], []
    dice_score_list = []

    GAZE_THRESHOLD_LOW = t1
    GAZE_THRESHOLD_HIGH = t2
    STABILITY_THRESHOLD = 0.981 # 981
    BLOCK_WEIGHTS = [0.1, 0.2, 0.3, 0.4]
    block_num = 7
    jigsaw_transform = Jigsaw()

    # Training
    for step, batch in enumerate(train_loader):
        start = time.time()

        # Load data
        img, label, gaze, fixation = batch['image'], batch['label'], batch['pseudo_label'], batch['fixation']
        img, label, gaze, fixation = img.to(device), label.to(device), gaze.to(device), fixation.to(device)
        data_loading_time = time.time() - start

        # Apply aug
        img_aug, shuffle_index = jigsaw_transform(img, block_num, block_num)

        # Forward
        features, outputs = model(img)
        features_aug, outputs_aug = model(img_aug)

        # Binarize and resize gaze
        gaze_binary = torch.where(gaze < GAZE_THRESHOLD_LOW, -1, gaze)
        gaze_binary = torch.where(gaze > GAZE_THRESHOLD_HIGH, 1, gaze_binary)
        gaze_binary = torch.where((gaze >= GAZE_THRESHOLD_LOW) & (gaze <= GAZE_THRESHOLD_HIGH), 0, gaze_binary)
        gazes_binary = []
        for i in range(len(features)):
            gazes_binary.append(torch.nn.functional.interpolate(gaze_binary, size=(features[i].shape[2], features[i].shape[3]), mode='nearest'))
        fixations = []
        for i in range(len(features)):
            size = 224 // features[i].shape[2]
            fixations.append(nn.MaxPool2d(size, size, padding=0)(fixation))

        loss_integrity, pred_keep_largest_connected = intergrity_loss(outputs[-1])
        output_aug_restore, _ = jigsaw_transform(outputs_aug[-1], block_num, block_num, shuffle_index)
        ce_loss = cross_entropy_loss(outputs[-1], output_aug_restore, gaze, GAZE_THRESHOLD_LOW, GAZE_THRESHOLD_HIGH) + 0.1 * loss_integrity

        # Consistency
        consistency_loss = torch.zeros(1).to(device)
        for i in range(len(features)):
            feature, feature_aug = features[i], features_aug[i]
            feature_aug_restore, _ = jigsaw_transform(feature_aug, block_num, block_num, shuffle_index)
            consistency_loss += BLOCK_WEIGHTS[i] * graph_consistency_loss(feature, feature_aug_restore)
        consistency_loss /= len(features)

        # Pseudo label
        feature_aug_restore, _ = jigsaw_transform(features_aug[-1], block_num, block_num, shuffle_index)
        PL = refine_label(features[-1], feature_aug_restore, gazes_binary[-1], STABILITY_THRESHOLD)
        PL_loss = torch.zeros(1).to(device)
        for i in range(len(outputs) - 1):
            output = outputs[i]
            PL_resize = torch.nn.functional.interpolate(PL.float(), size=(output.shape[2], output.shape[3]), mode='nearest')
            PL_loss += BLOCK_WEIGHTS[i] * deep_supervision_loss(output, PL_resize)
        PL_loss /= (len(outputs) - 1)

        # Graph
        gaze_graph_loss = torch.zeros(1).to(device)
        for i in range(len(features)):
            feature, fixation = features[i], fixations[i]
            gaze_graph_loss += BLOCK_WEIGHTS[i] * gaze_graph_supervision_loss(feature, fixation)
        gaze_graph_loss /= len(features)

        # Total loss
        loss = (ce_loss +
            LOSS_WEIGHTS['consistency'] * consistency_loss +
            LOSS_WEIGHTS['PL'] * PL_loss +
            LOSS_WEIGHTS['gaze_graph'] * gaze_graph_loss
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 30 == 0:
            img_list.append(img[0].detach())
            img_aug_list.append(img_aug[0].detach())
            label_list.append(label[0].detach())
            gaze_list.append(gaze[0].detach())
            fixation_list.append(fixation[0].detach())
            gaze_binary_list.append(gaze_binary[0].detach())
            output_list.append(torch.where(nn.Sigmoid()(outputs[-1][0]) > 0.5, 1, 0).detach())
            output_aug_list.append(torch.where(nn.Sigmoid()(outputs_aug[-1][0]) > 0.5, 1, 0).detach())
            output_aug_restore_list.append(torch.where(nn.Sigmoid()(output_aug_restore[0]) > 0.5, 1, 0).detach())
            pred_connected_list.append(pred_keep_largest_connected[0].detach())
            output_d3_list.append(torch.where(nn.Sigmoid()(outputs[-2][0]) > 0.5, 1, 0).detach())
            gaze_d3_list.append(gazes_binary[-1][0].detach())
            PL_d3_list.append(PL[0].detach())
            feature_d3_list.append(torch.mean(features[-1][0], 0, True).detach())

        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(loss=loss)
        metric_logger.update(ce_loss=ce_loss)
        metric_logger.update(cons_loss=consistency_loss)
        metric_logger.update(PL_loss=PL_loss)
        metric_logger.update(Gaze_Graph_loss=gaze_graph_loss)
        metric_logger.update(integrity_loss=loss_integrity)

        itertime = time.time() - start
        metric_logger.log_every(step, total_steps, data_loading_time, itertime, print_freq, header)

        output = torch.where(nn.Sigmoid()(outputs[-1]) > 0.5, 1, 0)
        dice_score_list.append(dc(label, output))

    # gather the stats from all processes
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('{} Total time: {} ({:.4f} s / it)'.format(header, total_time_str, total_time / total_steps))
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    dice_score = np.array(dice_score_list).mean()

    writer.add_scalar('loss', loss.item(), epoch)
    writer.add_scalar('ce_loss', ce_loss.item(), epoch)
    writer.add_scalar('cons_loss', consistency_loss.item(), epoch)
    writer.add_scalar('PL_loss', PL_loss.item(), epoch)
    writer.add_scalar('Gaze_Graph_loss', gaze_graph_loss.item(), epoch)
    writer.add_scalar('integrity_loss', loss_integrity.item(), epoch)
    writer.add_scalar('Train Dice Score', dice_score, epoch)

    def save_image(image, tag, epoch, writer):
        image = (image - image.min()) / (image.max() - image.min() + 1e-6)
        grid = torchvision.utils.make_grid(torch.tensor(image), nrow=4, pad_value=1)
        writer.add_image(tag, grid, epoch)

    save_image(torch.stack(img_list).float(), 'img', epoch, writer)
    save_image(torch.stack(img_aug_list).float(), 'img_aug', epoch, writer)
    save_image(torch.stack(label_list).float(), 'label', epoch, writer)
    save_image(torch.stack(gaze_list).float(), 'gaze', epoch, writer)
    save_image(torch.stack(fixation_list).float(), 'fixation', epoch, writer)
    save_image(torch.stack(gaze_binary_list).float(), 'gaze_binary', epoch, writer)
    save_image(torch.stack(output_list).float(), 'output', epoch, writer)
    save_image(torch.stack(output_aug_list).float(), 'output_aug', epoch, writer)
    save_image(torch.stack(output_aug_restore_list).float(), 'output_aug_restore', epoch, writer)
    save_image(torch.stack(pred_connected_list).float(), 'output_connected', epoch, writer)
    save_image(torch.stack(output_d3_list).float(), 'output_d3', epoch, writer)
    save_image(torch.stack(gaze_d3_list).float(), 'gaze_d3', epoch, writer)
    save_image(torch.stack(PL_d3_list).float(), 'PL_d3', epoch, writer)
    save_image(torch.stack(feature_d3_list).float(), 'feature_d3', epoch, writer)