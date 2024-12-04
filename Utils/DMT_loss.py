"""
Dual-Teacher
Copyright (c) 2023-present NAVER Cloud Corp.
distributed under NVIDIA Source Code License for SegFormer
--------------------------------------------------------
References:
SegFormer: https://github.com/NVlabs/SegFormer
--------------------------------------------------------
"""
import torch
import random
from PIL import ImageFilter
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
import torchvision.transforms.functional as transforms_f


def compute_cutmix(h, w, imgs, labels, criterion, model, ema_model, image_u, threshold):
    
    with torch.no_grad():
        pred = ema_model(image_u)
        pred = F.interpolate(pred, (h, w), mode="bilinear", align_corners=False)
        pred = torch.sigmoid(pred)  # Apply sigmoid for binary segmentation
        pred_label = (pred > threshold).float()  # Binarize predictions based on the threshold

    # Generate augmented images and labels
    image_aug, label_aug = cut_mixer(image_u, pred_label.clone())
    image_aug, label_aug, pred = batch_transform(
        image_aug, label_aug, pred,
        crop_size=(pred.shape[2], pred.shape[3]), scale_size=(1.0, 1.0), apply_augmentation=True
    )

    num_labeled = len(imgs)
    outputs = model(torch.cat([imgs, image_aug]))
    outputs, outputs_u = outputs[:num_labeled], outputs[num_labeled:]
# Đảm bảo labels có chiều kênh phù hợp
    if labels.dim() == 3:  # labels có dạng (N, H, W)
        labels = labels.unsqueeze(1)  # Thêm chiều channel -> (N, 1, H, W)

    # Interpolate outputs để khớp kích thước với labels
    outputs_labeled = torch.sigmoid(
        F.interpolate(outputs, size=labels.shape[2:], mode='bilinear', align_corners=False)
    )

    bce_loss = criterion[0](outputs_labeled, labels)
    dice_loss_value = criterion[1](outputs_labeled, labels)
    sup_loss = bce_loss + dice_loss_value

    # Compute unsupervised loss
    pred_u = torch.sigmoid(F.interpolate(outputs_u, (h, w), mode="bilinear", align_corners=False))
    cutmix_loss = compute_unsupervised_loss(pred_u, label_aug.clone(), pred, threshold)
    
    return sup_loss + cutmix_loss


def tensor_to_pil(im, label, logits):
    # Denormalize the image tensor
    im = denormalise(im)
    im = transforms_f.to_pil_image(im.cpu())  # Convert to PIL image

    # Process the label tensor
    if label.ndimension() == 4:  # If batch, take the first sample
        label = label[0]
    label = label.float() / 255.  # Normalize
    label = transforms_f.to_pil_image(label.cpu())  # Convert to PIL image

    # Process the logits tensor
    if logits.ndimension() == 4:  # If batch, take the first sample
        logits = logits[0]
    logits = transforms_f.to_pil_image(logits.cpu())  # Convert to PIL image

    return im, label, logits



def denormalise(x, imagenet=True):
    if imagenet:
        x = transforms_f.normalize(x, mean=[0., 0., 0.], std=[1 / 0.229, 1 / 0.224, 1 / 0.225])
        x = transforms_f.normalize(x, mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.])
        return x
    else:
        return (x + 1) / 2


def transform(image, label, logits=None, crop_size=(32, 32), scale_size=(0.8, 1.0), augmentation=True):
    # Random rescale image
    raw_w, raw_h = image.size
    scale_ratio = random.uniform(scale_size[0], scale_size[1])

    resized_size = (int(raw_h * scale_ratio), int(raw_w * scale_ratio))
    image = transforms_f.resize(image, resized_size, interpolation=transforms_f.InterpolationMode("bilinear"))  # Image.BILINEAR
    label = transforms_f.resize(label, resized_size, interpolation=transforms_f.InterpolationMode("nearest"))  # Image.NEAREST
    if logits is not None:
        logits = transforms_f.resize(logits, resized_size, interpolation=transforms_f.InterpolationMode("nearest"))  # Image.NEAREST

    # Add padding if rescaled image size is less than crop size
    if crop_size == -1:  # use original im size without crop or padding
        crop_size = (raw_h, raw_w)

    if crop_size[0] > resized_size[0] or crop_size[1] > resized_size[1]:
        right_pad, bottom_pad = max(crop_size[1] - resized_size[1], 0), max(crop_size[0] - resized_size[0], 0)
        image = transforms_f.pad(image, padding=(0, 0, right_pad, bottom_pad), padding_mode='reflect')
        label = transforms_f.pad(label, padding=(0, 0, right_pad, bottom_pad), fill=255, padding_mode='constant')
        if logits is not None:
            logits = transforms_f.pad(logits, padding=(0, 0, right_pad, bottom_pad), fill=0, padding_mode='constant')

    # Random Cropping
    i, j, h, w = transforms.RandomCrop.get_params(image, output_size=crop_size)
    image = transforms_f.crop(image, i, j, h, w)
    label = transforms_f.crop(label, i, j, h, w)
    if logits is not None:
        logits = transforms_f.crop(logits, i, j, h, w)

    if augmentation:
        # Random color jitter
        if torch.rand(1) > 0.2:
            color_transform = transforms.ColorJitter((0.75, 1.25), (0.75, 1.25), (0.75, 1.25), (-0.25, 0.25))
            image = color_transform(image)

        # Random Gaussian filter
        if torch.rand(1) > 0.5:
            sigma = random.uniform(0.15, 1.15)
            image = image.filter(ImageFilter.GaussianBlur(radius=sigma))

        # Random horizontal flipping
        if torch.rand(1) > 0.5:
            image = transforms_f.hflip(image)
            label = transforms_f.hflip(label)
            if logits is not None:
                logits = transforms_f.hflip(logits)

    # Transform to tensor
    image = transforms_f.to_tensor(image)
    label = (transforms_f.to_tensor(label) * 255).long()
    label[label == 255] = -1  # invalid pixels are re-mapped to index -1
    if logits is not None:
        logits = transforms_f.to_tensor(logits)

    # Apply (ImageNet) normalisation
    image = transforms_f.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if logits is not None:
        return image, label, logits
    else:
        return image, label


def batch_transform(data, label, logits, crop_size, scale_size, apply_augmentation):
    data_list, label_list, logits_list = [], [], []
    device = data.device

    for k in range(data.shape[0]):
        data_pil, label_pil, logits_pil = tensor_to_pil(data[k], label[k], logits[k])
        aug_data, aug_label, aug_logits = transform(data_pil, label_pil, logits_pil,
                                                    crop_size=crop_size,
                                                    scale_size=scale_size,
                                                    augmentation=apply_augmentation)
        data_list.append(aug_data.unsqueeze(0))
        label_list.append(aug_label)
        logits_list.append(aug_logits)

    data_trans, label_trans, logits_trans = \
        torch.cat(data_list).cuda(), torch.cat(label_list).cuda(), torch.cat(logits_list).to(device)
    return data_trans, label_trans, logits_trans


def compute_unsupervised_loss(predict, target, logits, threshold):
    # Compute valid mask
    valid_mask = (target >= 0).float()  # Ensure only valid pixels are considered
    threshold_mask = (logits > threshold).float()  # Mask for confident predictions

    # Convert target to float if not already in that type
    target = target.float()

    # Add extra channel dimension to target to match the input size
    target = target.unsqueeze(1)  # Add channel dimension if missing

    # Compute BCE loss for unsupervised targets
    bce_loss = F.binary_cross_entropy_with_logits(
        predict, target, reduction='none'
    )  # Raw BCE loss without reduction
    bce_loss = bce_loss * valid_mask * threshold_mask  # Mask invalid and low-confidence pixels

    # Compute final loss
    num_valid_pixels = valid_mask.sum()
    weighted_loss = bce_loss.sum() / (num_valid_pixels + 1e-8)  # Normalize by valid pixel count

    return weighted_loss



def rand_bbox_1(size, lam=None):
    # past implementation
    W = size[2]
    H = size[3]
    B = size[0]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(size=[B, ], low=int(W / 8), high=W)
    cy = np.random.randint(size=[B, ], low=int(H / 8), high=H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)

    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def cut_mixer(data, target):
    target = target.unsqueeze(dim=1)
    mix_data = data.clone()
    mix_target = target.clone()
    u_rand_index = torch.randperm(data.size()[0])[:data.size()[0]].cuda()
    u_bbx1, u_bby1, u_bbx2, u_bby2 = rand_bbox_1(data.size(), lam=np.random.beta(4, 4))

    for i in range(0, mix_data.shape[0]):
        mix_data[i, :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = \
            data[u_rand_index[i], :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]

        mix_target[i, :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = \
            target[u_rand_index[i], :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]

    del data, target
    torch.cuda.empty_cache()
    return mix_data, mix_target.squeeze(dim=1)

def get_bin_mask(b, argmax_occluder):
    for image_i in range(b):
        unique_vals = torch.unique(argmax_occluder[image_i])

        # Bỏ qua giá trị không hợp lệ (nếu có)
        unique_vals = unique_vals[unique_vals != 255]
        if len(unique_vals) == 1:  # Chỉ có một lớp foreground/background
            classes = unique_vals
        else:
            nclasses = len(unique_vals)
            classes = (unique_vals[torch.randperm(nclasses)[:nclasses // 2]]).cuda()

        if image_i == 0:
            binary_mask = generate_class_mask(argmax_occluder[image_i], classes).unsqueeze(0).cuda()
        else:
            binary_mask = torch.cat(
                (binary_mask, generate_class_mask(argmax_occluder[image_i], classes).unsqueeze(0).cuda())
            )
    return binary_mask


def compute_classmix(b, h, w, criterion, cm_loss_fn, model, ema_model, imgs, labels, unsup_imgs, image_u_strong, threshold):
    # Unlabeled Process
    with torch.no_grad():
        logits_occluder = ema_model(unsup_imgs)
        logits_occluder = F.interpolate(logits_occluder, (h, w), mode="bilinear", align_corners=False)
        sigmoid_occluder = torch.sigmoid(logits_occluder)  # Binary segmentation prediction
        argmax_occluder = (sigmoid_occluder > threshold).float()  # Thresholding to binary

    binary_mask = get_bin_mask(b, argmax_occluder)
    binary_mask = binary_mask.squeeze(dim=1)

    if b == 2:
        shuffle_index = torch.tensor([1, 0])
    else:
        shuffle_index = torch.randperm(b).cuda()

    # Create Class-Mixed Images
    class_mixed_img = class_mix(occluder_mask=binary_mask, occluder=image_u_strong, occludee=image_u_strong[shuffle_index])

    num_labeled = len(imgs)
    outputs = model(torch.cat([imgs, class_mixed_img]))
    outputs, outputs_u = outputs[:num_labeled], outputs[num_labeled:]

    # Compute Supervised Loss
    pred_large = torch.sigmoid(
        F.interpolate(outputs, size=labels.shape[2:], mode='bilinear', align_corners=False)
    )


    bce_loss = criterion[0](pred_large, labels)
    dice_loss_value = criterion[1](pred_large, labels)
    sup_loss = bce_loss + dice_loss_value

    del outputs, pred_large
    torch.cuda.empty_cache()

    # Compute Unsupervised Loss
    logits_class_mixed = F.interpolate(outputs_u, (h, w), mode="bilinear", align_corners=False)
    sigmoid_class_mixed = torch.sigmoid(logits_class_mixed)

    # Create Pseudo Labels
    class_mixed_mask = class_mix(occluder_mask=binary_mask, occluder=sigmoid_occluder, occludee=sigmoid_occluder[shuffle_index])
    pseudo_label = (class_mixed_mask > threshold).float()

    # Calculate Weight for Unlabeled Loss
    unlabeled_weight = torch.sum(class_mixed_mask.ge(threshold).float()) / class_mixed_mask.numel()
    pixel_weight = unlabeled_weight * torch.ones_like(class_mixed_mask).cuda()

    # Compute Class-Mix Loss
    class_mix_loss = cm_loss_fn(sigmoid_class_mixed, pseudo_label, pixel_weight)
    loss = sup_loss + class_mix_loss
    return loss


def class_mix(occluder_mask, occluder, occludee):
    if occluder.dim() == 4 and occluder.shape[1] == 1:  # Single-channel binary mask
        occluder_mask = occluder_mask.unsqueeze(dim=1)  # Expand to match dimensions

    # Blend images based on the occluder mask
    mixed_data = occluder_mask.float() * occluder + (1 - occluder_mask.float()) * occludee
    del occluder_mask, occluder, occludee
    torch.cuda.empty_cache()
    return mixed_data


def generate_class_mask(pred, classes):
    # Generate binary mask for specific classes
    binary_mask = torch.zeros_like(pred, dtype=torch.bool)  # Initialize as boolean
    for c in classes:
        binary_mask = binary_mask | (pred == c)
    return binary_mask.float()  # Convert to float if needed for downstream operations



def blur(img, p=0.5):
    if random.random() < p:
        sigma = np.random.uniform(0.1, 2.0)
        img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
    return img


def compute_ic(model, ema_model, image_u, image_u_strong, criterion, label_u, h, w, threshold):
    with torch.no_grad():
        # Predict using EMA model on unlabeled image
        logits = ema_model(image_u)  # Logits output from EMA model
        logits = F.interpolate(logits, (h, w), mode="bilinear", align_corners=False)  # Resize to standard size
        probs = torch.sigmoid(logits)  # Compute probabilities for each pixel (for binary segmentation)
        max_probs = probs.squeeze(1)  # Remove channel dimension [B, H, W]
        argmax_label = (max_probs >= threshold).float()  # Binarize based on threshold

    # Predict using current model on strongly augmented image
    pred_dc = model(image_u_strong) 
    pred_dc = F.interpolate(pred_dc, (h, w), mode="bilinear", align_corners=False)
    
    # Modify to use BCE Loss
    loss_dc = F.binary_cross_entropy_with_logits(
        pred_dc.squeeze(1), 
        argmax_label, 
        reduction='none'
    )
    
    # Create mask for valid pixels (above threshold and not ignore label)
    mask = ((max_probs >= threshold) & (label_u != 255))
    loss_dc = loss_dc * mask.float()  
    loss_dc = loss_dc.sum() / (mask.sum().item() + 1e-8)  # Add small epsilon to prevent division by zero
    
    return loss_dc.clone()


class ClassMixLoss(nn.Module):
    def __init__(self, weight=None, reduction='none'):
        super(ClassMixLoss, self).__init__()
        self.BCE = nn.BCEWithLogitsLoss(weight=weight, reduction=reduction)

    def forward(self, output, target, pixel_weight):
        if target.dim() == 3: 
            target = target.unsqueeze(1)  
        loss_per_pixel = self.BCE(output, target) 

        weighted_loss = loss_per_pixel.squeeze(1) * pixel_weight  # [B, H, W]
        loss = weighted_loss.mean()  # Tính giá trị trung bình

        return loss