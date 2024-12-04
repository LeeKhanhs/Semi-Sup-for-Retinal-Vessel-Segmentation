'''
The default exp_name is tmp. Change it before formal training! isic2018 PH2 DMF SKD
nohup python -u multi_train_adapt.py --exp_name test --config_yml Configs/multi_train_local.yml --model MedFormer --batch_size 16 --adapt_method False --num_domains 1 --dataset PH2  --k_fold 4 > 4MedFormer_PH2.out 2>&1 &
'''
import argparse
from sqlite3 import adapt
import yaml
import os, time
from datetime import datetime
import cv2

import pandas as pd
import torch.nn as nn
import torch.utils.data
import torch.optim as optim
import medpy.metric.binary as metrics
import torch.nn.functional as F

from Datasets.create_dataset import *
from Datasets.transform import normalize
from Utils.losses import dice_loss
# from Utils.losses import compute_contra_memobank_loss
from Utils.pieces import DotDict
from Utils.functions import fix_all_seed
from Utils.metrics import calc_metrics
from itertools import cycle
from Models.unet_u2pl import UNet_U2PL
from Models.unet import UNet

torch.cuda.empty_cache()

def main(config):
    
    dataset = get_dataset(config, img_size=config.data.img_size, 
                                                    supervised_ratio=config.data.supervised_ratio, 
                                                    train_aug=config.data.train_aug,
                                                    k=config.fold,
                                                    ulb_dataset=StrongWeakAugment2,
                                                    lb_dataset=SkinDataset2)

    l_train_loader = torch.utils.data.DataLoader(dataset['lb_dataset'],
                                                batch_size=config.train.l_batchsize,
                                                shuffle=True,
                                                num_workers=config.train.num_workers,
                                                pin_memory=True,
                                                drop_last=False)
    u_train_loader = torch.utils.data.DataLoader(dataset['ulb_dataset'],
                                                batch_size=config.train.u_batchsize,
                                                shuffle=True,
                                                num_workers=config.train.num_workers,
                                                pin_memory=True,
                                                drop_last=False)
    val_loader = torch.utils.data.DataLoader(dataset['val_dataset'],
                                                batch_size=config.test.batch_size,
                                                shuffle=False,
                                                num_workers=config.test.num_workers,
                                                pin_memory=True,
                                                drop_last=False)
    test_loader = torch.utils.data.DataLoader(dataset['val_dataset'],
                                                batch_size=config.test.batch_size,
                                                shuffle=False,
                                                num_workers=config.test.num_workers,
                                                pin_memory=True,
                                                drop_last=False)
    train_loader = {'l_loader':l_train_loader, 'u_loader':u_train_loader}
    print(len(u_train_loader), len(l_train_loader))

    
    model_teacher  = UNet()
    model_student  = UNet()



    total_trainable_params = sum(
                    p.numel() for p in model_teacher.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model_teacher.parameters())
    print('{}M total parameters'.format(total_params/1e6))
    print('{}M total trainable parameters'.format(total_trainable_params/1e6))

    model_teacher = model_teacher.cuda()
    model_student = model_student.cuda()
    
    criterion = [nn.BCELoss(), dice_loss]

    model = train_val(config, model_teacher, model_student, train_loader, val_loader, criterion)
    test(config, model, best_model_dir, test_loader, criterion)
    
def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))

def update_teacher_ema(model_student, model_teacher, ema_decay=0.999):
    """Update teacher model using Exponential Moving Average (EMA)"""
    with torch.no_grad():
        for param_student, param_teacher in zip(model_student.parameters(), model_teacher.parameters()):
            param_teacher.data = ema_decay * param_teacher.data + (1.0 - ema_decay) * param_student.data
def compute_contra_memobank_loss(rep_all, rep_all_teacher, label_l, label_u, num_labeled, memobank, queue_size):
    """Simple contrastive loss computation using memory bank"""
    
    # Define loss
    contra_loss = 0
    device = rep_all.device

    # Separate the labeled and unlabeled features
    features_l = rep_all[:num_labeled]  # Labeled features
    features_u = rep_all[num_labeled:]  # Unlabeled features

    # Flatten the features to 2D for similarity calculation
    features_l = features_l.view(features_l.size(0), -1)
    features_u = features_u.view(features_u.size(0), -1)

    # Ensure memory bank is on the same device
    mem_features = torch.cat([tensor.to(device) for tensor in memobank], dim=0)
    mem_features = mem_features.view(mem_features.size(0), -1)  # Flatten memory features
    
    # Calculate cosine similarity for labeled and unlabeled features
    similarities_l = torch.cosine_similarity(features_l, mem_features, dim=1).mean()
    similarities_u = torch.cosine_similarity(features_u, mem_features, dim=1).mean()

    # Contrastive loss is the mean similarity across both labeled and unlabeled features
    contra_loss += similarities_l + similarities_u

    # Update memory bank with new features (optional, if needed)
    # Here we assume that 'dequeue_and_enqueue' will handle updating the memory bank.
    # For simplicity, you can choose to skip this if you're not using a memory bank.
    # dequeue_and_enqueue(features_l, memobank, queue_size)  # Optionally update the memory

    return contra_loss
def train_val(config, model_teacher, model_student, train_loader, val_loader, criterion):
    """Training and validation loop with memory bank and contrastive loss"""
    # Optimizers
    if config.train.optimizer.mode == 'adam':
        # optimizer = optim.Adam(model.parameters(), lr=float(config.train.optimizer.adam.lr))
        print('choose wrong optimizer')
    elif config.train.optimizer.mode == 'adamw':
        optimizer1 = optim.AdamW(filter(lambda p: p.requires_grad, model_teacher.parameters()),lr=float(config.train.optimizer.adamw.lr),
                                weight_decay=float(config.train.optimizer.adamw.weight_decay))
        optimizer2 = optim.AdamW(filter(lambda p: p.requires_grad, model_student.parameters()),lr=float(config.train.optimizer.adamw.lr),
                                weight_decay=float(config.train.optimizer.adamw.weight_decay))
    scheduler1 = optim.lr_scheduler.StepLR(optimizer1, step_size=50, gamma=0.5)
    scheduler2 = optim.lr_scheduler.StepLR(optimizer2, step_size=50, gamma=0.5)

    # Initialize memory bank
    num_classes = 2
    memobank = []
    queue_size = [50000, 30000]  # Memory size for each class
    for i in range(num_classes):
        memobank.append([torch.zeros(0, 256)])  # Initialize memory for class i with empty tensor
    best_model_dir = 'best_model.pth'

    # Training loop
    epochs = config.train.num_epochs
    max_dice = 0
    best_epoch = 0
    max_se = 0 # use for record best model
    max_acc = 0 # use for record best model
    best_epoch = 0 # use for recording the best epoch
    for epoch in range(epochs):
        start = time.time()
        # ----------------------------------------------------------------------
        # train
        # ---------------------------------------------------------------------
        model_student.train()
        total_loss = 0
        total_sup_loss = 0
        total_unsup_loss = 0
        total_contra_loss = 0
        dice_train_sum = 0
        iou_train_sum = 0
        loss_train_sum = 0
        acc_train_sum = 0
        se_train_sum = 0
        sp_train_sum = 0
        num_train = 0
        iter = 0
        for idx, (batch_l, batch_u) in enumerate(zip(train_loader['l_loader'], train_loader['u_loader'])):
            img_l, label_l = batch_l['image'].cuda().float(), batch_l['label'].cuda().float()
            img_u = batch_u['img_w'].cuda().float()

            num_labeled = img_l.size(0)
            num_unlabeled = img_u.size(0)
            image_all = torch.cat([img_l, img_u], dim=0)

            # Step 1: Supervised Learning when epoch < sup_only_epoch
            if epoch < config.train.sup_only_epoch:
                # Forward pass: labeled data
                outs_student = model_student(img_l)
                pred_l_sigmoid = torch.sigmoid(outs_student)

                # Supervised loss (BCELoss + Dice Loss)
                # pred_l_sigmoid = torch.sigmoid(pred)
                losses_l1 = [function(pred_l_sigmoid, label_l) for function in criterion]
                sup_loss_1 = (losses_l1[0] + losses_l1[1]) / 2.0
                total_sup_loss += sup_loss_1.item()

                # Teacher forward pass (no gradient updates for teacher)
                # model_teacher.train()
                # _ = model_teacher(img_l)

                unsup_loss = 0 
                contra_loss = 0 

            else:  # Semi-supervised learning phase (using teacher's pseudo-labels)
                if epoch == config.train.sup_only_epoch:
                    # Copy student parameters to teacher
                    with torch.no_grad():
                        for t_params, s_params in zip(model_teacher.parameters(), model_student.parameters()):
                            t_params.data = s_params.data

                # Create pseudo-labels from teacher model
                model_teacher.eval()
                pred_u_teacher = model_teacher(img_u)
                pred_u_teacher = torch.sigmoid(pred_u_teacher)
                # logits_u_aug, label_u_aug = torch.max(pred_u_teacher, dim=1)
                pseudo_mask = (pred_u_teacher > config.semi.conf_thresh).float()
                # Forward pass: both labeled and unlabeled images
                outs = model_student(image_all)
                pred_all = outs
                pred_l, pred_u = pred_all[:num_labeled], pred_all[num_labeled:]
                pred_l = torch.sigmoid(pred_l)
                pred_u = torch.sigmoid(pred_u)
                # device = pred_all.device
                # # Update memory bank with features from current batch
                # for i in range(num_classes):
                #     memobank[i].append(rep_all[i].to(device))  # Append features to memory bank
                #     if len(memobank[i]) > queue_size[i]:
                #         memobank[i] = memobank[i][1:]  # Remove the oldest feature if the memory is full
        
                # Supervised loss
                losses_l1 = [function(pred_l, label_l) for function in criterion]
                sup_loss_1 = (losses_l1[0] + losses_l1[1]) / 2.0
                total_sup_loss += sup_loss_1.item()

                # Teacher forward pass
                model_teacher.train()
                with torch.no_grad():
                    out_t = model_teacher(image_all)
                    pred_all_teacher = torch.sigmoid(out_t)
                # label_u_aug = label_u_aug.unsqueeze(1).float()  # Thêm chiều tại vị trí 1
                # pred_u = pred_u.squeeze(1) 
                # Unsupervised loss
                losses_u = [function(pred_u, pseudo_mask) for function in criterion]
                unsup_loss = (losses_u[0] + losses_u[1]) / 2.0
                total_unsup_loss += unsup_loss.item()

                # Contrastive loss (using memory bank)
                # contra_loss = compute_contra_memobank_loss(rep_all, rep_all_teacher, label_l, label_u_aug, num_labeled, memobank, queue_size)
                # total_contra_loss += contra_loss.item()

            # Total loss
            # loss = sup_loss_1 + unsup_loss + contra_loss
            loss = sup_loss_1 + unsup_loss 

            # total_loss += loss.item()
            loss_train_sum += loss.item() * num_labeled
            # Backward pass and optimization
            # optimizer1.zero_grad()
            optimizer2.zero_grad()
            loss.backward()
            # optimizer1.step()
            optimizer2.step()


            # Update teacher model with EMA
            if epoch >= config.train.sup_only_epoch:
                update_teacher_ema(model_student, model_teacher)

      
            
            # calculate metrics
            with torch.no_grad():
                output = pred_l_sigmoid.cpu().numpy() > 0.5
                label = label_l.cpu().numpy()
                assert (output.shape == label.shape)
                dice_train = metrics.dc(output, label)
                iou_train = metrics.jc(output, label)
                # Tính ACC, SE, SP
                acc_train, se_train, sp_train = calc_metrics(output, label)
                
                dice_train_sum += dice_train * num_labeled
                iou_train_sum += iou_train * num_labeled
                acc_train_sum += acc_train * num_labeled
                se_train_sum += se_train * num_labeled
                sp_train_sum += sp_train * num_labeled
                

                
            file_log.write('Epoch {}, iter {}, Dice Sup Loss: {}, BCE Sup Loss: {}\n'.format(
                epoch, iter + 1, round(losses_l1[1].item(), 5), round(losses_l1[0].item(), 5)
            ))
            file_log.flush()
            print('Epoch {}, iter {}, Dice Sup Loss: {}, BCE Sup Loss: {}'.format(
                epoch, iter + 1, round(losses_l1[1].item(), 5), round(losses_l1[0].item(), 5)
            ))
            
            num_train += num_labeled
            iter +=1
            # end one test batch
            if config.debug: break
        file_log.write('Epoch {}, Total train step {} || AVG_loss: {}, Avg Dice score: {}, Avg IOU: {}, Avg ACC: {}, Avg SE: {}, Avg SP: {}\n'.format(
            epoch, 
            iter, 
            round(loss_train_sum / num_train, 5), 
            round(dice_train_sum / num_train, 4), 
            round(iou_train_sum / num_train, 4), 
            round(acc_train_sum / num_train, 4), 
            round(se_train_sum / num_train, 4), 
            round(sp_train_sum / num_train, 4)
        ))
        file_log.flush()

        # print to console
        print('Epoch {}, Total train step {} || AVG_loss: {}, Avg Dice score: {}, Avg IOU: {}, Avg ACC: {}, Avg SE: {}, Avg SP: {}'.format(
            epoch, 
            len(train_loader), 
            round(loss_train_sum / num_train, 5), 
            round(dice_train_sum / num_train, 4), 
            round(iou_train_sum / num_train, 4), 
            round(acc_train_sum / num_train, 4), 
            round(se_train_sum / num_train, 4), 
            round(sp_train_sum / num_train, 4)
        ))                     
        # -----------------------------------------------------------------
        # validate
        # ----------------------------------------------------------------
        model_student.eval()
        
        dice_val_sum = 0
        iou_val_sum = 0
        loss_val_sum = 0
        acc_val_sum = 0
        se_val_sum = 0
        sp_val_sum = 0
        num_val = 0
        for batch_id, batch in enumerate(val_loader):
            img = batch['image'].cuda().float()
            label = batch['label'].cuda().float()
            
            batch_len = img.shape[0]

            with torch.no_grad():
                output = model_student(img)
                
                output = torch.sigmoid(output)

                # calculate loss
                assert (output.shape == label.shape)
                losses = []
                for function in criterion:
                    losses.append(function(output, label))
                loss_val_sum += sum(losses) * batch_len

                # calculate metrics
                output_np = output.cpu().numpy() > 0.5
                label_np = label.cpu().numpy()
                
                dice_val_sum += metrics.dc(output_np, label_np) * batch_len
                iou_val_sum += metrics.jc(output_np, label_np) * batch_len
                
                # calculate additional metrics using your custom functions
                acc, se, sp = calc_metrics(output_np, label_np)  # Get accuracy, sensitivity, specificity

                acc_val_sum += acc * batch_len
                se_val_sum += se * batch_len
                sp_val_sum += sp * batch_len

                num_val += batch_len
                # end one val batch
                if config.debug: break                   
        loss_val_epoch, dice_val_epoch, iou_val_epoch = loss_val_sum/num_val, dice_val_sum/num_val, iou_val_sum/num_val     
        acc_val_epoch, se_val_epoch, sp_val_epoch = acc_val_sum/num_val, se_val_sum/num_val, sp_val_sum/num_val

        # print to log
        file_log.write('Epoch {}, Validation || sum_loss: {}, Dice score: {}, IOU: {}, ACC: {}, SE: {}, SP: {}\n'.
                format(epoch, round(loss_val_epoch.item(), 5), 
                round(dice_val_epoch, 4), round(iou_val_epoch, 4),
                round(acc_val_epoch, 4), round(se_val_epoch, 4),
                round(sp_val_epoch, 4)))
        file_log.flush()
        
        # print to console
        print('Epoch {}, Validation || sum_loss: {}, Dice score: {}, IOU: {}, ACC: {}, SE: {}, SP: {}'.
                format(epoch, round(loss_val_epoch.item(), 5), 
                round(dice_val_epoch, 4), round(iou_val_epoch, 4),
                round(acc_val_epoch, 4), round(se_val_epoch, 4),
                round(sp_val_epoch, 4)))
 
        # Scheduler step
        # scheduler1.step()
        scheduler2.step()


        # Save the best model based on loss
        if se_val_epoch > max_se:
            torch.save(model_student.state_dict(), best_model_dir)

            max_se = se_val_epoch
            best_epoch = epoch
            file_log.write('New best epoch {} based on ACC and SE!===============================\n'.format(epoch))
            file_log.flush()
            print('New best epoch {} based on ACC and SE!==============================='.format(epoch))
        end = time.time()
        time_elapsed = end-start
        file_log.write('Training and evaluating on epoch{} complete in {:.0f}m {:.0f}s\n'.
                    format(epoch, time_elapsed // 60, time_elapsed % 60))
        file_log.flush()
        print('Training and evaluating on epoch{} complete in {:.0f}m {:.0f}s'.
            format(epoch, time_elapsed // 60, time_elapsed % 60))

        # end one epoch
        if config.debug: return

    file_log.write('Complete training ---------------------------------------------------- \n The best epoch is {}\n'.format(best_epoch))
    file_log.flush()
    
    print('Complete training ---------------------------------------------------- \n The best epoch is {}'.format(best_epoch))

    return model_student



# ========================================================================================================
def test(config, model, model_dir, test_loader, criterion):
    model.load_state_dict(torch.load(model_dir), strict=False)

    model.eval()
    dice_test_sum= 0
    iou_test_sum = 0
    loss_test_sum = 0
    acc_test_sum = 0
    se_test_sum = 0
    sp_test_sum = 0
    num_test = 0
    for batch_id, batch in enumerate(test_loader):
        img = batch['image'].cuda().float()
        label = batch['label'].cuda().float()

        batch_len = img.shape[0]
            
        with torch.no_grad():
                
            output = model(img)

            output = torch.sigmoid(output)

            # calculate loss
            assert (output.shape == label.shape)
            losses = []
            for function in criterion:
                losses.append(function(output, label))
            loss_test_sum += sum(losses)*batch_len

            # calculate metrics
            output = output.cpu().numpy() > 0.5
            label = label.cpu().numpy()
            dice_test_sum += metrics.dc(output, label)*batch_len
            iou_test_sum += metrics.jc(output, label)*batch_len
            acc, se, sp = calc_metrics(output, label)

            acc_test_sum += acc * batch_len
            se_test_sum += se * batch_len
            sp_test_sum += sp * batch_len
            num_test += batch_len
            # end one test batch
            if config.debug: break

    # logging results for one dataset
    loss_test_epoch, dice_test_epoch, iou_test_epoch = loss_test_sum/num_test, dice_test_sum/num_test, iou_test_sum/num_test
    acc_test_epoch = acc_test_sum / num_test
    se_test_epoch = se_test_sum / num_test
    sp_test_epoch = sp_test_sum / num_test

    with open(test_results_dir, 'w') as f:
        f.write(f'Loss: {round(loss_test_epoch.item(),4)}, Dice: {round(dice_test_epoch,4)}, IOU: {round(iou_test_epoch,4)}, '
                f'ACC: {round(acc_test_epoch,4)}, SE: {round(se_test_epoch,4)}, SP: {round(sp_test_epoch,4)}')  # Removed AUC

    # Print
    print('========================================================================================')
    print(f'Test || Loss: {round(loss_test_epoch.item(),4)}, Dice: {round(dice_test_epoch,4)}, IOU: {round(iou_test_epoch,4)}, '
          f'ACC: {round(acc_test_epoch,4)}, SE: {round(se_test_epoch,4)}, SP: {round(sp_test_epoch,4)}')  # Removed AUC
    
    file_log.write('========================================================================================\n')
    file_log.write(f'Test || Loss: {round(loss_test_epoch.item(),4)}, Dice: {round(dice_test_epoch,4)}, IOU: {round(iou_test_epoch,4)}, '
                   f'ACC: {round(acc_test_epoch,4)}, SE: {round(se_test_epoch,4)}, SP: {round(sp_test_epoch,4)}\n')  # Removed AUC
    file_log.flush()

    return



if __name__=='__main__':
    now = datetime.now()
    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser(description='Train experiment')
    parser.add_argument('--exp', type=str,default='tmp')
    parser.add_argument('--config_yml', type=str,default='Configs/multi_train_local.yml')
    parser.add_argument('--adapt_method', type=str, default=False)
    parser.add_argument('--num_domains', type=str, default=False)
    parser.add_argument('--dataset', type=str, nargs='+', default='chase_db1')
    parser.add_argument('--k_fold', type=str, default='No')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--fold', type=int, default=1)
    parser.add_argument('--consistency', type=float,
                    default=0.1, help='consistency')
    parser.add_argument('--consistency_rampup', type=float,
                    default=200.0, help='consistency_rampup')
    args = parser.parse_args()
    config = yaml.load(open(args.config_yml), Loader=yaml.FullLoader)
    config['model_adapt']['adapt_method']=args.adapt_method
    config['model_adapt']['num_domains']=args.num_domains
    config['data']['k_fold'] = args.k_fold
    config['seed'] = args.seed
    config['fold'] = args.fold
    
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    fix_all_seed(config['seed'])

    # print config and args
    print(yaml.dump(config, default_flow_style=False))
    for arg in vars(args):
        print("{:<20}: {}".format(arg, getattr(args, arg)))
    
    store_config = config
    config = DotDict(config)
    
    # logging tensorbord, config, best model
    exp_dir = '{}/{}_{}/fold{}'.format(config.data.save_folder, args.exp, config['data']['supervised_ratio'], args.fold)
    os.makedirs(exp_dir, exist_ok=True)
    best_model_dir = '{}/best.pth'.format(exp_dir)
    test_results_dir = '{}/test_results.txt'.format(exp_dir)

    # store yml file
    if config.debug == False:
        yaml.dump(store_config, open('{}/exp_config.yml'.format(exp_dir), 'w'))
    
    file_log = open('{}/log.txt'.format(exp_dir), 'w')
    main(config)
