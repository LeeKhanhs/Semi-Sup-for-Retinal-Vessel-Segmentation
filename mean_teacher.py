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



    def create_model(ema=False):
        # Network definition
        model = UNet()
        if ema:
            for param in model.parameters():
                param.detach_()
        return model
    model = create_model()
    ema_model = create_model(ema=True)

    total_trainable_params = sum(
                    p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print('{}M total parameters'.format(total_params/1e6))
    print('{}M total trainable parameters'.format(total_trainable_params/1e6))

    model = model.cuda()
    ema_model = ema_model.cuda()
    
    criterion = [nn.BCELoss(), dice_loss]

    model = train_val(config, ema_model, model, train_loader, val_loader, criterion)
    test(config, model, best_model_dir, test_loader, criterion)
    
def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
def train_val(config, ema_model, model, train_loader, val_loader, criterion):
    """Training and validation loop with memory bank and contrastive loss"""
    
    # Optimizer setup
    if config.train.optimizer.mode == 'adam':
        print('Choose wrong optimizer')
    elif config.train.optimizer.mode == 'adamw':
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=float(config.train.optimizer.adamw.lr),
            weight_decay=float(config.train.optimizer.adamw.weight_decay)
        )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    # Training loop
    epochs = config.train.num_epochs
    iter = 0  # Initialize iteration counter
    num_train = 0  # Initialize number of trained samples
    max_iou = 0 # use for record best model
    max_dice = 0 # use for record best model
    max_se = 0 # use for record best model
    max_acc = 0 # use for record best model
    best_epoch = 0 
    for epoch in range(epochs):
        start = time.time()

        # ----------------------------------------------------------------------
        # Training Phase
        # ----------------------------------------------------------------------

        model.train()  # Set model to training mode
        loss_train_sum = 0
        total_sup_loss = 0
        total_unsup_loss = 0
        dice_train_sum = 0
        iou_train_sum = 0
        acc_train_sum = 0
        se_train_sum = 0
        sp_train_sum = 0

        # Iterate over batches of labeled and unlabeled data
        for batch_idx, (batch_l, batch_u) in enumerate(zip(train_loader['l_loader'], train_loader['u_loader'])):
            # Load labeled and unlabeled data
            img_l, label_l = batch_l['image'].cuda().float(), batch_l['label'].cuda().float()
            img_u = batch_u['img_w'].cuda().float()  # Unlabeled images
            sup_batch_len = img_l.shape[0]

            # Forward pass for labeled data (supervised loss)
            output_l = model(img_l)
            output_l_sigmoid = torch.sigmoid(output_l)  # Apply sigmoid for binary segmentation

            #Add noise
            # noise = torch.clamp(torch.randn_like(
            #     img_u) * 0.1, -0.1, 0.1)
            # img_u = img_u + noise


            # Compute supervised loss (labeled data)
            losses = []
            for function in criterion:
                losses.append(function(output_l_sigmoid, label_l))
             

            # Accumulate total supervised loss
            total_sup_loss = (losses[0] + losses[1]) / 2.0

            # Combine supervised loss for labeled data

            # Forward pass on unlabeled data (for consistency loss with EMA model)
            with torch.no_grad():
                ema_output_u = ema_model(img_u)
                ema_output_u_sigmoid = torch.sigmoid(ema_output_u)

            # Compute unsupervised loss (consistency loss between model and EMA model)
            output_u = model(img_u)
            output_u_sigmoid = torch.sigmoid(output_u)
            consistency_loss = torch.mean((output_u_sigmoid - ema_output_u_sigmoid) ** 2)

            # Accumulate unsupervised loss
            total_unsup_loss += consistency_loss.item()

            # Add consistency loss to the total batch loss
            total_loss_batch = total_sup_loss + consistency_loss

            # Backpropagation and optimization
            optimizer.zero_grad()
            total_loss_batch.backward()
            optimizer.step()

            # Update EMA model parameters
            update_ema_variables(model, ema_model, alpha=config.train.ema_decay, global_step=epoch)

            # Log metrics for each batch
            loss_train_sum += total_loss_batch.item() * sup_batch_len

            # Calculate metrics
            with torch.no_grad():
                output = output_l_sigmoid.cpu().numpy() > 0.5  # Convert to binary output
                label = label_l.cpu().numpy()  # Ground truth labels

                # Assert output and label have the same shape
                assert output.shape == label.shape

                # Calculate metrics: Dice, IoU, ACC, SE, SP
                dice_train = metrics.dc(output, label)
                iou_train = metrics.jc(output, label)
                acc_train, se_train, sp_train = calc_metrics(output, label)

                # Accumulate metric sums for averaging
                num_labeled = len(label_l)
                dice_train_sum += dice_train * num_labeled
                iou_train_sum += iou_train * num_labeled
                acc_train_sum += acc_train * num_labeled
                se_train_sum += se_train * num_labeled
                sp_train_sum += sp_train * num_labeled

            # Increment train samples count
            num_train += num_labeled
            iter += 1

            # Optionally stop early for debugging
            if config.debug: 
                break

        # Epoch-level logging
        avg_loss = loss_train_sum / num_train
        avg_dice = dice_train_sum / num_train
        avg_iou = iou_train_sum / num_train
        avg_acc = acc_train_sum / num_train
        avg_se = se_train_sum / num_train
        avg_sp = sp_train_sum / num_train

        # Log results to file
        file_log.write(f'Epoch {epoch}, Total Train Steps {iter} || '
                       f'Avg Loss: {avg_loss:.5f}, Avg Dice: {avg_dice:.4f}, '
                       f'Avg IoU: {avg_iou:.4f}, Avg ACC: {avg_acc:.4f}, '
                       f'Avg SE: {avg_se:.4f}, Avg SP: {avg_sp:.4f}\n')
        file_log.flush()

        # Print to console
        print(f'Epoch {epoch}, Total Train Steps {len(train_loader["l_loader"])} || '
              f'Avg Loss: {avg_loss:.5f}, Avg Dice: {avg_dice:.4f}, '
              f'Avg IoU: {avg_iou:.4f}, Avg ACC: {avg_acc:.4f}, '
              f'Avg SE: {avg_se:.4f}, Avg SP: {avg_sp:.4f}')                     
        # -----------------------------------------------------------------
        # validate
        # ----------------------------------------------------------------
        model.eval()
        
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
                output = model(img)
                
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
        scheduler.step()


        # Save the best model based on loss
        if dice_val_epoch > max_dice:
            torch.save(model.state_dict(), best_model_dir)

            max_dice = dice_val_epoch
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

    return model



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
