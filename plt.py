'''
The default exp_name is tmp. Change it before formal training! isic2018 PH2 DMF SKD
nohup python -u multi_train_adapt.py --exp_name test --config_yml Configs/multi_train_local.yml --model MedFormer --batch_size 1 --adapt_method False --num_domains 1 --dataset drive --k_fold 2 > 2MedFormer_drive.out 2>&1 &

'''
import argparse
from sqlite3 import adapt
import yaml
import os, time
from datetime import datetime

import pandas as pd
import torch.nn as nn
import torch.utils.data
import torch.optim as optim
import medpy.metric.binary as metrics

from Datasets.create_dataset import get_dataset, SkinDataset2
from Utils.losses import dice_loss
from Utils.pieces import DotDict
from Utils.functions import fix_all_seed
from Utils.metrics import calc_metrics
from Utils.metrics import calc_auc
# from Models.Transformer.SwinUnet import SwinUnet
from Models.unetCCT import UNet
from Models.csnet import CSNet  # Thay đổi đường dẫn nếu cần thiết

torch.cuda.empty_cache()

def main(config):
    
    dataset = get_dataset(config, img_size=config.data.img_size, 
                                                supervised_ratio=config.data.supervised_ratio, 
                                                train_aug=config.data.train_aug,
                                                k=config.fold,
                                                lb_dataset=SkinDataset2)

    train_loader = torch.utils.data.DataLoader(dataset['lb_dataset'],
                                                batch_size=config.train.l_batchsize,
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
    print(len(train_loader), len(dataset['lb_dataset']))

    
    model = UNet(in_chns=3, class_num=1)    



    total_trainable_params = sum(
                    p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print('{}M total parameters'.format(total_params/1e6))
    print('{}M total trainable parameters'.format(total_trainable_params/1e6))
    
    # from thop import profile
    # input = torch.randn(1,3,224,224)
    # flops, params = profile(model, (input,))
    # print(f"total flops : {flops/1e9} G")

    # test model
    # x = torch.randn(5,3,224,224)
    # y = model(x)
    # print(y.shape)

    model = model.cuda()
    
    criterion = [nn.BCELoss(), dice_loss]

    # only test
    if config.test.only_test == True:
        test(config, model, config.test.test_model_dir, test_loader, criterion)
    else:
        train_val(config, model, train_loader, val_loader, criterion)
        test(config, model, best_model_dir, test_loader, criterion)



# =======================================================================================================
def train_val(config, model, train_loader, val_loader, criterion):


    # optimizer loss
    if config.train.optimizer.mode == 'adam':
        # optimizer = optim.Adam(model.parameters(), lr=float(config.train.optimizer.adam.lr))
        print('choose wrong optimizer')
    elif config.train.optimizer.mode == 'adamw':
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),lr=float(config.train.optimizer.adamw.lr),
                                weight_decay=float(config.train.optimizer.adamw.weight_decay))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    # ---------------------------------------------------------------------------
    # Training and Validating
    #----------------------------------------------------------------------------
    epochs = config.train.num_epochs
    max_se = 0 # use for record best model
    max_acc = 0 # use for record best model
    max_dice = 0
    best_epoch = 0 # use for recording the best epoch
    # create training data loading iteration
    
    torch.save(model.state_dict(), best_model_dir)
    for epoch in range(epochs):
        start = time.time()
        # ----------------------------------------------------------------------
        # train
        # ---------------------------------------------------------------------
        model.train()

        dice_train_sum = 0
        iou_train_sum = 0
        loss_train_sum = 0
        acc_train_sum = 0
        se_train_sum = 0
        sp_train_sum = 0
        num_train = 0
        for idx, batch in enumerate(train_loader):
            img = batch['image'].cuda().float()
            label = batch['label'].cuda().float()
            
            batch_len = img.shape[0]
            
            output = model(img)
            output = torch.sigmoid(output)
            
            # calculate loss
            assert (output.shape == label.shape)
            losses = []
            for function in criterion:
                losses.append(function(output, label))
            
            loss = sum(losses) / 2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_train_sum += loss.item() * batch_len
            
            # calculate metrics
            with torch.no_grad():
                output = output.cpu().numpy() > 0.5
                label = label.cpu().numpy()
                assert (output.shape == label.shape)
                dice_train = metrics.dc(output, label)
                iou_train = metrics.jc(output, label)
                # Tính ACC, SE, SP
                acc_train, se_train, sp_train = calc_metrics(output, label)


                
                dice_train_sum += dice_train * batch_len
                iou_train_sum += iou_train * batch_len
                acc_train_sum += acc_train * batch_len
                se_train_sum += se_train * batch_len
                sp_train_sum += sp_train * batch_len
                
            iter = epoch * len(train_loader) + idx
                
            file_log.write('Epoch {}, iter {}, Dice Sup Loss: {}, BCE Sup Loss: {}\n'.format(
                epoch, iter + 1, round(losses[1].item(), 5), round(losses[0].item(), 5)
            ))
            file_log.flush()
            print('Epoch {}, iter {}, Dice Sup Loss: {}, BCE Sup Loss: {}'.format(
                epoch, iter + 1, round(losses[1].item(), 5), round(losses[0].item(), 5)
            ))
            
            num_train += batch_len
            
            # end one test batch
            if config.debug: break
                

        # # print
        # file_log.write('Epoch {}, Total train step {} || AVG_loss: {}, Avg Dice score: {}, Avg IOU: {}\n'.format(epoch, 
        #                                                                                               iter, 
        #                                                                                               round(loss_train_sum / num_train,5), 
        #                                                                                               round(dice_train_sum/num_train,4), 
        #                                                                                               round(iou_train_sum/num_train,4)))
        # file_log.flush()
        # print('Epoch {}, Total train step {} || AVG_loss: {}, Avg Dice score: {}, Avg IOU: {}'.format(epoch, 
        #                                                                                               len(train_loader), 
        #                                                                                               round(loss_train_sum / num_train,5), 
        #                                                                                               round(dice_train_sum/num_train,4), 
        #                                                                                               round(iou_train_sum/num_train,4)))
        # print to file
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

        # logging per epoch for one dataset
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


        # # scheduler step, record lr
        # scheduler.step()

        # # store model using the average iou
        # if dice_val_epoch > max_dice:
        #     torch.save(model.state_dict(), best_model_dir)
        #     max_dice = dice_val_epoch
        #     best_epoch = epoch
        #     file_log.write('New best epoch {}!===============================\n'.format(epoch))
        #     file_log.flush()
        #     print('New best epoch {}!==============================='.format(epoch))
        
        # end = time.time()
        # time_elapsed = end-start
        # file_log.write('Training and evaluating on epoch{} complete in {:.0f}m {:.0f}s\n'.
        #             format(epoch, time_elapsed // 60, time_elapsed % 60))
        # file_log.flush()
        # print('Training and evaluating on epoch{} complete in {:.0f}m {:.0f}s'.
        #     format(epoch, time_elapsed // 60, time_elapsed % 60))
                # scheduler step, record lr
        scheduler.step()

        # store model using the average acc and se
        if se_val_epoch > max_se and acc_val_epoch > max_acc:
            torch.save(model.state_dict(), best_model_dir)
            max_acc = acc_val_epoch
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

    return 




# ========================================================================================================
# def test(config, model, model_dir, test_loader, criterion):
#     model.load_state_dict(torch.load(model_dir))
#     model.eval()
#     dice_test_sum= 0
#     iou_test_sum = 0
#     loss_test_sum = 0
#     num_test = 0
#     for batch_id, batch in enumerate(test_loader):
#         img = batch['image'].cuda().float()
#         label = batch['label'].cuda().float()

#         batch_len = img.shape[0]
            
#         with torch.no_grad():
                
#             output = model(img)

#             output = torch.sigmoid(output)

#             # calculate loss
#             assert (output.shape == label.shape)
#             losses = []
#             for function in criterion:
#                 losses.append(function(output, label))
#             loss_test_sum += sum(losses)*batch_len

#             # calculate metrics
#             output = output.cpu().numpy() > 0.5
#             label = label.cpu().numpy()
#             dice_test_sum += metrics.dc(output, label)*batch_len
#             iou_test_sum += metrics.jc(output, label)*batch_len

#             num_test += batch_len
#             # end one test batch
            
#             if config.debug: break

#     # logging results for one dataset
#     loss_test_epoch, dice_test_epoch, iou_test_epoch = loss_test_sum/num_test, dice_test_sum/num_test, iou_test_sum/num_test


#     # logging average and store results
#     with open(test_results_dir, 'w') as f:
#         f.write(f'loss: {loss_test_epoch.item()}, Dice_score {dice_test_epoch}, IOU: {iou_test_epoch}')

#     # print
#     file_log.write('========================================================================================\n')
#     file_log.write('Test || Average loss: {}, Dice score: {}, IOU: {}\n'.
#                         format(round(loss_test_epoch.item(),5), 
#                         round(dice_test_epoch,4), round(iou_test_epoch,4)))
#     file_log.flush()
    
#     print('========================================================================================')
#     print('Test || Average loss: {}, Dice score: {}, IOU: {}'.
#             format(round(loss_test_epoch.item(),5), 
#             round(dice_test_epoch,4), round(iou_test_epoch,4)))

#     return
def test(config, model, model_dir, test_loader, criterion):
    model.load_state_dict(torch.load(model_dir))
    model.eval()
    
    dice_test_sum = 0
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
            loss_test_sum += sum(losses) * batch_len

            # calculate metrics
            output_np = output.cpu().numpy() > 0.5
            label_np = label.cpu().numpy()

            # Dice and IOU
            dice_test_sum += metrics.dc(output_np, label_np) * batch_len
            iou_test_sum += metrics.jc(output_np, label_np) * batch_len

            # Use calc_metrics to get accuracy, sensitivity, specificity
            acc, se, sp = calc_metrics(output_np, label_np)

            acc_test_sum += acc * batch_len
            se_test_sum += se * batch_len
            sp_test_sum += sp * batch_len
            
            num_test += batch_len
            
            if config.debug:
                break

    # Calculate average metrics
    loss_test_epoch = loss_test_sum / num_test
    dice_test_epoch = dice_test_sum / num_test
    iou_test_epoch = iou_test_sum / num_test
    acc_test_epoch = acc_test_sum / num_test
    se_test_epoch = se_test_sum / num_test
    sp_test_epoch = sp_test_sum / num_test

    # Logging average and storing results
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
# """
# Training script for CS-Net
# """
# import os
# import argparse
# import yaml
# import time
# from datetime import datetime
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader
# import medpy.metric.binary as metrics

# from Datasets.create_dataset import get_dataset, SkinDataset2
# from Models.csnet import CSNet
# from Utils.losses import dice_loss
# from Utils.pieces import DotDict
# from Utils.functions import fix_all_seed
# from Utils.dice_loss_single_class import dice_coeff_loss
# from Models.unet import UNet
# # Cấu hình
# def main(config):
#     dataset = get_dataset(config, img_size=config.data.img_size, 
#                            supervised_ratio=config.data.supervised_ratio, 
#                            train_aug=config.data.train_aug,
#                            k=config.fold,
#                            lb_dataset=SkinDataset2)

#     train_loader = DataLoader(dataset['lb_dataset'],
#                               batch_size=config.train.l_batchsize,
#                               shuffle=True,
#                               num_workers=config.train.num_workers,
#                               pin_memory=True,
#                               drop_last=False)
#     val_loader = DataLoader(dataset['val_dataset'],
#                             batch_size=config.test.batch_size,
#                             shuffle=False,
#                             num_workers=config.test.num_workers,
#                             pin_memory=True,
#                             drop_last=False)
#     print(len(train_loader), len(dataset['lb_dataset']))

#     # model = CSNet(classes=1, channels=3).cuda()
#     model = UNet().cuda()
#     criterion = [nn.BCELoss(), dice_loss] #nn.MSELoss()
#     # criterion = [nn.BCELoss(), dice_coeff_loss]
#     if config.test.only_test:
#         test(config, model, config.test.test_model_dir, val_loader, criterion)
#     else:
#         train_val(config, model, train_loader, val_loader, criterion)

# def train_val(config, model, train_loader, val_loader, criterion):
#     torch.autograd.set_detect_anomaly(True)

#     optimizer = optim.AdamW(model.parameters(), lr=float(config.train.optimizer.adamw.lr),
#                             weight_decay=float(config.train.optimizer.adamw.weight_decay))
#     scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

#     epochs = config.train.num_epochs
#     max_dice = 0
#     best_epoch = 0

#     for epoch in range(epochs):
#         start = time.time()
#         model.train()
#         loss_train_sum = 0
#         num_train = 0

#         for idx, batch in enumerate(train_loader):
#             img = batch['image'].cuda().float()
#             label = batch['label'].cuda().float()
            
#             output = model(img)
#             output = torch.sigmoid(output)

#             losses = [function(output, label) for function in criterion]
#             # loss = sum(losses) / 2
#             loss = sum(losses)
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#             loss_train_sum += loss.item() * img.size(0)

#             num_train += img.size(0)

#             # Ghi log cho mỗi bước huấn luyện
#             print(f'Epoch {epoch}, Iter {idx}, Loss: {loss.item()}')

#         loss_train_avg = loss_train_sum / num_train
#         print(f'Epoch {epoch}, Average Training Loss: {loss_train_avg}')

#         # Validate
#         model.eval()
#         dice_val_sum = 0
#         num_val = 0

#         for batch in val_loader:
#             img = batch['image'].cuda().float()
#             label = batch['label'].cuda().float()
            
#             with torch.no_grad():
#                 output = model(img)
#                 output = torch.sigmoid(output)
#                 losses = [function(output, label) for function in criterion]
#                 dice_val = metrics.dc((output > 0.5).cpu().numpy(), label.cpu().numpy())
#                 dice_val_sum += dice_val * img.size(0)
#                 num_val += img.size(0)

#         dice_val_avg = dice_val_sum / num_val
#         print(f'Epoch {epoch}, Average Validation Dice: {dice_val_avg}')

#         if dice_val_avg > max_dice:
#             torch.save(model.state_dict(), f'best_model_epoch_{epoch}.pth')
#             max_dice = dice_val_avg
#             best_epoch = epoch

#         scheduler.step()
#         end = time.time()
#         print(f'Training and validating epoch {epoch} complete in {end - start:.2f} seconds')

# def test(config, model, model_dir, test_loader, criterion):
#     model.load_state_dict(torch.load(model_dir))
#     model.eval()
#     loss_test_sum = 0
#     num_test = 0

#     for batch in test_loader:
#         img = batch['image'].cuda().float()
#         label = batch['label'].cuda().float()

#         with torch.no_grad():
#             output = model(img)
#             output = torch.sigmoid(output)
#             losses = [function(output, label) for function in criterion]
#             loss_test_sum += sum(losses).item() * img.size(0)

#             num_test += img.size(0)

#     loss_test_avg = loss_test_sum / num_test
#     print(f'Test Average Loss: {loss_test_avg}')

# if __name__ == '__main__':
#     now = datetime.now()
#     parser = argparse.ArgumentParser(description='Train experiment')
#     parser.add_argument('--exp', type=str, default='tmp')
#     parser.add_argument('--config_yml', type=str, default='Configs/multi_train_local.yml')
#     parser.add_argument('--adapt_method', type=str, default=False)
#     parser.add_argument('--num_domains', type=str, default=False)
#     parser.add_argument('--dataset', type=str, nargs='+', default='chase_db1')
#     parser.add_argument('--k_fold', type=str, default='No')
#     parser.add_argument('--gpu', type=str, default='0')
#     parser.add_argument('--seed', type=int, default=1)
#     parser.add_argument('--fold', type=int, default=1)
#     args = parser.parse_args()
#     config = yaml.load(open(args.config_yml), Loader=yaml.FullLoader)
#     config['model_adapt']['adapt_method'] = args.adapt_method
#     config['model_adapt']['num_domains'] = args.num_domains
#     config['data']['k_fold'] = args.k_fold
#     config['seed'] = args.seed
#     config['fold'] = args.fold

#     os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#     os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
#     fix_all_seed(config['seed'])

#     # Print config and args
#     print(yaml.dump(config, default_flow_style=False))
#     for arg in vars(args):
#         print("{:<20}: {}".format(arg, getattr(args, arg)))

#     store_config = config
#     config = DotDict(config)

#     # Create directories for saving models and logs
#     exp_dir = f'{config.data.save_folder}/{args.exp}_{config.data.supervised_ratio}/fold{args.fold}'
#     os.makedirs(exp_dir, exist_ok=True)
#     best_model_dir = f'{exp_dir}/best.pth'
#     test_results_dir = f'{exp_dir}/test_results.txt'

#     # Store YAML file
#     if not config.debug:
#         yaml.dump(store_config, open(f'{exp_dir}/exp_config.yml', 'w'))
    
#     file_log = open(f'{exp_dir}/log.txt', 'w')
#     main(config)
