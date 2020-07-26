# encoding: utf-8

import sys
import argparse
import logging
import os
import random
import numpy as np
import math
import torch
import json
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR,ExponentialLR,ReduceLROnPlateau
from torch.autograd import Variable
import torch.nn as nn
from tqdm import tqdm
tqdm.monitor_interval = 0
import torchnet
#  from torchnet.meter import ConfusionMeter,aucmeter
from torchnet.logger import VisdomPlotLogger, VisdomLogger,MeterLogger
import torch.backends.cudnn as cudnn

from utils import  utils
from utils.utils import str2bool
import  data_loader
from model import HCN
import numpy as np
import Loss

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_dir', default='/data0/', help="root directory for all the datasets")
parser.add_argument('--dataset_name', default='NTU-RGB-D-CV', help="dataset name ") # NTU-RGB-D-CS,NTU-RGB-D-CV
parser.add_argument('--model_dir', default='./',
                    help="parents directory of model")

parser.add_argument('--model_name', default='HCN',help="model name")
parser.add_argument('--load_model',
        help='Optional, load trained models')
parser.add_argument('--load',
        type=str2bool,
        default=False,
        help='load a trained model or not ')
parser.add_argument('--mode', default='train', help='train,test,or load_train')
parser.add_argument('--num', default='01', help='num of trials (type: list)')
parser.add_argument('--finetune',default = False,help='transfer last fc layer to 30 channel')
parser.add_argument('--visdom_env',default = 'HCN',help = 'visdom Env')

def train(epoch_num,model, ad_net, grl,ad_net_m,grl_m,ad_net_t, grl_t,ad_net_mt,grl_mt,optimizer, loss_fn, dataloader, metrics, params,logger):
    """Train the model on `num_steps` batches

    Args:
        model: (torch.nn.Module) the neural network
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn:
        dataloader:
        metrics: (dict)
        params: (Params) hyperparameters
    """

    # set model to training mode
    model.train()

    # summary for current training loop and a running average object for loss
    summ = []
    loss_avg = utils.RunningAverage()
    confusion_meter = torchnet.meter.ConfusionMeter(params.model_args["num_class"], normalized=True)
    confusion_meter.reset()
    trade_off=0.1

    # Use tqdm for progress bar
    with tqdm(total=len(dataloader)) as t:
        for i, (data_batch, labels_batch) in enumerate(dataloader):
            # move to GPU if available
            if params.cuda:
                if params.data_parallel:
                    data_batch, labels_batch = data_batch.cuda(async=True), labels_batch.cuda(async=True)
                else:
                    data_batch, labels_batch = data_batch.cuda(params.gpu_id), labels_batch.cuda(params.gpu_id)

            # convert to torch Variables
            data_batch, labels_batch = Variable(data_batch), Variable(labels_batch)
            batch_size = data_batch.shape[0]
            #print(batch_size)
            z = data_batch[:,:,:,:,:,0]    # get source data to train
            y = data_batch[:,:,:,:,:,1]
            data_batch = torch.cat((z,y),dim =0)

            # compute model output and loss
            output_batch, feature = model(data_batch.float(),target=labels_batch)
            
            ad_target =  Variable(torch.from_numpy(np.array([[1]] * batch_size + [[0]] * batch_size)).float())  
            ad_target = ad_target.cuda()  
            out_labels = output_batch.clone()
            output_batch = output_batch[:batch_size]
            # feature = torch.mean(feature,1)
# 
            feature = feature.view(feature.size(0), -1)
            softmax_out = nn.Softmax(dim=1)(out_labels)
            entropy = Loss.Entropy(softmax_out)
            loss_bag = loss_fn(output_batch,labels_batch,current_epoch=params.current_epoch, params=params)

            transfer_loss = Loss.CDAN([feature,softmax_out],ad_net,grl,entropy,grl.get_coeff(),None)

            loss = loss_bag['ls_CE'] + (transfer_loss) * trade_off 


            confusion_meter.add(output_batch.data,
                                labels_batch.data)

            # clear previous gradients, compute gradients of all variables wrt loss
            optimizer.zero_grad()
            loss.backward()


            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), params.clip*params.batch_size_train)

            # print(total_norm,params.clip*params.batch_size)

            # performs updates using calculated gradients
            optimizer.step()




            # Evaluate summaries only once in a while # not every epoch count in train accuracy
            if i % params.save_summary_steps == 0:
                # extract data from torch Variable, move to cpu, convert to numpy arrays
                output_batch = output_batch.data
                labels_batch = labels_batch.data

                # compute all metrics on this batch
                summary_batch = {metric:metrics[metric](output_batch, labels_batch)
                                 for metric in metrics}
                summary_batch['loss'] = loss.data.item()
                summary_batch['ls_BCE'] = transfer_loss.data.item() 
                for l,v in loss_bag.items():
                    summary_batch[l]=v.data.item()

                summ.append(summary_batch)

            # update the average loss # main for progress bar, not logger
            loss_running = loss.data.item()
            loss_avg.update(loss_running )

            t.set_postfix(loss_running ='{:05.3f}'.format(loss_avg()))
            t.update()

    # compute mean of all metrics in summary
    #print([metric for metric in summ[0]])

    #metrics_mean = {metric:np.mean([x[metric] for x in summ]) for metric in summ[0]}
    AccTop5_list_train = [x['accuracytop5'][0] for x in summ]
    AccTop1_list_train = [x['accuracytop1'][0] for x in summ]
    AccTop5_mean_train = torch.mean(torch.stack(AccTop5_list_train)).cpu().numpy()
    AccTop1_mean_train = torch.mean(torch.stack(AccTop1_list_train)).cpu().numpy()
    loss_mean_train = np.mean([x['loss'] for x in summ])
    ls_all_mean_train = np.mean([x['ls_all'] for x in summ])
    ls_ce_mean_train = np.mean([x['ls_CE'] for x in summ])
    ls_bce_mean_train = np.mean([x['ls_BCE'] for x in summ])
    metrics_mean = {'accuracytop5':AccTop5_mean_train ,'accuracytop1':AccTop1_mean_train , 'loss':loss_mean_train,'ls_all':ls_all_mean_train,'ls_CE':ls_ce_mean_train,'ls_BCE':ls_bce_mean_train}
    print(metrics_mean)

    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    logger.info("- Train metrics: " + metrics_string)

    return metrics_mean,confusion_meter

def evaluate(model, loss_fn, dataloader, metrics, params,logger):
    """Evaluate the model on `num_steps` batches.

    Args:
        model: (torch.nn.Module) the neural network
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches data
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        num_steps: (int) number of batches to train on, each of size params.batch_size
    """

    # set model to evaluation mode
    # model.train()
    if params.mode == 'test':
        pass
    else:
        model.eval()


    # summary for current eval loop
    summ = []
    confusion_meter = torchnet.meter.ConfusionMeter(params.model_args["num_class"], normalized=True)
    confusion_meter.reset()

    # compute metrics over the dataset
    for data_batch, labels_batch in dataloader:

        # move to GPU if available
        if params.cuda:
            if params.data_parallel:
                data_batch, labels_batch = data_batch.cuda(), labels_batch.cuda()
            else:
                data_batch, labels_batch = data_batch.cuda(params.gpu_id), labels_batch.cuda(params.gpu_id)
        # fetch the next evaluation batch
        data_batch, labels_batch = Variable(data_batch), Variable(labels_batch)
        data_batch = data_batch[:,:,:,:,:,0]
        # compute model output
        output_batch,_ = model(data_batch.float())

        loss_bag  = loss_fn(output_batch,labels_batch,current_epoch=params.current_epoch, params=params)
        loss = loss_bag['ls_all']

        confusion_meter.add(output_batch.data,
                            labels_batch.data)

        # extract data from torch Variable, move to cpu, convert to numpy arrays
        output_batch = output_batch.data
        labels_batch = labels_batch.data

        # compute all metrics on this batch
        summary_batch = {metric: metrics[metric](output_batch, labels_batch)
                         for metric in metrics}
        summary_batch['loss'] = loss.item()
        for l, v in loss_bag.items():
            summary_batch[l] = v.data.item()
        summ.append(summary_batch)

    # compute mean of all metrics in summary
    
    #print([metric for metric in summ[0]])
    #metrics_mean = {metric: np.mean([x[metric] for x in summ]) for metric in summ[0]}
    
    AccTop5_list_eva = [x['accuracytop5'][0] for x in summ]
    AccTop1_list_eva = [x['accuracytop1'][0] for x in summ]
    AccTop5_mean_eva = torch.mean(torch.stack(AccTop5_list_eva)).cpu().numpy()
    AccTop1_mean_eva = torch.mean(torch.stack(AccTop1_list_eva)).cpu().numpy()
    loss_mean_eva = np.mean([x['loss'] for x in summ])
    ls_all_mean_eva = np.mean([x['ls_all'] for x in summ])
    ls_ce_mean_eva = np.mean([x['ls_CE'] for x in summ])
    metrics_mean = {'accuracytop5':AccTop5_mean_eva ,'accuracytop1':AccTop1_mean_eva , 'loss':loss_mean_eva,'ls_all':ls_all_mean_eva,'ls_CE':ls_ce_mean_eva}

    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    logger.info("- Eval metrics : " + metrics_string)

    return metrics_mean,confusion_meter

def single_evaluate(model, loss_fn, dataloader, metrics, params,logger):
    """Evaluate the model on `num_steps` batches.

    Args:
        model: (torch.nn.Module) the neural network
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches data
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        num_steps: (int) number of batches to train on, each of size params.batch_size
    """

    # set model to evaluation mode
    # model.train()
    if params.mode == 'single_test' or params.mode == 'test' :
        pass
    else:
        model.eval()


    # summary for current eval loop
    summ = []
    logits = []
    preds = []
    confusion_meter = torchnet.meter.ConfusionMeter(params.model_args["num_class"], normalized=True)
    confusion_meter.reset()

    # compute metrics over the dataset
    for data_batch, labels_batch in dataloader:

        # move to GPU if available
        if params.cuda:
            if params.data_parallel:
                data_batch, labels_batch = data_batch.cuda(), labels_batch.cuda()
            else:
                data_batch, labels_batch = data_batch.cuda(params.gpu_id), labels_batch.cuda(params.gpu_id)
        # fetch the next evaluation batch
        data_batch, labels_batch = Variable(data_batch), Variable(labels_batch)

        # compute model output
        out = model(data_batch)

        output_batch = out
        # loss = loss_fn(output_batch, labels_batch,current_epoch=None,params=params)
        confusion_meter.add(output_batch.data,
                            labels_batch.data)

        logit, pred = F.log_softmax(output_batch,dim=1).topk(k=5,dim=1, largest=True,sorted= True)
        logits.append(logit)
        preds.append(pred)

        # extract data from torch Variable, move to cpu, convert to numpy arrays
        output_batch = output_batch.data
        labels_batch = labels_batch.data

        # compute all metrics on this batch
        summary_batch = {metric: metrics[metric](output_batch, labels_batch)
                         for metric in metrics}
        # summary_batch['loss'] = loss.item()
        summ.append(summary_batch)

    # compute mean of all metrics in summary
    
    #metrics_mean = {metric: np.mean([x[metric] for x in summ]) for metric in summ[0]}
    
    AccTop5_list_seva = [x['accuracytop5'][0] for x in summ]
    AccTop1_list_seva = [x['accuracytop1'][0] for x in summ]
    AccTop5_mean_seva = torch.mean(torch.stack(AccTop5_list_seva)).cpu().numpy()
    AccTop1_mean_seva = torch.mean(torch.stack(AccTop1_list_seva)).cpu().numpy()
    loss_mean_seva = np.mean([x['loss'] for x in summ])
    ls_all_mean_seva = np.mean([x['ls_all'] for x in summ])
    ls_ce_mean_seva = np.mean([x['ls_CE'] for x in summ])
    metrics_mean = {'accuracytop5':AccTop5_mean_seva ,'accuracytop1':AccTop1_mean_seva , 'loss':loss_mean_seva,'ls_all':ls_all_mean_seva,'ls_CE':ls_ce_mean_seva}

    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    logger.info("- Eval metrics : " + metrics_string)

    logits=torch.cat(logits,dim=0)
    preds = torch.cat(preds,dim=0)

    return metrics_mean,confusion_meter,logits,preds

def train_and_evaluate(model,ad_net,grl,ad_net_m,grl_m ,ad_net_t,grl_t,ad_net_mt,grl_mt,Myacc ,train_dataloader, val_dataloader, optimizer,
                       loss_fn, metrics, params, model_dir,logger,restore_file=None):
    """Train the model and evaluate every epoch.

    Args:
        model: (torch.nn.Module) the neural network
        params: (Params) hyperparameters
        model_dir: (string) directory containing config, weights and log
        restore_file: (string) - name of file to restore from (without its extension .pth.tar)
    """

    best_val_acc = 0.0
    # reload weights from restore_file if specified
    #if args.finetune:
    #    num_ftrs = model.fc8.in_features
    #    model.fc8 = nn.Linear(num_ftrs, 60)
    #    model = model.cuda()
    #logger.info(model)
    if restore_file is not None:
        logging.info("Restoring parameters from {}".format(restore_file))
        checkpoint = utils.load_checkpoint(restore_file, model, optimizer)
        params.start_epoch = checkpoint['epoch']

        best_val_acc = checkpoint['best_val_acc']
        print('best_val_acc=',best_val_acc)
        print(optimizer.state_dict()['param_groups'][0]['lr'], checkpoint['epoch'])

    # learning rate schedulers for different models:
    if params.lr_decay_type == None:
        logging.info("no lr decay")
    else:
        assert params.lr_decay_type in ['multistep','exp','plateau']
        logging.info("lr decay:{}".format(params.lr_decay_type))
    if params.lr_decay_type == 'multistep':
        scheduler = MultiStepLR(optimizer, milestones=params.lr_step, gamma=params.scheduler_gamma,last_epoch= params.start_epoch-1)

    elif params.lr_decay_type == 'exp':
        scheduler =  ExponentialLR(optimizer, gamma=params.scheduler_gamma2,
                            last_epoch=params.start_epoch - 1)
    elif params.lr_decay_type == 'plateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=params.scheduler_gamma3, patience=params.patience, verbose=False,
                                                   threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0,
                                                   eps=1e-08)
    if args.finetune:
        num_ftrs = model.fc8.in_features
        model.fc8 = nn.Linear(num_ftrs, 30)
        model = model.cuda()
    logger.info(model)
    for epoch in range(params.start_epoch,params.num_epochs):
        params.current_epoch = epoch
        if params.lr_decay_type != 'plateau':
            scheduler.step()

        # Run one epoch
        logger.info("Epoch {}/{}".format(epoch + 1, params.num_epochs))

        # compute number of batches in one epoch (one full pass over the training set)
        train_metrics,train_confusion_meter = train(epoch , model, ad_net, grl,ad_net_m,grl_m,ad_net_t,grl_t,ad_net_mt,grl_mt ,optimizer, loss_fn, train_dataloader, metrics, params,logger)

        # Evaluate for one epoch on validation set
        val_metrics,val_confusion_meter = evaluate(model, loss_fn, val_dataloader, metrics, params,logger)


        # vis logger
        accs = [100. * (1 - train_metrics['accuracytop1']),100. * (1 - train_metrics['accuracytop5']),
               100. * (1 - val_metrics['accuracytop1']),100. * (1 - val_metrics['accuracytop5']),]
        error_logger15.log([epoch]*4,accs )
        Myacc.append(100. * (1 - val_metrics['accuracytop1']))

        losses = [train_metrics['loss'],val_metrics['loss']]
        loss_logger.log([epoch]*2,losses )
        train_confusion_logger.log(train_confusion_meter.value())
        test_confusion_logger.log(val_confusion_meter.value())

        # log split loss
        if epoch == params.start_epoch:
            loss_key_train = []
            loss_key_val = []
            for key in [k for k,v in train_metrics.items()] :
                if 'ls' in key: loss_key_train.append(key)
            for key in [k for k,v in val_metrics.items()] :
                if 'ls' in key: loss_key_val.append(key)
            loss_split_key = ['train_'+k for k in loss_key_train] + ['val_'+k for k in loss_key_val]
            loss_logger_split.opts['legend'] = loss_split_key

        loss_split = [train_metrics[k] for k in loss_key_train]+[val_metrics[k] for k in loss_key_val]
        loss_logger_split.log([epoch] * len(loss_split_key),loss_split)

        if params.lr_decay_type == 'plateau':
            scheduler.step(val_metrics['ls_all'])

        val_acc = val_metrics['accuracytop1']
        is_best = val_acc >= best_val_acc
        # Save weights
        utils.save_checkpoint( {'epoch': epoch + 1,
                               'state_dict': model.state_dict(),
                               'optim_dict': optimizer.state_dict(),
                                'best_val_acc':best_val_acc
                                },
                              epoch= epoch+1,
                              is_best=is_best,
                              save_best_ever_n_epoch = params.save_best_ever_n_epoch,
                              checkpointpath=params.experiment_path+'/checkpoint',
                               start_epoch = params.start_epoch)

        val_metrics['best_epoch'] = epoch + 1
        # If best_eval, best_save_path, metric
        if is_best:
            logger.info("- Found new best accuracy")
            best_val_acc = val_acc

            # Save best val metrics in a json file in the model directory
            best_json_path = os.path.join(params.experiment_path, "metrics_val_best_weights.json")
            utils.save_dict_to_json(val_metrics, best_json_path)

        # Save latest val metrics in a json file in the model directory
        last_json_path = os.path.join(params.experiment_path, "metrics_val_last_weights.json")
        utils.save_dict_to_json(val_metrics, last_json_path)

def test_only(model,train_dataloader, val_dataloader, optimizer,
                       loss_fn, metrics, params, model_dir,logger,restore_file=None):
    # reload weights from restore_file if specified
    if restore_file is not None:
        logging.info("Restoring parameters from {}".format(restore_file))
        checkpoint = utils.load_checkpoint(restore_file, model, optimizer)

        best_val_acc = checkpoint['best_val_acc']
        params.current_epoch = checkpoint['epoch']
        print('best_val_acc=',best_val_acc)
        print(optimizer.state_dict()['param_groups'][0]['lr'], checkpoint['epoch'])

    train_confusion_logger = VisdomLogger('heatmap', port=port,
                                          opts={'title': params.experiment_path + 'train_Confusion matrix',
                                                'columnnames': columnnames, 'rownames': rownames},env='Test')
    test_confusion_logger = VisdomLogger('heatmap', port=port,
                                         opts={'title': params.experiment_path + 'test_Confusion matrix',
                                               'columnnames': columnnames, 'rownames': rownames},env='Test')
    diff_confusion_logger = VisdomLogger('heatmap', port=port,
                                         opts={'title': params.experiment_path + 'diff_Confusion matrix',
                                               'columnnames': columnnames, 'rownames': rownames},env='Test')

    # Evaluate for one epoch on validation set
    # model.train()
    model.eval()
    train_metrics, train_confusion_meter =  evaluate(model, loss_fn, train_dataloader, metrics, params, logger)
    train_confusion_logger.log(train_confusion_meter.value())
    model.eval()
    val_metrics,test_confusion_meter = evaluate(model, loss_fn, val_dataloader, metrics, params, logger)
    test_confusion_logger.log(test_confusion_meter.value())
    diff_confusion_meter = train_confusion_meter.value()-test_confusion_meter.value()
    diff_confusion_logger.log(diff_confusion_meter)
    pass



if __name__ == '__main__':
    # Load the parameters from json file
    args = parser.parse_args()
    experiment_path =  os.path.join(args.model_dir,'experiments',args.dataset_name,args.model_name+args.num)
    if not os.path.isdir(experiment_path):
        os.makedirs(experiment_path)


    json_file = os.path.join(experiment_path,'params.json')
    if not os.path.isfile(json_file):
        with open(json_file,'w') as f:
            print("No json configuration file found at {}".format(json_file))
            f.close()
            print('successfully made file: {}'.format(json_file))

    params = utils.Params(json_file)

    if args.load :
        print("args.load=",args.load)
        if args.load_model:
            params.restore_file = args.load_model
        else:
            params.restore_file = experiment_path + '/checkpoint/best.pth.tar'


    params.dataset_dir = args.dataset_dir
    params.dataset_name = args.dataset_name
    params.model_version = args.model_name
    params.experiment_path = experiment_path
    params.mode = args.mode
    if params.gpu_id >= -1:
        params.cuda = True

    # Set the random seed for reproducible experiments
    torch.manual_seed(params.seed)
    np.random.seed(params.seed)
    random.seed(params.seed)
    if params.gpu_id >= -1:
        torch.cuda.manual_seed(params.seed)
    torch.backends.cudnn.deterministic = False # must be True to if you want reproducible,but will slow the speed

    cudnn.benchmark = True # https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936
    torch.cuda.empty_cache() # release cache
    # Set the logger
    if params.mode =='train':
        utils.set_logger(os.path.join(experiment_path,'train.log'))
    elif params.mode =='test':
        utils.set_logger(os.path.join(experiment_path, 'test.log'))
    elif params.mode == 'load_train':
        utils.set_logger(os.path.join(experiment_path, 'load_train.log'))

    logger = logging.getLogger()

    port,env = 8098,args.visdom_env
    columnnames,rownames = list(range(1,params.model_args["num_class"]+1)),list(range(1,params.model_args["num_class"]+1))
    loss_logger = VisdomPlotLogger('line',port=port,opts={'title': params.experiment_path + '_Loss','legend':['train','test']}, win=None,env=env)
    loss_logger_split = VisdomPlotLogger('line', port=port,
                                   opts={'title': params.experiment_path + '_Loss_split'},
                                   win=None, env=env)
    # error_logger = VisdomPlotLogger('line',port=port, opts={'title': params.experiment_path + '_Error @top1','legend':['train','test']},win=None,env=env)
    error_logger15 = VisdomPlotLogger('line', port=port, opts={'title': params.experiment_path + '_Error @top1@top5',
           'legend': ['train@top1','train@top5','test@top1','test@top5']}, win=None, env=env)
    train_confusion_logger = VisdomLogger('heatmap', port=port, opts={'title': params.experiment_path + 'train_Confusion matrix',
        'columnnames': columnnames,'rownames': rownames},win=None,env=env)
    test_confusion_logger = VisdomLogger('heatmap', port=port, opts={'title': params.experiment_path + 'test_Confusion matrix',
        'columnnames':columnnames,'rownames': rownames},win=None,env=env)
    # diff_confusion_logger = VisdomLogger('heatmap', port=port, opts={'title': params.experiment_path + 'diff_Confusion matrix',
    #     'columnnames':columnnames,'rownames': rownames},win=None,env=env)

    # log all params
    d_args = vars(args)
    for k in d_args.keys():
        logging.info('{0}: {1}'.format(k, d_args[k]))
    d_params = vars(params)
    for k in d_params.keys():
        logger.info('{0}: {1}'.format(k, d_params[k]))


    if 'HCN' in params.model_version:
        model = HCN.HCN(**params.model_args)
        grl = HCN.GRL()
        #3D Online 1536
        # N2B 2048
        ad_net = HCN.AdversarialNetwork(6144)
        grl_m = HCN.GRL()

        if params.data_parallel:
            model = torch.nn.DataParallel(model).cuda()
        else:
            model = model.cuda(params.gpu_id)
            #grl = grl.cuda(params.gpu_id)
            ad_net = ad_net.cuda(params.gpu_id)
            ad_net_m = ad_net_m.cuda(params.gpu_id)
            ad_net_t = ad_net_t.cuda(params.gpu_id)
            ad_net_mt = ad_net_mt.cuda(params.gpu_id)

        loss_fn = HCN.loss_fn
        metrics = HCN.metrics
    # elif # add other model

    #para_list = list(filter(lambda p: p.requires_grad, model.parameters()))

    #print(para_list)
    #para_list += list(filter(lambda p: p.requires_grad, ad_net.parameters()))
    #print(type(model.parameters()))
    para_dic = [{'params':filter(lambda p: p.requires_grad, model.parameters())},{'params':filter(lambda p: p.requires_grad, ad_net.parameters()),'lr':5e-3}
    ,{'params':filter(lambda p: p.requires_grad, ad_net_m.parameters()),'lr':5e-3},{'params':filter(lambda p: p.requires_grad, ad_net_t.parameters()),'lr':5e-3
    }
    ,{'params':filter(lambda p: p.requires_grad, ad_net_mt.parameters()),'lr':5e-3}]
    #print(para_list)
    #para_list = filter(para_list)

    if params.optimizer == 'Adam':

        optimizer = optim.Adam(para_dic, lr=params.lr, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=params.weight_decay)
        #print(optimizer)
    elif params.optimizer == 'SGD':
        optimizer = optim.SGD(para_dic, lr=params.lr, momentum=0.9,nesterov=True,weight_decay=params.weight_decay)

    logger.info(model)
    # Create the input data pipeline
    logger.info("Loading the datasets...")
    # fetch dataloaders
    train_dl = data_loader.fetch_dataloader('train', params)
    test_dl = data_loader.fetch_dataloader('test', params)
    logger.info("- done.")
    Myacc = []
    if params.mode == 'train' or params.mode == 'load_train':
        # Train the model
        logger.info("Starting training for {} epoch(s)".format(params.num_epochs))
        train_and_evaluate(model,ad_net, grl,ad_net_m,grl_m,ad_net_t, grl_t,ad_net_mt,grl_mt, Myacc ,train_dl, test_dl, optimizer, loss_fn, metrics, params,
                           args.model_dir,logger, params.restore_file)
    elif params.mode == 'test':
        test_only(model, train_dl,test_dl, optimizer,
                  loss_fn, metrics, params, args.model_dir,logger, params.restore_file)
    else:
        print('mode input error!')
    np.save('./acc.npy',np.array(Myacc))

