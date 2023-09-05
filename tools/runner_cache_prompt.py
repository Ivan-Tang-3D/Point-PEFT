import torch
import torch.nn as nn
from tools import builder
from utils import misc, dist_utils
import time
from utils.logger import *
from utils.AverageMeter import AverageMeter

import numpy as np
from datasets import data_transforms
from pointnet2_ops import pointnet2_utils
from torchvision import transforms

train_transforms = transforms.Compose(
    [
         # data_transforms.PointcloudScale(),
         #data_transforms.PointcloudRotate(),
         # data_transforms.PointcloudTranslate(),
         # data_transforms.PointcloudJitter(),
         # data_transforms.PointcloudRandomInputDropout(),
         # data_transforms.RandomHorizontalFlip(),
         data_transforms.PointcloudScaleAndTranslate(),
    ]
)

test_transforms = transforms.Compose(
    [
        # data_transforms.PointcloudScale(),
        # data_transforms.PointcloudRotate(),
        # data_transforms.PointcloudTranslate(),
        data_transforms.PointcloudScaleAndTranslate(),
    ]
)

train_transforms_scan = transforms.Compose(
    [
        data_transforms.PointcloudScaleAndTranslate(scale_low=0.9, scale_high=1.1, translate_range=0),
        data_transforms.PointcloudRotate(),
    ]
)

class Acc_Metric:
    def __init__(self, acc = 0.):
        if type(acc).__name__ == 'dict':
            self.acc = acc['acc']
        elif type(acc).__name__ == 'Acc_Metric':
            self.acc = acc.acc
        else:
            self.acc = acc

    def better_than(self, other):
        if self.acc > other.acc:
            return True
        else:
            return False

    def state_dict(self):
        _dict = dict()
        _dict['acc'] = self.acc
        return _dict

def run_net(args, config, config_cp, train_writer=None, val_writer=None):
    logger = get_logger(args.log_name)
    # build dataset
    (train_sampler, train_dataloader), (_, test_dataloader),= builder.dataset_builder(args, config.dataset.train), \
                                                            builder.dataset_builder(args, config.dataset.val)
    # build model
    base_model = builder.model_builder(config.model)
    #cp
    cp_model = builder.model_builder(config_cp.model)
    builder.load_model(cp_model, "./ckpts/pre-train.pth", logger = logger)
    cp_model.to(args.local_rank)
    cp_model.eval()
    cp_model = nn.DataParallel(cp_model).cuda()

    # parameter setting
    start_epoch = 0
    best_metrics = Acc_Metric(0.)
    best_metrics_vote = Acc_Metric(0.)
    metrics = Acc_Metric(0.)

    # resume ckpts
    if args.resume:
        start_epoch, best_metric = builder.resume_model(base_model, args, logger = logger)
        best_metrics = Acc_Metric(best_metrics)
    else:
        if args.ckpts is not None:
            base_model.load_model_from_ckpt(args.ckpts)
        else:
            print_log('Training from scratch', logger = logger)

    if args.use_gpu:    
        base_model.to(args.local_rank)
    # DDP
    if args.distributed:
        # Sync BN
        if args.sync_bn:
            base_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(base_model)
            print_log('Using Synchronized BatchNorm ...', logger = logger)
        base_model = nn.parallel.DistributedDataParallel(base_model, device_ids=[args.local_rank % torch.cuda.device_count()])
        print_log('Using Distributed Data parallel ...' , logger = logger)
    else:
        print_log('Using Data parallel ...' , logger = logger)
        base_model = nn.DataParallel(base_model).cuda()

     # only finetune cls head
    for name, param in base_model.named_parameters():
        if 'prompt_cor' in name or 'proj.bias' in name or 'fc2.bias' in name or 'fc1.bias' in name or 'norm2.bias' in name or 'norm1.bias' in name or 'LoRA_b_fc2' in name or 'LoRA_a_fc2' in name or 'LoRA_b_fc1' in name or 'LoRA_a_fc1' in name or 'LoRA_proj_b' in name or 'LoRA_proj_a' in name or 'prefix_tokens_key' in name or 'prefix_tokens_value' in name or 'LoRA_a' in name or 'LoRA_b' in name or 'attn.qkv.bias' in name or 'cls_head_' in name or "norm." in name or ".adapter" in name or "adapter1" in name or "out_transform" in name or "norm3" in name or "attn1." in name or "prompt_embeddings" in name or "cp" in name or ".gate" in name or "ad_gate" in name: 
            param.requires_grad_(True)
        else:
            param.requires_grad_(False)

    total_para_nums = 0
    adapter_para_nums = 0
    norm_para_nums = 0
    head_para_nums = 0
    out_transform_nums = 0
    norm_linear_nums = 0
    norm3_nums = 0
    attn1_nums = 0
    ln1_nums = 0
    cp_nums = 0
    ln2_nums = 0
    prompt_nums = 0
    qkv_bias_num = 0
    gate_nums = 0
    ad_gate_nums = 0
    lora_nums = 0
    prefix_num = 0
    bias_num = 0
    encoders_num = 0
    pos_embeds_num = 0
    for name,param in base_model.named_parameters():
        print(name,param.requires_grad)
        if param.requires_grad:
            total_para_nums += param.numel()
            if '.adapter' in name:
                adapter_para_nums += param.numel()
            elif 'adapter1' in name:
                adapter_para_nums += param.numel()
            elif 'norm.' in name:
                norm_para_nums += param.numel()
            elif 'cls_head_' in name:
                head_para_nums += param.numel()
            elif 'out_transform' in name:
                out_transform_nums += param.numel()
            elif 'norm_linear' in name:
                norm_linear_nums += param.numel()
            elif "norm3" in name:
                norm3_nums += param.numel()
            elif "attn1." in name:
                attn1_nums += param.numel()
            elif 'prompt_embeddings' in name or 'prompt_cor' in name:
                prompt_nums += param.numel()
            elif "cp" in name:
                cp_nums += param.numel()
            elif ".gate" in name:
                gate_nums += param.numel()
            elif "ad_gate" in name:
                ad_gate_nums += param.numel()
            elif 'attn.qkv.bias' in name:
                qkv_bias_num += param.numel()
            elif 'LoRA_a' in name:
                lora_nums += param.numel()
            elif 'LoRA_b' in name:
                lora_nums += param.numel()
            elif 'prefix_tokens_key' in name:
                prefix_num += param.numel()
            elif 'prefix_tokens_value' in name:
                prefix_num += param.numel()
            elif 'LoRA_proj_a' in name:
                lora_nums += param.numel()
            elif 'LoRA_proj_b' in name:
                lora_nums += param.numel()
            elif  'LoRA_b_fc2' in name or 'LoRA_a_fc2' in name or 'LoRA_b_fc1' in name or 'LoRA_a_fc1' in name:
                lora_nums += param.numel()
            elif 'proj.bias' in name or 'fc2.bias' in name or 'fc1.bias' in name or 'norm2.bias' in name or 'norm1.bias' in name:
                bias_num += param.numel()
            elif '0.bias' in name or '1.bias' in name or '2.bias' in name or '3.bias' in name:
                bias_num += param.numel()
            elif 'encoders' in name:
                encoders_num += param.numel()#pos_embeds
         

    print('parameters:',total_para_nums,'adapter',adapter_para_nums,'norm',norm_para_nums,'cls_head',head_para_nums, 'out_transform',out_transform_nums,'norm_linear',norm_linear_nums, "norm3", norm3_nums,"attn1_nums", attn1_nums,
          "ln1_nums", ln1_nums,"ln2_nums", ln2_nums, "prompt", prompt_nums, "cp", cp_nums, "gate", gate_nums, "ad_gate", ad_gate_nums, 'attn.qkv.bias', qkv_bias_num, 'lora', lora_nums, 'prefix', prefix_num, 'bias', bias_num,
          "encoders", encoders_num, "pos_embeds", pos_embeds_num)
    for name,param in cp_model.named_parameters():
        param.requires_grad_(False)
        print(name,param.requires_grad)
    # optimizer & scheduler
    optimizer, scheduler = builder.build_opti_sche(base_model, config)
    
    if args.resume:
        builder.resume_optimizer(optimizer, args, logger = logger)

    # trainval
    # training
    base_model.zero_grad()

    for epoch in range(start_epoch, config.max_epoch + 1):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        base_model.train()

        epoch_start_time = time.time()
        batch_start_time = time.time()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter(['loss', 'acc'])
        num_iter = 0
        base_model.train()  # set model to training mode
        n_batches = len(train_dataloader)

        npoints = config.npoints
        for idx, (taxonomy_ids, model_ids, data) in enumerate(train_dataloader):
            num_iter += 1
            n_itr = epoch * n_batches + idx
            
            data_time.update(time.time() - batch_start_time)
            
            points = data[0].cuda()
            label = data[1].cuda()

            if npoints == 1024:
                point_all = 1200
            elif npoints == 2048:
                point_all = 2400
            elif npoints == 4096:
                point_all = 4800
            elif npoints == 8192:
                point_all = 8192
            else:
                raise NotImplementedError()

            if points.size(1) < point_all:
                point_all = points.size(1)

            fps_idx = pointnet2_utils.furthest_point_sample(points, point_all)  # (B, npoint)
            fps_idx = fps_idx[:, np.random.choice(point_all, npoints, False)]
            points = pointnet2_utils.gather_operation(points.transpose(1, 2).contiguous(), fps_idx).transpose(1, 2).contiguous()  # (B, N, 3)
            #import pdb; pdb.set_trace()
            if config.model['NAME'] == 'PointTransformer_best':
                points = train_transforms_scan(points)
            else:
                points = train_transforms(points)#data_aug
            #points = train_transforms(points)
            batch_forward_start_cp_time = time.time()
            cp_feat = cp_model(points, eval=True)
            batch_forward_end_cp_time = time.time()
            #cp_feat = None
            batch_forward_start_base_time = time.time()
            ret = base_model(points, cp_feat=cp_feat, args=args)
            
            loss, acc = base_model.module.get_loss_acc(ret, label)
            batch_forward_end_base_time = time.time()
            _loss = loss
            batch_backward_start_time = time.time()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            #batch_backward_loss_start_time = time.time()
            _loss.backward()
            end_event.record()
            torch.cuda.synchronize()
            elapsed_time = start_event.elapsed_time(end_event)/1000
            #batch_backward_loss_end_time = time.time()
            # forward
            if num_iter == config.step_per_update:
                if config.get('grad_norm_clip') is not None:
                    torch.nn.utils.clip_grad_norm_(base_model.parameters(), config.grad_norm_clip, norm_type=2)
                num_iter = 0
                batch_backward_optimizere_start_time = time.time()
                optimizer.step()
                batch_backward_optimizer_end_time = time.time()
                batch_backward_zero_start_time = time.time()
                base_model.zero_grad()
                batch_backward_zero_end_time = time.time()

            if args.distributed:
                loss = dist_utils.reduce_tensor(loss, args)
                acc = dist_utils.reduce_tensor(acc, args)
                losses.update([loss.item(), acc.item()])
            else:
                losses.update([loss.item(), acc.item()])


            if args.distributed:
                torch.cuda.synchronize()
            batch_backward_end_time = time.time()

            if train_writer is not None:
                train_writer.add_scalar('Loss/Batch/Loss', loss.item(), n_itr)
                train_writer.add_scalar('Loss/Batch/TrainAcc', acc.item(), n_itr)
                train_writer.add_scalar('Loss/Batch/LR', optimizer.param_groups[0]['lr'], n_itr)


            batch_time.update(time.time() - batch_start_time)
            batch_start_time = time.time()

            # if idx % 10 == 0:#batch_backward_loss_end_time-batch_backward_loss_start_time
            #     print_log('[Epoch %d/%d][Batch %d/%d] Batch_forward_cp_Time = %.3f Batch_forward_base_Time = %.3f Batch_backward_loss_Time = %.3f Batch_backward_op_Time = %.3f Batch_backward_zero_Time = %.3f Batch_backward_Time = %.3f BatchTime = %.3f (s) DataTime = %.3f (s) Loss+Acc = %s lr = %.6f' %
            #                 (epoch, config.max_epoch, idx + 1, n_batches,batch_forward_end_cp_time-batch_forward_start_cp_time,batch_forward_end_base_time-batch_forward_start_base_time,elapsed_time,batch_backward_optimizer_end_time-batch_backward_optimizere_start_time,batch_backward_zero_end_time-batch_backward_zero_start_time,batch_backward_end_time-batch_backward_start_time, batch_time.val(), data_time.val(),
            #                 ['%.4f' % l for l in losses.val()], optimizer.param_groups[0]['lr']), logger = logger)
        batch_backward_step_start_time = time.time()    
        if isinstance(scheduler, list):
            for item in scheduler:
                item.step(epoch)
        else:
            scheduler.step(epoch)
        epoch_end_time = time.time()
        batch_backward_step_end_time = time.time()
        if train_writer is not None:
            train_writer.add_scalar('Loss/Epoch/Loss', losses.avg(0), epoch)

        print_log('[Training] EPOCH: %d Epoch_back_Time = %.3f EpochTime = %.3f (s) Losses = %s lr = %.6f' %
            (epoch,batch_backward_step_end_time-batch_backward_step_start_time,  epoch_end_time - epoch_start_time, ['%.4f' % l for l in losses.avg()],optimizer.param_groups[0]['lr']), logger = logger)

        if epoch % args.val_freq == 0 and epoch != 0:
            # Validate the current model
            metrics = validate(base_model, cp_model, test_dataloader, epoch, val_writer, args, config, logger=logger)

            better = metrics.better_than(best_metrics)
            # Save ckeckpoints
            if better:
                best_metrics = metrics
                builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-best', args, logger = logger)
                print_log("--------------------------------------------------------------------------------------------", logger=logger)
            if args.vote:
                if metrics.acc > 92.1 or (better and metrics.acc > 91):
                    metrics_vote = validate_vote(base_model, test_dataloader, epoch, val_writer, args, config, logger=logger)
                    if metrics_vote.better_than(best_metrics_vote):
                        best_metrics_vote = metrics_vote
                        print_log(
                            "****************************************************************************************",
                            logger=logger)
                        builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics_vote, 'ckpt-best_vote', args, logger = logger)

        builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-last', args, logger = logger)      
        # if (config.max_epoch - epoch) < 10:
        #     builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, f'ckpt-epoch-{epoch:03d}', args, logger = logger)
    if train_writer is not None:
        train_writer.close()
    if val_writer is not None:
        val_writer.close()

def validate(base_model, cp_model, test_dataloader, epoch, val_writer, args, config, logger = None):
    # print_log(f"[VALIDATION] Start validating epoch {epoch}", logger = logger)
    base_model.eval()  # set model to eval mode

    test_pred  = []
    test_label = []
    npoints = config.npoints
    cp_model.eval()
    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            points = data[0].cuda()
            label = data[1].cuda()

            points,_ = misc.fps(points, npoints)

            cp_feat = cp_model(points, eval=True)
            logits = base_model(points, cp_feat=cp_feat, args=args)
            target = label.view(-1)

            pred = logits.argmax(-1).view(-1)

            test_pred.append(pred.detach())
            test_label.append(target.detach())

        test_pred = torch.cat(test_pred, dim=0)
        test_label = torch.cat(test_label, dim=0)

        if args.distributed:
            test_pred = dist_utils.gather_tensor(test_pred, args)
            test_label = dist_utils.gather_tensor(test_label, args)

        acc = (test_pred == test_label).sum() / float(test_label.size(0)) * 100.
        print_log('[Validation] EPOCH: %d  acc = %.4f' % (epoch, acc), logger=logger)

        if args.distributed:
            torch.cuda.synchronize()

    # Add testing results to TensorBoard
    if val_writer is not None:
        val_writer.add_scalar('Metric/ACC', acc, epoch)

    return Acc_Metric(acc)


def validate_vote(base_model, test_dataloader, epoch, val_writer, args, config, logger = None, times = 10):
    print_log(f"[VALIDATION_VOTE] epoch {epoch}", logger = logger)
    base_model.eval()  # set model to eval mode

    test_pred  = []
    test_label = []
    npoints = config.npoints
    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            points_raw = data[0].cuda()
            label = data[1].cuda()
            if npoints == 1024:
                point_all = 1200
            elif npoints == 4096:
                point_all = 4800
            elif npoints == 8192:
                point_all = 8192
            else:
                raise NotImplementedError()
                
            if points_raw.size(1) < point_all:
                point_all = points_raw.size(1)

            fps_idx_raw = pointnet2_utils.furthest_point_sample(points_raw, point_all)  # (B, npoint)
            local_pred = []

            for kk in range(times):
                fps_idx = fps_idx_raw[:, np.random.choice(point_all, npoints, False)]
                points = pointnet2_utils.gather_operation(points_raw.transpose(1, 2).contiguous(), 
                                                        fps_idx).transpose(1, 2).contiguous()  # (B, N, 3)

                points = test_transforms(points)

                logits = base_model(points)
                target = label.view(-1)

                local_pred.append(logits.detach().unsqueeze(0))

            pred = torch.cat(local_pred, dim=0).mean(0)
            _, pred_choice = torch.max(pred, -1)


            test_pred.append(pred_choice)
            test_label.append(target.detach())

        test_pred = torch.cat(test_pred, dim=0)
        test_label = torch.cat(test_label, dim=0)

        if args.distributed:
            test_pred = dist_utils.gather_tensor(test_pred, args)
            test_label = dist_utils.gather_tensor(test_label, args)

        acc = (test_pred == test_label).sum() / float(test_label.size(0)) * 100.
        print_log('[Validation_vote] EPOCH: %d  acc_vote = %.4f' % (epoch, acc), logger=logger)

        if args.distributed:
            torch.cuda.synchronize()

    # Add testing results to TensorBoard
    if val_writer is not None:
        val_writer.add_scalar('Metric/ACC_vote', acc, epoch)

    return Acc_Metric(acc)



def test_net(args, config, config_cp, train_writer=None, val_writer=None):
    logger = get_logger(args.log_name)
    print_log('Tester start ... ', logger = logger)
    _, test_dataloader = builder.dataset_builder(args, config.dataset.test)
    base_model = builder.model_builder(config.model)
    # load checkpoints
    # builder.load_model(base_model, args.ckpts, logger = logger) # for finetuned transformer
    base_model.load_model_from_ckpt(args.ckpts) # for BERT
    if args.use_gpu:
        base_model.to(args.local_rank)
    #  DDP    
    if args.distributed:
        raise NotImplementedError()
     
    test(base_model, test_dataloader, args, config, logger=logger, config_cp=config_cp)
    
def test(base_model, test_dataloader, args, config, logger = None, config_cp = None):

    base_model.eval()  # set model to eval mode

    test_pred  = []
    test_label = []
    npoints = config.npoints

    cp_model = builder.model_builder(config_cp.model)
    builder.load_model(cp_model, "./ckpts/pre-train.pth", logger = logger)
    cp_model.to(args.local_rank)
    cp_model.eval()
    cp_model = nn.DataParallel(cp_model).cuda()

    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            points = data[0].cuda()
            label = data[1].cuda()

            points,_ = misc.fps(points, npoints)

            cp_feat = cp_model(points, eval=True)
   
            logits = base_model(points, cp_feat=cp_feat, args=args)
            target = label.view(-1)

            pred = logits.argmax(-1).view(-1)

            test_pred.append(pred.detach())
            test_label.append(target.detach())

        test_pred = torch.cat(test_pred, dim=0)
        test_label = torch.cat(test_label, dim=0)

        if args.distributed:
            test_pred = dist_utils.gather_tensor(test_pred, args)
            test_label = dist_utils.gather_tensor(test_label, args)

        acc = (test_pred == test_label).sum() / float(test_label.size(0)) * 100.
        print_log('[TEST] acc = %.4f' % acc, logger=logger)
        
        if args.vote == False:
            return
        if args.distributed:
            torch.cuda.synchronize()

        print_log(f"[TEST_VOTE]", logger = logger)
        acc = 0.
        for time in range(1, 300):
            this_acc = test_vote(base_model, test_dataloader, 1, None, args, config, logger=logger, times=10)
            if acc < this_acc:
                acc = this_acc
            print_log('[TEST_VOTE_time %d]  acc = %.4f, best acc = %.4f' % (time, this_acc, acc), logger=logger)
        print_log('[TEST_VOTE] acc = %.4f' % acc, logger=logger)

def test_vote(base_model, test_dataloader, epoch, val_writer, args, config, logger = None, times = 10):

    base_model.eval()  # set model to eval mode

    test_pred  = []
    test_label = []
    npoints = config.npoints
    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            points_raw = data[0].cuda()
            label = data[1].cuda()
            if npoints == 1024:
                point_all = 1200
            elif npoints == 4096:
                point_all = 4800
            elif npoints == 8192:
                point_all = 8192
            else:
                raise NotImplementedError()
                
            if points_raw.size(1) < point_all:
                point_all = points_raw.size(1)

            fps_idx_raw = pointnet2_utils.furthest_point_sample(points_raw, point_all)  # (B, npoint)
            local_pred = []

            for kk in range(times):
                fps_idx = fps_idx_raw[:, np.random.choice(point_all, npoints, False)]
                points = pointnet2_utils.gather_operation(points_raw.transpose(1, 2).contiguous(), 
                                                        fps_idx).transpose(1, 2).contiguous()  # (B, N, 3)

                points = test_transforms(points)

                logits = base_model(points)
                target = label.view(-1)

                local_pred.append(logits.detach().unsqueeze(0))

            pred = torch.cat(local_pred, dim=0).mean(0)
            _, pred_choice = torch.max(pred, -1)


            test_pred.append(pred_choice)
            test_label.append(target.detach())

        test_pred = torch.cat(test_pred, dim=0)
        test_label = torch.cat(test_label, dim=0)

        if args.distributed:
            test_pred = dist_utils.gather_tensor(test_pred, args)
            test_label = dist_utils.gather_tensor(test_label, args)

        acc = (test_pred == test_label).sum() / float(test_label.size(0)) * 100.

        if args.distributed:
            torch.cuda.synchronize()

    # Add testing results to TensorBoard
    if val_writer is not None:
        val_writer.add_scalar('Metric/ACC_vote', acc, epoch)
    # print_log('[TEST] acc = %.4f' % acc, logger=logger)
    
    return acc
