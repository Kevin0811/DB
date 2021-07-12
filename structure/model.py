import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchsummary import summary

import backbones
import decoders


class BasicModel(nn.Module):
    def __init__(self, args):
        nn.Module.__init__(self)

        print("Backbone: " + str(args['backbone']))

        if str(args['backbone']) == "DETR":
            self.transpose = backbones.get_pose_net(50, True) # 50: ResNet50
        else:
            self.backbone = getattr(backbones, args['backbone'])(**args.get('backbone_args', {}))
            self.decoder = getattr(decoders, args['decoder'])(**args.get('decoder_args', {}))

        #self.transpose = backbones.get_pose_net(50, True) # 50: ResNet50

        #print(self.transpose)
        if (args['backbone']) == "DETR":
            try:
                summary(self.transpose, (3,480,480), batch_size=-1, device="cpu") #image size
            except:
                print("skip summary")
                #summary(self.transpose, (3,480,480), batch_size=-1, device="cuda") #image size
        else:
            print("skip summary")

    def forward(self, data,backbone_type, *args, **kwargs):        
        #print(**kwargs)
        #self.backbone_type = kwargs.get('backbone_type',"DETR")

        if backbone_type == "DETR":
            return self.transpose(data)

        return self.decoder(self.backbone(data), *args, **kwargs)


def parallelize(model, distributed, local_rank):
    if distributed:
        # data for multi GPU
        return nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=[local_rank],
            find_unused_parameters=True)
    else:
        return nn.DataParallel(model)

class SegDetectorModel(nn.Module):
    def __init__(self, args, device, distributed: bool = False, local_rank: int = 0):
        super(SegDetectorModel, self).__init__()
        from decoders.seg_detector_loss import SegDetectorLossBuilder

        self.model = BasicModel(args)
        # for loading models
        self.model = parallelize(self.model, distributed, local_rank)
        # set Loss function (criterion)
        self.criterion = SegDetectorLossBuilder(
            args['loss_class'], *args.get('loss_args', []), **args.get('loss_kwargs', {})).build()
        self.criterion = parallelize(self.criterion, distributed, local_rank)
        self.device = device
        self.to(self.device)

    @staticmethod
    def model_name(args):
        return os.path.join('seg_detector', args['backbone'], args['loss_class'])

    def forward(self, batch, training=True, backbone_type='DETR'):
        if isinstance(batch, dict):
            data = batch['image'].to(self.device)
        else:
            data = batch.to(self.device)
        data = data.float()
        pred = self.model(data, backbone_type, training=self.training)

        if self.training:
            for key, value in batch.items():
                if value is not None:
                    if hasattr(value, 'to'):
                        batch[key] = value.to(self.device)
            #print(pred)
            loss_with_metrics = self.criterion(pred, batch)
            loss, metrics = loss_with_metrics
            return loss, pred, metrics
        return pred