
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import utils


class Scan2SimWithImageLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.labels = None
        self.last_local_batch_size = None

    def forward(self, outputs):
        pc_embed = outputs['pc_embed']
        text_embed = outputs['text_embed']
        image_embed = outputs['image_embed']
        logit_scale = outputs['logit_scale']
        local_batch_size = pc_embed.size(0)

        if local_batch_size != self.last_local_batch_size:
            self.labels = local_batch_size * utils.get_rank() + torch.arange(
                local_batch_size, device=pc_embed.device
            )
            self.last_local_batch_size = local_batch_size

        # normalized features
        pc_embed = F.normalize(pc_embed, dim=-1, p=2)
        text_embed = F.normalize(text_embed, dim=-1, p=2)
        image_embed = F.normalize(image_embed, dim=-1, p=2)

        # gather features from all GPUs
        pc_embed_all, text_embed_all, image_embed_all = \
            utils.all_gather_batch([pc_embed, text_embed, image_embed])

        # cosine similarity as logits
        logits_per_pc_text = logit_scale * pc_embed @ text_embed_all.t()
        logits_per_text_pc = logit_scale * text_embed @ pc_embed_all.t()
        logits_per_pc_image = logit_scale * pc_embed @ image_embed_all.t()
        logits_per_image_pc = logit_scale * image_embed @ pc_embed_all.t()

        loss = (F.cross_entropy(logits_per_pc_text, self.labels) + \
                F.cross_entropy(logits_per_text_pc, self.labels)) / 2 + \
                (F.cross_entropy(logits_per_pc_image, self.labels) + F.cross_entropy(logits_per_image_pc, self.labels)) / 2

        # compute accuracy
        with torch.no_grad():
            pred = torch.argmax(logits_per_pc_text, dim=-1)
            correct = pred.eq(self.labels).sum()
            pc_text_acc = 100 * correct / local_batch_size

            pred = torch.argmax(logits_per_pc_image, dim=-1)
            correct = pred.eq(self.labels).sum()
            pc_image_acc = 100 * correct / local_batch_size

        return {'loss': loss, 'scan2sim_loss': loss, 'scan2sim_pc_image_acc': pc_image_acc, 'scan2sim_pc_text_acc': pc_text_acc}

# todo new loss
class Scan2SimWithGroupLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.labels = None
        self.last_local_batch_size = None
        self.margin = 1.0



    def forward(self, outputs):
        pc_image_embed = outputs['pc_image_embed']
        pc_text_embed = outputs['pc_text_embed']
        pc_cls = outputs['pc_cls']
        text_embed = outputs['text_embed']
        image_embed = outputs['image_embed']
        zero_images_mask = (1 - outputs['zero_images_mask']).cuda()
        # logit_scale = outputs['logit_scale']
        cls_gt_one_hot = outputs['cls_gt'].cuda()



        # normalized features
        text_embed = F.normalize(text_embed, dim=-1, p=2)
        image_embed = F.normalize(image_embed, dim=-1, p=2)

        # gather features from all GPUs
        pc_image_embed_all, pc_text_embed_all, text_embed_all, image_embed_all, pc_cls_all, zero_images_mask, cls_gt_one_hot  = \
            utils.all_gather_batch_with_grad([pc_image_embed, pc_text_embed, text_embed, image_embed, pc_cls, zero_images_mask, cls_gt_one_hot])

        local_batch_size = image_embed_all.size(0)

        # unpack features
        group_cnt = int(pc_image_embed_all.size(0) / local_batch_size)
        pc_image_embed_all_group = pc_image_embed_all.reshape(local_batch_size, group_cnt, pc_image_embed_all.size(-1))
        pc_image_embed_all_group = F.normalize(pc_image_embed_all_group, dim=-1, p=2)


        pc_text_embed_all_group = pc_text_embed_all.reshape(local_batch_size, group_cnt, pc_text_embed_all.size(-1))
        pc_text_embed_all_group = F.normalize(pc_text_embed_all_group, dim=-1, p=2)


        pc_cls_group = pc_cls_all.reshape(local_batch_size, group_cnt, pc_cls_all.size(-1))
        pc_cls_group = pc_cls_group.squeeze()


        logits_per_image_pc_group = torch.einsum('ijk, ik->ij', [pc_image_embed_all_group, image_embed_all])
        logits_per_text_pc_group = torch.einsum('ijk, ik->ij', [pc_image_embed_all_group, text_embed_all]) # different input, only test
        logits_per_text_pc_random = torch.einsum('ijk, ik->ij', [pc_text_embed_all_group, text_embed_all])
        logits_per_image_pc_random = torch.einsum('ijk, ik->ij', [pc_text_embed_all_group, image_embed_all])
        cls_gt = torch.argmax(cls_gt_one_hot, dim=1)



        mask = zero_images_mask.expand_as(logits_per_image_pc_group)

        logits_per_image_pc_group = logits_per_image_pc_group * mask
        logits_per_image_pc_random = logits_per_image_pc_random * mask


        loss_scan2sim_add = F.cross_entropy(pc_cls_group + logits_per_image_pc_group + logits_per_text_pc_group, cls_gt)
        loss_scan2sim_image_text = F.cross_entropy(logits_per_text_pc_random + logits_per_image_pc_random , cls_gt)

        # add all loss
        loss = loss_scan2sim_add + loss_scan2sim_image_text # + loss_scan2sim_image + loss_scan2sim_text

        # compute accuracy
        with torch.no_grad():
            logits_all = pc_cls_group + logits_per_image_pc_group + logits_per_text_pc_group
            _, predicted_classes = torch.max(logits_all, dim=1)  # [batch_size]
            correct = (predicted_classes == cls_gt).float()  # [batch_size]
            scan2sim_acc_1 = correct.mean()

            # _, predicted_classes = torch.max(logits_per_image_pc, dim=1)  # [batch_size]
            # correct = (predicted_classes == cls_gt).float()  # [batch_size]
            # scan2sim_acc_image = correct.mean()
            #
            # _, predicted_classes = torch.max(logits_per_text_pc, dim=1)  # [batch_size]
            # correct = (predicted_classes == cls_gt).float()  # [batch_size]
            # scan2sim_acc_text = correct.mean()
            #
            # _, predicted_classes = torch.max(pc_cls_group, dim=1)  # [batch_size]
            # correct = (predicted_classes == cls_gt).float()  # [batch_size]
            # acc_quality = correct.mean()


        return {'loss': loss, 'scan2sim_loss_image': loss_scan2sim_image_text, 'scan2sim_loss_text': loss_scan2sim_image_text,
                'scan2sim_pc_image_acc': scan2sim_acc_1, 'scan2sim_pc_text_acc': scan2sim_acc_1, 'pc_quality_acc': scan2sim_acc_1}