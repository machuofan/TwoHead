from data.data_pipe import de_preprocess, get_train_loader, get_val_data
from model import Backbone, PreheadID, Arcface, Attrhead
from verification import evaluate
import torch
from torch import optim
import numpy as np
from tqdm import tqdm
from tensorboardX import SummaryWriter
from matplotlib import pyplot as plt
plt.switch_backend('agg')
from utils import get_time, gen_plot, hflip_batch, separate_bn_paras
from PIL import Image
from torchvision import transforms as trans
import math
import bcolz
from celebA import CelebA
from torch.utils.data import Dataset, ConcatDataset, DataLoader
from pathlib import Path

class face_learner(object):
    def __init__(self, conf, inference=False):
        self.backbone = Backbone().to(conf.device)
        self.idprehead = PreheadID().to(conf.device)
        self.idhead = Arcface().to(conf.device)
        self.attrhead = Attrhead().to(conf.device)
        print('model generated'.format(conf.net_mode, conf.net_depth))
        
        if not inference:
            self.milestones = conf.milestones
            train_dataset = CelebA(
                'dataset',
                'celebA_train.txt',
                trans.Compose([
                    trans.RandomHorizontalFlip(),
                    trans.ToTensor(),
                    trans.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                ]))
            valid_dataset = CelebA(
                'dataset',
                'celebA_validation.txt',
                trans.Compose([
                    trans.RandomHorizontalFlip(),
                    trans.ToTensor(),
                    trans.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                ]))
            self.loader = DataLoader(train_dataset, batch_size=conf.batch_size, shuffle=True, 
                                        pin_memory=conf.pin_memory, num_workers=conf.num_workers)
            self.valid_loader = DataLoader(valid_dataset, batch_size=conf.batch_size, shuffle=True, 
                                        pin_memory=conf.pin_memory, num_workers=conf.num_workers)    

            self.writer = SummaryWriter(conf.log_path)
            self.step = 0

            paras_only_bn_1, paras_wo_bn_1 = separate_bn_paras(self.backbone)
            paras_only_bn_2, paras_wo_bn_2 = separate_bn_paras(self.idprehead)
            paras_only_bn_3, paras_wo_bn_3 = separate_bn_paras(self.attrhead)
            paras_only_bn = paras_only_bn_1 + paras_only_bn_2 + paras_only_bn_3
            paras_wo_bn = paras_wo_bn_1 + paras_wo_bn_2 + paras_wo_bn_3
            
            self.optimizer = optim.SGD([
                                {'params': paras_wo_bn + [self.idhead.kernel], 'weight_decay': 1e-4},
                                {'params': paras_only_bn}
                            ], lr = conf.lr, momentum = conf.momentum)
#             self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=40, verbose=True)

            print('optimizers generated')    
            self.board_loss_every = len(self.loader)//8
            self.evaluate_every = len(self.loader)//4
            self.save_every = len(self.loader)//2
            self.agedb_30, self.cfp_fp, self.lfw, self.agedb_30_issame, self.cfp_fp_issame, self.lfw_issame = get_val_data(Path("data/faces_emore/"))
        else:
            self.threshold = conf.threshold
    
    def save_state(self, conf, accuracy, to_save_folder=False, extra=None, model_only=False):
        if to_save_folder:
            save_path = conf.save_path
        else:
            save_path = conf.model_path
        torch.save(
            self.backbone.state_dict(), save_path /
            ('backbone_{}_accuracy:{}_step:{}_{}.pth'.format(get_time(), accuracy, self.step, extra)))
        torch.save(
            self.idprehead.state_dict(), save_path /
            ('idprehead_{}_accuracy:{}_step:{}_{}.pth'.format(get_time(), accuracy, self.step, extra)))
        if not model_only:
            torch.save(
                self.idhead.state_dict(), save_path /
                ('idhead_{}_accuracy:{}_step:{}_{}.pth'.format(get_time(), accuracy, self.step, extra)))
            torch.save(
                self.attrhead.state_dict(), save_path /
                ('attrhead_{}_accuracy:{}_step:{}_{}.pth'.format(get_time(), accuracy, self.step, extra)))
            torch.save(
                self.optimizer.state_dict(), save_path /
                ('optimizer_{}_accuracy:{}_step:{}_{}.pth'.format(get_time(), accuracy, self.step, extra)))
    
    # def load_state(self, conf, fixed_str, from_save_folder=False, model_only=False):
    #     if from_save_folder:
    #         save_path = conf.save_path
    #     else:
    #         save_path = conf.model_path            
    #     self.model.load_state_dict(torch.load(save_path/'model_{}'.format(fixed_str)))
    #     if not model_only:
    #         self.head.load_state_dict(torch.load(save_path/'head_{}'.format(fixed_str)))
    #         self.optimizer.load_state_dict(torch.load(save_path/'optimizer_{}'.format(fixed_str)))
        
    def board_val(self, db_name, accuracy, best_threshold, roc_curve_tensor):
        self.writer.add_scalar('{}_accuracy'.format(db_name), accuracy, self.step)
        self.writer.add_scalar('{}_best_threshold'.format(db_name), best_threshold, self.step)
        self.writer.add_image('{}_roc_curve'.format(db_name), roc_curve_tensor, self.step)
#         self.writer.add_scalar('{}_val:true accept ratio'.format(db_name), val, self.step)
#         self.writer.add_scalar('{}_val_std'.format(db_name), val_std, self.step)
#         self.writer.add_scalar('{}_far:False Acceptance Ratio'.format(db_name), far, self.step)
        
    def evaluate(self, conf, carray, issame, nrof_folds = 5, tta = False):
        self.backbone.eval()
        self.idprehead.eval()
        idx = 0
        embeddings = np.zeros([len(carray), conf.embedding_size])
        with torch.no_grad():
            while idx + conf.batch_size <= len(carray):
                batch = torch.tensor(carray[idx:idx + conf.batch_size])
                if tta:
                    fliped = hflip_batch(batch)
                    emb_batch = self.idprehead(self.backbone(batch.to(conf.device))) + self.idprehead(self.backbone(fliped.to(conf.device)))
                    embeddings[idx:idx + conf.batch_size] = l2_norm(emb_batch)
                else:
                    embeddings[idx:idx + conf.batch_size] = self.idprehead(self.backbone(batch.to(conf.device))).cpu()
                idx += conf.batch_size
            if idx < len(carray):
                batch = torch.tensor(carray[idx:])            
                if tta:
                    fliped = hflip_batch(batch)
                    emb_batch = self.idprehead(self.backbone(batch.to(conf.device))) + self.idprehead(self.backbone(fliped.to(conf.device)))
                    embeddings[idx:] = l2_norm(emb_batch)
                else:
                    embeddings[idx:] = self.idprehead(self.backbone(batch.to(conf.device))).cpu()
        tpr, fpr, accuracy, best_thresholds = evaluate(embeddings, issame, nrof_folds)
        buf = gen_plot(fpr, tpr)
        roc_curve = Image.open(buf)
        roc_curve_tensor = trans.ToTensor()(roc_curve)
        return accuracy.mean(), best_thresholds.mean(), roc_curve_tensor
    
    # def find_lr(self,
    #             conf,
    #             init_value=1e-8,
    #             final_value=10.,
    #             beta=0.98,
    #             bloding_scale=3.,
    #             num=None):
    #     if not num:
    #         num = len(self.loader)
    #     mult = (final_value / init_value)**(1 / num)
    #     lr = init_value
    #     for params in self.optimizer.param_groups:
    #         params['lr'] = lr
    #     self.model.train()
    #     avg_loss = 0.
    #     best_loss = 0.
    #     batch_num = 0
    #     losses = []
    #     log_lrs = []
    #     for i, (imgs, labels) in tqdm(enumerate(self.loader), total=num):

    #         imgs = imgs.to(conf.device)
    #         labels = labels.to(conf.device)
    #         batch_num += 1          

    #         self.optimizer.zero_grad()

    #         embeddings = self.model(imgs)
    #         thetas = self.head(embeddings, labels)
    #         loss = conf.ce_loss(thetas, labels)          
          
    #         #Compute the smoothed loss
    #         avg_loss = beta * avg_loss + (1 - beta) * loss.item()
    #         self.writer.add_scalar('avg_loss', avg_loss, batch_num)
    #         smoothed_loss = avg_loss / (1 - beta**batch_num)
    #         self.writer.add_scalar('smoothed_loss', smoothed_loss,batch_num)
    #         #Stop if the loss is exploding
    #         if batch_num > 1 and smoothed_loss > bloding_scale * best_loss:
    #             print('exited with best_loss at {}'.format(best_loss))
    #             plt.plot(log_lrs[10:-5], losses[10:-5])
    #             return log_lrs, losses
    #         #Record the best loss
    #         if smoothed_loss < best_loss or batch_num == 1:
    #             best_loss = smoothed_loss
    #         #Store the values
    #         losses.append(smoothed_loss)
    #         log_lrs.append(math.log10(lr))
    #         self.writer.add_scalar('log_lr', math.log10(lr), batch_num)
    #         #Do the SGD step
    #         #Update the lr for the next step

    #         loss.backward()
    #         self.optimizer.step()

    #         lr *= mult
    #         for params in self.optimizer.param_groups:
    #             params['lr'] = lr
    #         if batch_num > num:
    #             plt.plot(log_lrs[10:-5], losses[10:-5])
    #             return log_lrs, losses    

    def train(self, conf, epochs):
        self.backbone.train()
        self.idprehead.train()
        self.attrhead.train()
        running_loss = 0.            
        for e in range(epochs):
            print('epoch {} started'.format(e))
            if e == self.milestones[0]:
                self.schedule_lr()
            if e == self.milestones[1]:
                self.schedule_lr()      
            if e == self.milestones[2]:
                self.schedule_lr()                                 
            for imgs, labels in tqdm(iter(self.loader)):
                imgs = imgs.to(conf.device)
                labels = labels.to(conf.device)
                attributes = labels[:, :40]
                attributes = (attributes + 1) * 0.5
                ids = labels[:, 40]
                self.optimizer.zero_grad()
                embeddings = self.backbone(imgs)
                thetas = self.idhead(self.idprehead(embeddings), ids)
                # attrs = self.attrhead(embeddings)
                # attributes = attributes.type_as(attrs)
                loss = conf.ce_loss(thetas, ids)
                loss.backward()
                running_loss += loss.item()
                self.optimizer.step()
                
                if self.step % self.board_loss_every == 0 and self.step != 0:
                    loss_board = running_loss / self.board_loss_every
                    self.writer.add_scalar('train_loss', loss_board, self.step)
                    running_loss = 0.
                
                if self.step % self.evaluate_every == 0 and self.step != 0:
                    accuracy, best_threshold, roc_curve_tensor = self.evaluate(conf, self.agedb_30, self.agedb_30_issame)
                    self.board_val('agedb_30', accuracy, best_threshold, roc_curve_tensor)
                    accuracy, best_threshold, roc_curve_tensor = self.evaluate(conf, self.lfw, self.lfw_issame)
                    self.board_val('lfw', accuracy, best_threshold, roc_curve_tensor)
                    accuracy, best_threshold, roc_curve_tensor = self.evaluate(conf, self.cfp_fp, self.cfp_fp_issame)
                    self.board_val('cfp_fp', accuracy, best_threshold, roc_curve_tensor)

                    # attr_loss, attr_accu = self.validate_attr(conf)
                    # print(attr_loss, attr_accu)

                    self.backbone.train()
                    self.idprehead.train()
                    self.attrhead.train()
                if self.step % self.save_every == 0 and self.step != 0:
                    self.save_state(conf, accuracy)
                    
                self.step += 1
                
        self.save_state(conf, accuracy, to_save_folder=True, extra='final')

    def schedule_lr(self):
        for params in self.optimizer.param_groups:                 
            params['lr'] /= 10
        print(self.optimizer)
    
    def validate_attr(self, conf):
        self.backbone.eval()
        self.attrhead.eval()
        losses = []
        accuracies = []
        with torch.no_grad():
            for i, (input, target) in enumerate(self.valid_loader):
                input = input.to(conf.device)
                target = target.to(conf.device)
                target = target[:, :40]
                target = (target + 1) * 0.5
                target = target.type(torch.cuda.FloatTensor)
                # compute output
                embedding = self.backbone(input)
                output = self.attrhead(embedding)
                # measure accuracy and record loss
                loss = conf.bc_loss(output, target)
                pred = torch.where(torch.sigmoid(output) > 0.5, 1.0, 0.0) 
                accuracies.append((pred == target).float().sum() / (target.size()[0] * target.size()[1]))
                losses.append(loss)

        loss_avg = sum(losses) / len(losses)
        accu_avg = sum(accuracies) / len(accuracies)
        return loss_avg, accu_avg



    # def infer(self, conf, faces, target_embs, tta=False):
    #     '''
    #     faces : list of PIL Image
    #     target_embs : [n, 512] computed embeddings of faces in facebank
    #     names : recorded names of faces in facebank
    #     tta : test time augmentation (hfilp, that's all)
    #     '''
    #     embs = []
    #     for img in faces:
    #         if tta:
    #             mirror = trans.functional.hflip(img)
    #             emb = self.model(conf.test_transform(img).to(conf.device).unsqueeze(0))
    #             emb_mirror = self.model(conf.test_transform(mirror).to(conf.device).unsqueeze(0))
    #             embs.append(l2_norm(emb + emb_mirror))
    #         else:                        
    #             embs.append(self.model(conf.test_transform(img).to(conf.device).unsqueeze(0)))
    #     source_embs = torch.cat(embs)
        
    #     diff = source_embs.unsqueeze(-1) - target_embs.transpose(1,0).unsqueeze(0)
    #     dist = torch.sum(torch.pow(diff, 2), dim=1)
    #     minimum, min_idx = torch.min(dist, dim=1)
    #     min_idx[minimum > self.threshold] = -1 # if no match, set idx to -1
    #     return min_idx, minimum               