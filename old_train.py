import torch
import torch.nn as nn
from model.model import My,GetMyBase,GetMySmall,GetMyLarge,load_pretrained,GetMyBase512,GetMyLarge384,GetMyBaseL,GetMyBase12
from model.spintrack import build_spin
from model_vit.vit_track import vit_base_patch16_224,vit_base_patch16_384
import torch.optim as optim
from torch.autograd import Variable
from datasets_old.dataload import getDataloader,genSettings,getOnline
from lossutil.util import  getCriterion,getCriterionvit,get_gaussian_maps
import torch.optim.lr_scheduler
from tqdm import tqdm
from warmup_scheduler import GradualWarmupScheduler
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler
import argparse
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", type=int)

parser.add_argument("--cb", action="store_false")
parser.add_argument("--rb", action="store_false")
parser.add_argument("--prec", action="store_true")
parser.add_argument("--twoc", action="store_true")
parser.add_argument("--chd", action="store_false")

parser.add_argument("--dh", action="store_false")
parser.add_argument("--ch", action="store_true")

parser.add_argument("--nopart", action="store_true")
parser.add_argument("--notpretrain", action="store_true")
parser.add_argument("--notailpara", action="store_true")
parser.add_argument("--use_roll", action="store_false")
parser.add_argument("--roll_kv", action="store_true")

parser.add_argument("--checkpoint",type=str,default = '')
args = parser.parse_args()

print('')
print('conv bias:{},rel bias:{}'.format(args.cb,args.rb))
print('pre cross:{},two cross:{}'.format(args.prec,args.twoc))
print('change dim:{}'.format(args.chd))
print('double head:{},center head:{}'.format(args.dh,args.ch))
print('')

torch.cuda.set_device(args.local_rank)
torch.distributed.init_process_group(backend='nccl')

train_func = GetMyLarge384
pre_trained_path = './useful_weight/Large384_330.pth' #mymodel_pre.pth'#spinT_pre.pth'mylarge256_pre
criterion = getCriterion().cuda()

batch_size = 30
num_workers = 6
search_size = 256
template_size = 128
lr1 = 0.0001#backbone
lr2 = 0.0002
backbone = 'basepart'
pretrained = True
train_online = False
def initSettings(model_name):
    global batch_size,search_size,template_size
    batch_size = 96
    num_workers = 8
    search_size = 256
    template_size = 128
    search_area_factor = 4.0
    if train_func == GetMyBase or train_func == GetMyBase12:
        batch_size = 16
        num_workers = 4
        template_size = 192
        search_size = 384
    if train_func == GetMyBaseL:
        batch_size = 11
        num_workers = 4
        template_size = 192
        search_size = 384
    if train_func == build_spin:
        batch_size = 24
        num_workers = 8
        template_size = 192
        search_size = 384
    if train_func == GetMyBase512:
        batch_size = 18
        num_workers = 4
        template_size = 256
        search_size = 512
        search_area_factor = 5.0
    if train_func == GetMyLarge:
        batch_size = 24
        template_size = 128
        search_size = 256
        search_area_factor = 4.0
    if train_func == GetMyLarge384:
        batch_size = 8
        template_size = 192
        search_size = 384
        search_area_factor = 5.0
    if train_func == vit_base_patch16_224:
        batch_size = 32
        template_size = 128
        search_size = 256
    if train_func == vit_base_patch16_384:
        batch_size = 16
        template_size = 192
        search_size = 384
    return genSettings(batch_size=batch_size,num_workers=num_workers,search_size=search_size,template_size=template_size,search_area_factor = search_area_factor)


settings = initSettings('base')#genSettings(batch_size=batch_size,num_workers=num_workers,search_size=search_size,template_size=template_size)



def preData(batch_data):
    template,target,bbox,temp_bbox = batch_data['template_images'][:,0],batch_data['search_images'],batch_data['search_anno'],batch_data['template_anno']
    bbox[:,0] = bbox[:,0] + bbox[:,2]/2
    bbox[:,1] = bbox[:,1] + bbox[:,3]/2
    #online =  batch_data['template_images'][:,1]
    # temp_bbox[:,0] = temp_bbox[:,0] + temp_bbox[:,2]/2
    # temp_bbox[:,1] = temp_bbox[:,1] + temp_bbox[:,3]/2


    bbox = bbox/search_size    
    # temp_bbox = temp_bbox/template_size
    online = None
    if train_online:
        online = batch_data['online_images'][:,0]
        online_bbox = batch_data['online_anno']
    return template,target,bbox,temp_bbox,online#,online

def doOneEpoch(model,dataset,criterion,optimal = None ):
    loss_sum = 0
    loss2_sum = 0

    iou = 0
    bbox_sum = 0

    vari_sum = 0
    cls_label = ['labels']*batch_size
    boxes = ['boxes']*batch_size

    cls_label_v = [[0]]*batch_size
    cls_label_v = torch.Tensor(cls_label_v).long().cuda()
    sub = len(dataset)
    for batch_data in tqdm(dataset):
        template,target,bbox,temp_bbox,online = preData(batch_data)

        if online is not None:
            online = Variable(online).cuda()
        target = Variable(target).cuda()
        template = Variable(template).cuda()

        bbox = bbox.cuda()
        bbox_s = bbox

        bbox = bbox.unsqueeze(1)
        zip1 = zip(cls_label,cls_label_v)
        zip2 = zip(boxes,bbox)

        dic1 = [{k:v} for k,v in zip1]
        dic2 = [{k:v} for k,v in zip2]

        zip3 = zip(dic1,dic2)
        [v.update(k) for k,v in zip3]
        label = dic2
        optimal.zero_grad()
        losses = []
        
        outputs1,outputs2 = model(template,target,online)#output1[576,576] output2[576,144]            
        for output in outputs1:
            ind1,maps1 = criterion.getIndices(output,label,False)
            criterion.set_vari()
            loss = criterion(output,label,bbox_s,ind1,maps1)
            losses.append(loss)
        for output in outputs2:
            ind2,maps2 = criterion.getIndices(output,label,True)
            criterion.set_ce()
            loss = criterion(output,label,bbox_s,ind2,maps2)
            losses.append(loss)
            #loss2 = criterion(output2,label,temp_bbox)

        iou += losses[1]['iou'].item() 
        bbox_sum += losses[1]['loss_giou'].item()
        vari_sum += losses[1]['varifocal'].item()#varifocal

        loss = 0
        loss2 = 0
#         w = [0.1665,0.1665,0.333,0.1333,0.1333,0.1333]#[de,de,en,en]
        w =  [0.333,0.333,0.2,0.2]
#         w = [1]
        i = 0
        for l in losses:
            if 'loss_giou' in l:
                l = 2 *l['loss_giou']  + 8.3 * l['varifocal'] + 5*l['loss_bbox'] #+ 2*l['weight']
            else:
                l = 8.3 * l['varifocal']
            loss = loss + l * w[i]
            loss2 = l
            i += 1
        if torch.isnan(loss):
            continue
        loss.backward()
        optimal.step()
        loss_sum = loss_sum + loss.item()
        loss2_sum = loss2_sum + loss2.item()/2
#     if model.module.forward_cnt !=0:
#         print('encoder_len:{},decoder_len:{},forward_cnt:          {}'.format(model.module.encoder_len//model.module.forward_cnt,model.module.decoder_len//model.module.forward_cnt,model.module.forward_cnt))
    ret = {'loss_sum':loss_sum/sub,'loss_cls':vari_sum/sub,'iou':iou/sub,'loss_giou':bbox_sum/sub,'loss2_sum':loss2_sum/sub}#,'loss_vari':vari_sum/sub}

    return ret

# def doOneEpochhalf(model,dataset,criterion,optimal = None ):
#     loss_sum = 0
#     loss2_sum = 0

#     iou = 0
#     bbox_sum = 0
#     scaler = GradScaler()
#     vari_sum = 0
#     cls_label = ['labels']*batch_size
#     boxes = ['boxes']*batch_size

#     cls_label_v = [[0]]*batch_size
#     cls_label_v = torch.Tensor(cls_label_v).long().cuda()
#     sub = len(dataset)
#     for batch_data in tqdm(dataset):
#         template,target,bbox,temp_bbox,online = preData(batch_data)

#         if online is not None:
#             online = Variable(online).cuda()
#         target = Variable(target).cuda()
#         template = Variable(template).cuda()

#         bbox = bbox.cuda()
#         bbox_s = bbox

#         bbox = bbox.unsqueeze(1)
#         zip1 = zip(cls_label,cls_label_v)
#         zip2 = zip(boxes,bbox)

#         dic1 = [{k:v} for k,v in zip1]
#         dic2 = [{k:v} for k,v in zip2]

#         zip3 = zip(dic1,dic2)
#         [v.update(k) for k,v in zip3]
#         label = dic2
#         optimal.zero_grad()
#         losses = []

#         with autocast():
#             outputs1,outputs2 = model(template,target,online)#output1[576,576] output2[576,144]            
#             for output in outputs1:
#                 ind1,maps1 = criterion.getIndices(output,label)
#                 loss = criterion(output,label,bbox_s,ind1,maps1)
#                 losses.append(loss)
#             for output in outputs2:
#                 ind2,maps2 = criterion.getIndices(output,label,False)
#                 loss = criterion(output,label,bbox_s,ind2,maps2)
#                 losses.append(loss)
#             #loss2 = criterion(output2,label,temp_bbox)

#         iou += losses[1]['iou'].item() 

#         bbox_sum += losses[1]['loss_giou'].item()
#         vari_sum += losses[1]['varifocal'].item()#varifocal

#         loss = 0
#         loss2 = 0
#         w = [0.333,0.333,0.1667,0.1667]#[de,de,en,en]
#         i = 0
#         for l in losses:
#             # loss2 = loss2 + l['weight']
#             l = 2 *l['loss_giou']  + 8.3 * l['varifocal'] + 5*l['loss_bbox'] #+ 2*l['weight']
#             loss = loss + l * w[i]
#             loss2 = l
#             i += 1
#         if torch.isnan(loss):
#             continue
#         scaler.scale(loss).backward()
#         scaler.step(optimal)
#         scaler.update()
#         loss_sum = loss_sum + loss.item()
#         loss2_sum = loss2_sum + loss2.item()/2
#     ret = {'loss_sum':loss_sum/sub,'loss_cls':vari_sum/sub,'iou':iou/sub,'loss_giou':bbox_sum/sub,'loss2_sum':loss2_sum/sub}#,'loss_vari':vari_sum/sub}

#     return ret



def train(model,epoch,criterion,traindata,optimal,lr_schduler,save_name = "model.pth",vari = False,load_epoch = 0):
    name = save_name
    read_ep = load_epoch
    log_file ='./log_files/'+ model.str + '.txt'
    
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)    
    model = model.cuda()
    model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=False)
    model.module.set_not_all()
    file = open(log_file,'a')
    for step in range(read_ep,epoch):
        if step > 19 and not args.nopart:
            model.module.set_not_all()
        model.train()
        loss = doOneEpoch(model,traindata,criterion,optimal)
        lr_schduler.step()
        print("epoch:{}, {}".format(step,loss))

        lr = optimal.param_groups[0]['lr']
        lr2 = optimal.param_groups[1]['lr'] 
        print('lr:{},lr2:{}'.format(lr,lr2))
        file.write("epoch:{}, {}".format(step,loss))
        file.write('\n')
        file.write('lr:{},lr2:{}'.format(lr,lr2))
        file.write('\n')


        if (step) % 50 == 0 or lr == 0.:
            state = {'net':model.state_dict(),'epoch': 1 +step,'lr':lr,'lr2':lr2,'lr_schduler':lr_schduler}
            save_name ='_' + str(step) + name
            torch.save(state,save_name)

        if (step+1)%5 ==0:
            state = {'net':model.state_dict(),'epoch':1 +step,'lr':lr,'lr2':lr2,'lr_schduler':lr_schduler}
            s ='temp_'  + name
            torch.save(state,s)

def start(load_pth = None,pre_trained_path = None):
    mymodel = train_func(False,
              relative_bias = args.rb,
              use_conv_bias = args.cb,
              change_dim = args.chd,
              two_cross = args.twoc,
              pre_cross = args.prec,
              use_dhead = args.dh,
              use_chead = args.ch,
              use_roll = args.use_roll,
              roll_kv = args.roll_kv)
    print('part cross:{}'.format(args.nopart))
    if args.nopart:
        mymodel.str +='_npart_'
    if args.notpretrain:
        print('no pre trained')
        mymodel.str += '_nopretrain_'
    total = sum([param.nelement() for param in mymodel.parameters()])
    print("Number of parameter: %.2fM" % (total/1e6))

    epoch = 220
    lr = lr1#2e-5

    param_dicts = [
        {"params": [p for n, p in mymodel.named_parameters() if backbone in n and p.requires_grad]},
        {
            "params": [p for n, p in mymodel.named_parameters() if backbone not in n and p.requires_grad],
            "lr": lr2#0.0002,
        },
    ]
    print(backbone,lr1)
    load_epoch = 0
    if pretrained and not args.notpretrain:
        # ld = torch.load('temp.pth')['net']
        # mymodel.load_state_dict(ld)
        ld = torch.load(pre_trained_path,map_location=torch.device('cpu'))['net']
        ld = {k.replace('module.',''):v for k,v in ld.items()}
        
#         predict = {k:v for k,v in predict.items() if 'tail_layers.1' not in k}
#         predict = {k:v for k,v in predict.items() if 'head' not in k}
#         predict = {k:v for k,v in predict.items() if 'rel_bias' not in k}
#         ld = {k:v for k,v in ld.items() if 'basepart' in k}
        # ld = {k:v for k,v in ld.items() if 'pos' in k}
        # ld = {k:v for k,v in ld.items() if 'head' in k}
        if args.notailpara:
            ld = {k:v for k,v in ld.items() if 'base' in k}
            mymodel.str += '_notailpara_'
            print('no tail parameter')
        mymodel.load_state_dict(ld,strict=False)
        print('old loaded')
        ld = None
        predict = None


    if load_pth is not None:
        param_dicts = None
        pth = torch.load(load_pth,map_location=torch.device('cpu'))
        predict = {k.replace('module.',''):v for k,v in pth['net'].items()}
        
        mymodel.load_state_dict(predict,strict=False)
        load_epoch = pth['epoch']
        print('load_epoch:',load_epoch)
        scheduler_warmup = pth['lr_schduler']
        lr = pth['lr']
        #load_pretrained(mymodel,pre_trained_path)
        param_dicts = [
            {"params": [p for n, p in mymodel.named_parameters() if backbone in n and p.requires_grad]},
            {
                "params":[p for n, p in mymodel.named_parameters() if backbone not in n and p.requires_grad],
                "lr":pth['lr2']
            }
        ]        
       
        optimal = optim.AdamW(param_dicts,lr=lr,weight_decay = 1e-4)
        scheduler_warmup.optimizer = optimal

        scheduler_warmup.after_scheduler.optimizer = optimal
        pth = None
        predict = None
        print('model loaded')
    else:
        optimal = optim.AdamW(param_dicts,lr=lr,weight_decay = 1e-4)
        #scheduler_steplr = torch.optim.lr_scheduler.StepLR(optimal, step_size=130, gamma=0.1)
        scheduler_steplr = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimal,T_0=200,T_mult=1,eta_min=1e-8)#, gamma=0.1)
        scheduler_warmup = GradualWarmupScheduler(optimal, multiplier=1, total_epoch=20, after_scheduler=scheduler_steplr)
        optimal.zero_grad()
        optimal.step()
        scheduler_warmup.step()
    settings.only_got = True
    if settings.search_area_factor == 5.0:
        print('search_area_factor:5')
        mymodel.str += '_ar5'
    if settings.only_got:
        mymodel.str += '_got'
    if lr1 *10 > lr2:
        print('backbone lr large')
        mymodel.str +='_0.2lr'
    if lr1  ==  lr2:
        print('backbone lr same')
        mymodel.str +='_samelr'
    dataload = getDataloader(settings=settings)
    save_name = mymodel.str
    mymodel.return2 = True
    
    train(model = mymodel,epoch = epoch,criterion= criterion,traindata = dataload,
    optimal = optimal,lr_schduler = scheduler_warmup,vari=False,save_name=save_name,load_epoch = load_epoch)
    state = {'net':mymodel.state_dict(), 'optimal':optimal.state_dict(), 'epoch':epoch}
    torch.save(state,save_name)


if __name__ == "__main__":
    path = None
    pre_path = None
    if args.checkpoint != '':
        path = args.checkpoint
#     path = 'temp.pth'
    pre_path = pre_trained_path#'mymodel_pre.pth'
    #start_online('OTBgood_100depth_8embed_384_w_spin_base.pth')
    start(load_pth = path,pre_trained_path=pre_path)

    """
    model = My(embed_dim=384,depth=21,num_heads=16,num_addon=51,num_classes=1)
    model.load_state_dict(torch.load("num_addon_32_embed_dim_384_dpeth_16_epoch_420_lr_3e-05.pth")["net"])
    model.cuda(device)
    
    dirs = "./data/mydata/train"
    traindata = getDataLoader(dir = dirs,label_dir="./data/mydata/train/train_label.npy",batch_size=1,num_work=0,shuffle= False)

    imgs_path = glob.glob("./data/mydata/train/target/*.jpg")
    length = len("./data/mydata/train")
    imgs_path = sorted(imgs_path,key= lambda name: ord(name[length + 8:length+9])*10000 + int(name[length + 10:-4]))
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    outVideo = cv2.VideoWriter('save_train_video.avi', fourcc, 24, (256, 256))
    
    for i,batch_data in enumerate(traindata):
        img = cv2.imread(imgs_path[i])
        target,template,_,_ = batch_data

        output = model(target.cuda(),template.cuda())
        logit = output['pred_logits']
        bbox = output['pred_boxes']
        indx = torch.argmax(logit,dim = 1)[:,0].item()
        bbox = bbox[:,indx,:].squeeze()
        
        centerx,centery = bbox[0].item()*256,bbox[1].item()*256
        W,H = bbox[2].item()*128,bbox[3].item()*128
        pt1 = (int(centerx - W),int(centery - H))
        pt2 = (int(centerx + W),int(centery + H))
        img = cv2.rectangle(img,pt1,pt2,(0,255,0),2)
        
        
        outVideo.write(img)
    outVideo.release()
    """
