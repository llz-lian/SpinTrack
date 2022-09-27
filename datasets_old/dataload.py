
from loader import TrackingSampler,LTRLoader,OnlineTrackingSampler
from data_got10k import Got10k
from lasot import Lasot
from coco_seq import MSCOCOSeq
from tracking_net import TrackingNet
from tracking_net_lmdb import TrackingNet_lmdb
from processing import Processing,OnlineProcessing
import transfrom as tfm
from torch.utils.data.distributed import DistributedSampler




class Settings:
    """ Training settings, e.g. the paths to datasets and networks."""
    def __init__(self):
        self.set_default()

    def set_default(self):
        self.root = './data/GOT10K/train'
        # self.laroot = './data/LaSOTBenchmark'
        # self.cocoroot = './data/COCO'
        # self.trackingnet = './data/trackingnet'
        self.use_gpu = True


def genSettings(batch_size = 1,num_workers = 1,search_size = 258,template_size = 128,search_area_factor = 4.0):
    print('batch_size:{},num_workers:{},search_size:{},template_size:{}'.format(batch_size,num_workers,search_size,template_size))
    print('search_area_factor:{}'.format(search_area_factor))
    settings = Settings()
    settings.batch_size = batch_size
    settings.num_workers = num_workers
    settings.multi_gpu = False
    settings.normalize_mean = [0.485, 0.456, 0.406]
    settings.normalize_std = [0.229, 0.224, 0.225]
    settings.search_area_factor = search_area_factor
    settings.template_area_factor = 2.0

    settings.search_sz = search_size#256
    settings.temp_sz = template_size#112
    if search_area_factor == 4.0:
        settings.center_jitter_factor = {'search': 3, 'template': 0.1,'online':0.2}
        settings.scale_jitter_factor = {'search': 0.25, 'template': 0.1,'online':0.1}
    else:
        settings.center_jitter_factor = {'search': 4, 'template': 0.1,'online':0.2}
        settings.scale_jitter_factor = {'search': 0.5, 'template': 0.1,'online':0.1}
    settings.only_got = False
    return settings


def getDataloader(settings):
    transform_joint = tfm.Transform(tfm.ToGrayscale(probability=0.05),tfm.RandomHorizontalFlip(probability=0.2))
    print('create transform_joint')
    transform_train = tfm.Transform(tfm.ToTensorAndJitter(0.2),tfm.RandomHorizontalFlip(probability=0.2),
                                    tfm.Normalize(mean=settings.normalize_mean, std=settings.normalize_std))
    print('create transform_train')
    data_processing_train = Processing(search_area_factor=settings.search_area_factor,
                                                      template_area_factor = settings.template_area_factor,
                                                      search_sz=settings.search_sz,
                                                      temp_sz=settings.temp_sz,
                                                      center_jitter_factor=settings.center_jitter_factor,
                                                      scale_jitter_factor=settings.scale_jitter_factor,
                                                      mode='sequence',
                                                      transform=transform_train,
                                                      joint_transform=transform_joint)
    print('create data_processing_train')    
    got10k_train = Got10k(settings.root)
    print('only train on got10k')
    dataset_train = TrackingSampler([got10k_train], [1],num_template_frames=1,
                                samples_per_epoch=1000*settings.batch_size, max_gap=120, processing=data_processing_train)
    train_sampler = DistributedSampler(dataset_train)
    loader_train = LTRLoader('train', dataset_train,sampler=train_sampler, training=True, batch_size=settings.batch_size, num_workers=settings.num_workers,
                             shuffle=False, drop_last=False, stack_dim=0,pin_memory = True)
    return loader_train

def getOnline(settings = genSettings()):
    transform_joint = tfm.Transform(tfm.ToGrayscale(probability=0.05),tfm.RandomHorizontalFlip(probability=0.1))

    transform_train = tfm.Transform(tfm.ToTensorAndJitter(0.2),tfm.RandomHorizontalFlip(probability=0.1),
                                    tfm.Normalize(mean=settings.normalize_mean, std=settings.normalize_std))

    data_processing_train = OnlineProcessing(search_area_factor=settings.search_area_factor,
                                                      template_area_factor = settings.template_area_factor,
                                                      online_area_factor=settings.search_area_factor,
                                                      search_sz=settings.search_sz,
                                                      temp_sz=settings.temp_sz,
                                                      online_sz=settings.search_sz,
                                                      center_jitter_factor=settings.center_jitter_factor,
                                                      scale_jitter_factor=settings.scale_jitter_factor,
                                                      mode='sequence',
                                                      transform=transform_train,
                                                      joint_transform=transform_joint)
                                        
    lasot_train = Lasot(root = settings.laroot,split='train')#,vid_ids=[i for i in range(1,20)])
    got10k_train = Got10k(settings.root)
    coco_train = MSCOCOSeq(root = settings.cocoroot,version='2017')
    tracking_net = TrackingNet(root = settings.trackingnet,set_ids = [0,1,2,3,4,5,6])

    dataset_train = OnlineTrackingSampler([got10k_train,lasot_train,coco_train,tracking_net], [1,1,1,1],num_template_frames=1,
                                samples_per_epoch=1000*settings.batch_size, max_gap=200, processing=data_processing_train)
    loader_train = LTRLoader('train', dataset_train, training=True, batch_size=settings.batch_size, num_workers=settings.num_workers,
                             shuffle=True, drop_last=True, stack_dim=0,pin_memory = True)
    return loader_train

if __name__ == "__main__":
    datas = getDataloader()
    for i,data in enumerate(datas):
        pass
    