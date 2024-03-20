import datetime
from dataset import *
import evaluation
# from cff_sgfm_mga import CFF_UVOS
# from cff_cbm import CFF_UVOS
# from cff_net import CFF_UVOS
from CFF_Net import CFF_UVOS
from trainer import Trainer
from optparse import OptionParser
import warnings
from utils.YouTubeObjects_to_eval import traverse_files, del_filepath

warnings.filterwarnings('ignore')

parser = OptionParser()
parser.add_option('--train', action='store_true', default=None)
parser.add_option('--test', action='store_true', default=None)
options = parser.parse_args()[0]

#  /mnt/31f271cb-1eab-41e4-aa15-7caf8b6e7528/gck/Second/YouTubeObjects
def train_duts_davis(model, ver, date):
    duts_set = TrainDUTS('/mnt/31f271cb-1eab-41e4-aa15-7caf8b6e7528/gck/Second/DUTS', clip_n=384)
    davis_set = TrainDAVIS('/mnt/31f271cb-1eab-41e4-aa15-7caf8b6e7528/gck/Second/DAVIS', '2016', 'train', clip_n=128)
    train_set = torch.utils.data.ConcatDataset([duts_set, davis_set])
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=8, shuffle=True, num_workers=4, pin_memory=True)
    val_set = TestDAVIS('/mnt/31f271cb-1eab-41e4-aa15-7caf8b6e7528/gck/Second/DAVIS', '2016', 'val')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    trainer = Trainer(model, ver, optimizer, train_loader, val_set, date, save_name='CFF_UVOS', save_step=500, val_step=100)
    trainer.train(4000)


def test_davis(model, date_time):
    evaluator = evaluation.Evaluator(TestDAVIS('/mnt/31f271cb-1eab-41e4-aa15-7caf8b6e7528/gck/Second/DAVIS', '2016', 'val'))
    evaluator.evaluate(model, os.path.join('outputs', date_time, 'DAVIS16_val'))


def test_fbms(model, date_time):
    test_set = TestFBMS('/mnt/31f271cb-1eab-41e4-aa15-7caf8b6e7528/gck/Second/FBMS/TestSet')
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, num_workers=4)
    model.cuda()
    ious = []

    # inference
    for vos_data in test_loader:
        imgs = vos_data['imgs'].cuda()
        flows = vos_data['flows'].cuda()
        masks = vos_data['masks']
        video_name = vos_data['video_name'][0]
        files = vos_data['files']
        os.makedirs('outputs/{}/FBMS_test/{}'.format(date_time, video_name), exist_ok=True)
        vos_out = model(imgs, flows)

        # get iou of each sequence
        iou = 0
        count = 0
        for i in range(masks.size(1)):
            tv.utils.save_image(vos_out['masks'][0, i].float().cpu(), 'outputs/{}/FBMS_test/{}/{}'.format(date_time, video_name, files[i][0].split('/')[-1]))
            if torch.sum(masks[0, i]) == 0:
                continue
            iou = iou + torch.sum(masks[0, i] * vos_out['masks'][0, i].cpu()) / torch.sum((masks[0, i] + vos_out['masks'][0, i].cpu()).clamp(0, 1))
            count = count + 1
        print('{} iou: {:.5f}'.format(video_name, iou / count))
        ious.append(iou / count)
    print('total seqs\' iou: {:.5f}\n'.format(sum(ious) / len(ious)))


def test_ytobj(model, date_time):
    test_set = TestYTOBJ('/mnt/31f271cb-1eab-41e4-aa15-7caf8b6e7528/gck/Second/YouTubeObjects')
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, num_workers=4)
    model.cuda()
    ious = {'aeroplane': [], 'bird': [], 'boat': [], 'car': [], 'cat': [], 'cow': [], 'dog': [], 'horse': [], 'motorbike': [], 'train': []}
    total_iou = 0
    total_count = 0

    for vos_data in test_loader:
        imgs = vos_data['imgs'].cuda()
        flows = vos_data['flows'].cuda()
        masks = vos_data['masks']
        class_name = vos_data['class_name'][0]
        video_name = vos_data['video_name'][0]
        files = vos_data['files']
        os.makedirs('outputs/{}/YouTubeObjects/{}/{}'.format(date_time, class_name, video_name), exist_ok=True)
        vos_out = model(imgs, flows)

        # get iou of each sequence
        iou = 0
        count = 0
        for i in range(masks.size(1)):
            tv.utils.save_image(vos_out['masks'][0, i].float().cpu(), 'outputs/{}/YouTubeObjects/{}/{}/{}'.format(date_time, class_name, video_name, files[i][0].split('/')[-1]))
            if torch.sum(masks[0, i]) == 0:
                continue
            iou = iou + torch.sum(masks[0, i] * vos_out['masks'][0, i].cpu()) / torch.sum((masks[0, i] + vos_out['masks'][0, i].cpu()).clamp(0, 1))
            count = count + 1
        if count == 0:
            continue
        print('{}_{} iou: {:.5f}'.format(class_name, video_name, iou / count))
        ious[class_name].append(iou / count)
        total_iou = total_iou + iou / count
        total_count = total_count + 1

    # calculate overall iou
    for class_name in ious.keys():
        print('class: {} seqs\' iou: {:.5f}'.format(class_name, sum(ious[class_name]) / len(ious[class_name])))
    print('total seqs\' iou: {:.5f}\n'.format(total_iou / total_count))


def test_lvid(model, date_time):
    test_set = TestLVID('/mnt/31f271cb-1eab-41e4-aa15-7caf8b6e7528/gck/Second/LongVideos')
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, num_workers=4)
    model.cuda()
    ious = []


    for vos_data in test_loader:
        imgs = vos_data['imgs'].cuda()
        flows = vos_data['flows'].cuda()
        masks = vos_data['masks']
        video_name = vos_data['video_name'][0]
        files = vos_data['files']
        os.makedirs('outputs/{}/LongVideos/{}'.format(date_time, video_name), exist_ok=True)
        vos_out = model(imgs, flows)

        # get iou of each sequence
        iou = 0
        count = 0
        for i in range(masks.size(1)):
            tv.utils.save_image(vos_out['masks'][0, i].float().cpu(), 'outputs/{}/LongVideos/{}/{}'.format(date_time, video_name, files[i][0].split('/')[-1]))
            if torch.sum(masks[0, i]) == 0:
                continue
            iou = iou + torch.sum(masks[0, i] * vos_out['masks'][0, i].cpu()) / torch.sum((masks[0, i] + vos_out['masks'][0, i].cpu()).clamp(0, 1))
            count = count + 1
        print('{} iou: {:.5f}'.format(video_name, iou / count))
        ious.append(iou / count)

    # calculate overall iou
    print('total seqs\' iou: {:.5f}\n'.format(sum(ious) / len(ious)))


def main():
    # set device
    torch.cuda.set_device(0)
    ver = 'rn101'
    aos = True
    model = CFF_UVOS(ver, aos).eval()

    model_name = 'CFF-Net-test'

    # training stage
    if options.train:
        model = torch.nn.DataParallel(model)
        train_duts_davis(model, ver, model_name)

    # testing stage
    if options.test:
        model.load_state_dict(torch.load('weights/CFF_Net_best.pth', map_location='cpu'))
        with torch.no_grad():
            test_davis(model, model_name)
            test_fbms(model, model_name)
            test_ytobj(model, model_name)
            # test_lvid(model, date_time)
main()