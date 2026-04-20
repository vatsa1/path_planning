import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import os
import image
import numpy as np
from random import seed
from sim import get_tableau_palette
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR

# ==================================================
mean_rgb = [0.485, 0.456, 0.406]
std_rgb = [0.229, 0.224, 0.225]
# ==================================================

class RGBDataset(Dataset):
    def __init__(self, img_dir):
        """
            Initialize instance variables.
            :param img_dir (str): path of train or test folder.
            :return None:
        """
        # TODO: complete this method
        # ===============================================================================
        self.img_dir = img_dir
        self.dataset_length = len(os.listdir(os.path.join(img_dir, 'rgb')))
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean_rgb, std=std_rgb)])
        # rgb_dir = os.path.join(img_dir, "rgb")
        pass
        # ===============================================================================

    def __len__(self):
        """
            Return the length of the dataset.
            :return dataset_length (int): length of the dataset, i.e. number of samples in the dataset
        """
        # TODO: complete this method
        # ===============================================================================
        return self.dataset_length
        
        # ===============================================================================

    def __getitem__(self, idx):
        """
            Given an index, return paired rgb image and ground truth mask as a sample.
            :param idx (int): index of each sample, in range(0, dataset_length)
            :return sample: a dictionary that stores paired rgb image and corresponding ground truth mask.
        """
        # TODO: complete this method
        # Hint:
        # - Use image.read_rgb() and image.read_mask() to read the images.
        # - Think about how to associate idx with the file name of images.
        # - Remember to apply transform on the sample.
        # ===============================================================================
        rgb_dir = os.path.join(self.img_dir, "rgb")
        rgb_filename = f"{idx}_rgb.png"
        rgb_path= os.path.join(rgb_dir, rgb_filename)
        rgb_img = image.read_rgb(rgb_path)
        
        gt_dir = os.path.join(self.img_dir, "gt")
        gt_filename = f"{idx}_gt.png"
        gt_path = os.path.join(gt_dir, gt_filename)
        gt_mask = image.read_mask(gt_path)

        sample = {'input': self.transform(rgb_img), 'target': torch.LongTensor(gt_mask)}
        return sample
        # ===============================================================================


class miniUNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        """
        A simplified U-Net with twice of down/up sampling and single convolution.
        ref: https://arxiv.org/abs/1505.04597, https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py
        :param n_channels (int): number of channels (for grayscale 1, for rgb 3)
        :param n_classes (int): number of segmentation classes (num objects + 1 for background)
        """
        super(miniUNet, self).__init__()
        # TODO: complete this method
        # ===============================================================================
        self.n_channels = n_channels
        self.n_classes = n_classes

        #contracting path
        self.conv1 = nn.Conv2d(n_channels, 16, kernel_size= 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        #expansive path
        self.conv6 = nn.Conv2d(256+128, 128, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(128+64, 64, kernel_size=3, padding=1)
        self.conv8 = nn.Conv2d(64+32, 32, kernel_size=3, padding=1)
        self.conv9 = nn.Conv2d(32+16, 16, kernel_size=3, padding=1)
        self.conv10 = nn.Conv2d(16, n_classes, kernel_size=1)
        self.interpolate = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        #self.interpolate = nn.Upsample(scale_factor=2)
        # ===============================================================================

    def forward(self, x):
        # TODO: complete this method
        # ===============================================================================
        #print(x.shape)
        x1 = F.relu(self.conv1(x)) #3 to 16
        x2 = self.pool(x1)
        x2 = F.relu(self.conv2(x2)) #16 to 32
        x3 = self.pool(x2)
        x3 = F.relu(self.conv3(x3)) #32 to 64
        x4 = self.pool(x3)
        x4 = F.relu(self.conv4(x4)) #64 to 128
        x5 = self.pool(x4)
        x5 = F.relu(self.conv5(x5)) #128 to 256 
        x6 = self.interpolate(x5)
        x7= torch.cat([x4,x6], dim=1)
        x7 = F.relu(self.conv6(x7)) # 256 to 128
        x7 = self.interpolate(x7)
        x8 = torch.cat([x3, x7], dim=1)
        x8 = F.relu(self.conv7(x8))  # 128 to 64
        #x8 = self.interpolate(x8.unsqueeze(2)).squeeze(2)
        x8 = self.interpolate(x8)
        x9 = torch.cat([x2, x8], dim=1)
        x9 = F.relu(self.conv8(x9))  # 64 to 32
        x9 = self.interpolate(x9)
        x10 = torch.cat([x1, x9], dim=1)
        x10 = F.relu(self.conv9(x10))  # 32 to 16
        x11 = F.relu(self.conv10(x10))  # 16 to 6

        output = x11
        return output# ===============================================================================

    # def forward(self, x):
    #     # TODO: complete this method
    #     #print(x.shape)
    #     x1 = F.relu(self.conv1(x)) #3 to 16
    #     x2 = self.pool(x1)
    #     x2 = F.relu(self.conv2(x2)) #16 to 32
    #     x3 = self.pool(x2)
    #     x3 = F.relu(self.conv3(x3)) #32 to 64
    #     x4 = self.pool(x3)
    #     x4 = F.relu(self.conv4(x4)) #64 to 128
    #     x5 = self.pool(x4)
    #     x5 = F.relu(self.conv5(x5)) #128 to 256 
    #     x6 = self.interpolate(x5.unsqueeze(2))[:, :, :-1, :-1]
    #     x7= torch.cat([x4,x6], dim=1)
    #     x7 = F.relu(self.conv6(x7)) # 256 to 128
    #     x7 = self.interpolate(x7.unsqueeze(2))[:, :, :-1, :-1]
    #     x8 = torch.cat([x3, x7], dim=1)
    #     x8 = F.relu(self.conv7(x8))  # 128 to 64
    #     x8 = self.interpolate(x8.unsqueeze(2))[:, :, :-1, :-1]
    #     x9 = torch.cat([x2, x8], dim=1)
    #     x9 = F.relu(self.conv8(x9))  # 64 to 32
    #     x9 = self.interpolate(x9.unsqueeze(2))[:, :, :-1, :-1]
    #     x10 = torch.cat([x1, x9], dim=1)
    #     x10 = F.relu(self.conv9(x10)) # 32 to 16
    #     x10 = self.conv10(x10) # 16 to n_classes
    #     return x10

def save_chkpt(model, epoch, test_miou, chkpt_path):
    """
        Save the trained model.
        :param model (torch.nn.module object): miniUNet object in this homework, trained model.
        :param epoch (int): current epoch number.
        :param test_miou (float): miou of the test set.
        :return: None
    """
    state = {'model_state_dict': model.state_dict(),
             'epoch': epoch,
             'model_miou': test_miou, }
    torch.save(state, chkpt_path)
    print("checkpoint saved at epoch", epoch)


def load_chkpt(model, chkpt_path, device):
    """
        Load model parameters from saved checkpoint.
        :param model (torch.nn.module object): miniUNet model to accept the saved parameters.
        :param chkpt_path (str): path of the checkpoint to be loaded.
        :return model (torch.nn.module object): miniUNet model with its parameters loaded from the checkpoint.
        :return epoch (int): epoch at which the checkpoint is saved.
        :return model_miou (float): miou of the test set at the checkpoint.
    """
    checkpoint = torch.load(chkpt_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    model_miou = checkpoint['model_miou']
    print("epoch, model_miou:", epoch, model_miou)
    return model, epoch, model_miou


def save_prediction(model, dataloader, dump_dir, device, BATCH_SIZE):
    """
        For all datapoints d in dataloader, save  ground truth segmentation mask (as {id}.png)
          and predicted segmentation mask (as {id}_pred.png) in dump_dir.
        :param model (torch.nn.module object): trained miniUNet model
        :param dataloader (torch.utils.data.DataLoader object): dataloader to use for getting predictions
        :param dump_dir (str): dir path for dumping predictions
        :param device (torch.device object): pytorch cpu/gpu device object
        :param BATCH_SIZE (int): batch size of dataloader
        :return: None
    """
    print(f"Saving predictions in directory {dump_dir}")
    if not os.path.exists(dump_dir):
        os.makedirs(dump_dir)

    model.eval()
    with torch.no_grad():
        for batch_ID, sample_batched in enumerate(dataloader):
            data, target = sample_batched['input'].to(device), sample_batched['target'].to(device)
            output = model(data)
            _, pred = torch.max(output, dim=1)
            for i in range(pred.shape[0]):
                gt_image = convert_seg_split_into_color_image(target[i].cpu().numpy())
                pred_image = convert_seg_split_into_color_image(pred[i].cpu().numpy())
                combined_image = np.concatenate((gt_image, pred_image), axis=1)
                test_ID = batch_ID * BATCH_SIZE + i
                image.write_mask(combined_image, f"{dump_dir}/{test_ID}_gt_pred.png")


# def iou(pred, target, n_classes=4):
#     """
#         Compute IoU on each object class and return as a list.
#         :param pred (np.array object): predicted mask
#         :param target (np.array object): ground truth mask
#         :param n_classes (int): number of classes
#         :return cls_ious (list()): a list of IoU on each object class
#     """
#     cls_ious = []
#     # Flatten
#     pred = pred.view(-1)
#     target = target.view(-1)
#     for cls in range(1, n_classes):  # class 0 is background
#         pred_P = pred == cls
#         target_P = target == cls
#         pred_N = ~pred_P
#         target_N = ~target_P
#         if target_P.sum() == 0:
#             # print("class", cls, "doesn't exist in target")  # testing (comment out later, don't delete)
#             continue
#         else:
#             intersection = pred_P[target_P].sum()  # TP
#             FP = pred_P[target_N].sum()
#             FN = pred_N[target_P].sum()
#             union = intersection + FN + FP  # or pred_P.sum() + target_P.sum() - intersection
#             cls_ious.append(float(intersection) / float(union))
#     return cls_ious

def iou(prediction, target):
    """
    In:
        prediction: Tensor [batchsize, class, height, width], predicted mask.
        target: Tensor [batchsize, height, width], ground truth mask.
    Out:
        batch_ious: a list of floats, storing IoU on each batch.
    Purpose:
        Compute IoU on each data and return as a list.
    """
    _, pred = torch.max(prediction, dim=1)
    batch_num = prediction.shape[0]
    class_num = prediction.shape[1]
    batch_ious = list()
    for batch_id in range(batch_num):
        class_ious = list()
        for class_id in range(1, class_num):  # class 0 is background
            mask_pred = (pred[batch_id] == class_id).int()
            mask_target = (target[batch_id] == class_id).int()
            if mask_target.sum() == 0: # skip the occluded object
                continue
            intersection = (mask_pred * mask_target).sum()
            union = (mask_pred + mask_target).sum() - intersection
            class_ious.append(float(intersection) / float(union))
        batch_ious.append(np.mean(class_ious))
    return batch_ious

def run(model, loader, criterion, optimizer=None, is_train= False):
    """
        Run forward pass for each sample in the dataloader. Run backward pass and optimize if training.
        Calculate and return mean_epoch_loss and mean_iou
        :param model (torch.nn.module object): miniUNet model object
        :param loader (torch.utils.data.DataLoader object): dataloader 
        :param criterion (torch.nn.module object): Pytorch criterion object
        :param is_train (bool): True if training
        :param optimizer (torch.optim.Optimizer object): Pytorch optimizer object
        :return mean_epoch_loss (float): mean loss across this epoch
        :return mean_iou (float): mean iou across this epoch
    """
    model.train(is_train)
    # TODO: complete this function 
    # ===============================================================================
    mean_epoch_loss, mean_iou = 0.0, 0.0
    n = 0
    for batch_idx, batch_data in enumerate(loader):
        inputs, targets = batch_data['input'].to(device), batch_data['target'].to(device)
        # targets = targets.type(torch.LongTensor)
        # inputs=inputs.permute(0,3,1,2)
        # print(f'inputs: {inputs.shape}')
        outputs = model(inputs)
        # print(f'output shape= {outputs.shape}')
        # print(f'targets: {targets.shape}')
        loss = criterion(outputs, targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Compute batch IoU and add to total
        batch_iou = iou(outputs, targets)
        mean_iou += sum(batch_iou)

        # Compute batch loss and add to total
        mean_epoch_loss += loss.item()
        n += inputs.size(0)
    return mean_epoch_loss/n, mean_iou/n
    # ===============================================================================

def convert_seg_split_into_color_image(img):
    color_palette = get_tableau_palette()
    colored_mask = np.zeros((*img.shape, 3))

    print(np.unique(img))

    for i, unique_val in enumerate(np.unique(img)):
        if unique_val == 0:
            obj_color = np.array([0, 0, 0])
        else:
            obj_color = np.array(color_palette[i-1]) * 255
        obj_pixel_indices = (img == unique_val)
        colored_mask[:, :, 0][obj_pixel_indices] = obj_color[0]
        colored_mask[:, :, 1][obj_pixel_indices] = obj_color[1]
        colored_mask[:, :, 2][obj_pixel_indices] = obj_color[2]
    return colored_mask.astype(np.uint8)


if __name__ == "__main__":
    # ==============Part 4 (a) Training Segmentation model ================
    # Complete all the TODO's in this file
    # - HINT: Most TODO's in this file are exactly the same as homework 2.

    seed(0)
    torch.manual_seed(0)


    # Check if GPU is being detected
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    root_dir = './dataset/'
    train_dir = root_dir + 'train/'
    test_dir = root_dir + 'test/'

    # TODO: Prepare train and test datasets
    # Load the "dataset" directory using RGBDataset class as a pytorch dataset
    # Split the above dataset into train and test dataset in 9:1 ratio using `torch.utils.data.random_split` method
    split= 0.1
    dataset = RGBDataset(root_dir)
    # test_dataset = RGBDataset(test_dir)
    dataset_size = len(dataset)
    test_size = int(np.floor(split * dataset_size))
    train_size = dataset_size - test_size
    train_set, test_set = torch.utils.data.random_split(dataset, [train_size, test_size])

    # TODO: Prepare train and test Dataloaders. Use appropriate batch size
    train_loader = DataLoader(train_set, batch_size=32)
    test_loader = DataLoader(test_set, batch_size=4)
    
     # TODO: Prepare model
    model = miniUNet(n_channels=3, n_classes=4)
    model.to(device)


    # TODO: Define criterion, optimizer and learning rate scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),lr=0.01)
    #lr_scheduler = StepLR(optimizer, step_size=5, gamma=0.1)



    # TODO: Train and test the model. 
    # Tips:
    # - Remember to save your model with best mIoU on objects using save_chkpt function
    # - Try to achieve Test mIoU >= 0.9 (Note: the value of 0.9 only makes sense if you have sufficiently large test set)
    # - Visualize the performance of a trained model using save_prediction method. Make sure that the predicted segmentation mask is almost correct.

    train_loss_list, train_miou_list, test_loss_list, test_miou_list = list(), list(), list(), list()
    epoch, max_epochs = 1, 30  # TODO: you may want to make changes here
    best_miou = float('-inf')
    while epoch <= max_epochs:
        print('Epoch (', epoch, '/', max_epochs, ')')
        train_loss, train_miou = run(model, train_loader, criterion, is_train=True,optimizer= optimizer)
        test_loss, test_miou = run(model, test_loader, criterion, is_train=False, optimizer= optimizer)
        #lr_scheduler.step()
        train_loss_list.append(train_loss)
        train_miou_list.append(train_miou)
        test_loss_list.append(test_loss)
        test_miou_list.append(test_miou)
        print('Train loss & mIoU: %0.2f %0.2f' % (train_loss, train_miou))
        print('Test loss & mIoU: %0.2f %0.2f' % (test_loss, test_miou))
        print('---------------------------------')
        if test_miou > best_miou:
            best_miou = test_miou
            chkpt_path = 'checkpoint.pth.tar'
            save_chkpt(model, epoch, test_miou, chkpt_path)
        epoch += 1

    # Load the best checkpoint, use save_prediction() on the validation set and test set
    model, epoch, best_miou = load_chkpt(model, 'checkpoint.pth.tar',device)
    save_prediction(model = model ,dataloader= test_loader,dump_dir='./dataset/dump_dir', device = device, 
                    BATCH_SIZE= 8)
    