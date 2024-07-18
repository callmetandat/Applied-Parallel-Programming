from dataset.HumanLoader import HumanMattingDataset
from torch.utils.data import DataLoader
import numpy as np
from model.unet import U_net
import torch


def iou(pred, target, smooth=1e-6):
    intersection = (pred * target).sum((2, 3))  # sum over spatial dimensions
    union = (pred + target).sum((2, 3)) - intersection  # sum over spatial dimensions
    iou_score = (intersection + smooth) / (union + smooth)
    return iou_score


def valdiate(model: U_net, dataloader: DataLoader, thresholds: list, device: str = 'cuda'):
    model.eval()
    iou_scores = {threshold: [] for threshold in thresholds}

    with torch.no_grad():
        for img, mask, _, _ in dataloader:
            img = img.to(device)
            mask = mask.to(device)

            preds = model(img)

            preds = torch.sigmoid(preds)

            for threshold in thresholds:
                preds_threshold = preds > threshold
                batch_iou = iou(preds_threshold, mask)
                iou_scores[threshold].extend(batch_iou.cpu().numpy())

    mean_iou = {threshold: np.mean(scores) for threshold, scores in iou_scores.items()}
    return mean_iou


def main():
    model = U_net(1).cuda()
    weights_dict = torch.load(r'weights/unet_model_best.pth')

    model.load_state_dict(weights_dict)
    human_dataset = HumanMattingDataset(split='val')
    human_loader = DataLoader(human_dataset, batch_size=4, num_workers=4)

    thresholds = [0.1, 0.3, 0.5]
    print(f'Data length {human_dataset.__len__()}')
    
    results = valdiate(model, human_loader, thresholds)
    
    for threshold, mean_iou in results.items():
        print(f'mIoU threshold {threshold}: {mean_iou}')


if __name__ == '__main__':
    main()
