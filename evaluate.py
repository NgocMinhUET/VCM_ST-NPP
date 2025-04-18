import argparse
import torch
from models.combined_model import CombinedModel
from utils.model_utils import load_checkpoint
from utils.data_utils import get_dataloader
from utils.metric_utils import compute_map, compute_miou, compute_mota

def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CombinedModel().to(device)
    load_checkpoint(model, args.checkpoint)
    model.eval()

    eval_loader = get_dataloader(args.dataset, batch_size=1, split='val')

    all_preds, all_targets = [], []

    with torch.no_grad():
        for batch in eval_loader:
            inputs, targets, qp = batch['input'].to(device), batch['target'].to(device), batch['qp'].to(device)
            output = model(inputs, qp)
            all_preds.append(output.cpu())
            all_targets.append(targets.cpu())

    if args.task == "detection":
        map_score = compute_map(all_preds, all_targets)
        print(f"mAP: {map_score:.4f}")
    elif args.task == "segmentation":
        miou_score = compute_miou(all_preds, all_targets)
        print(f"mIoU: {miou_score:.4f}")
    elif args.task == "tracking":
        mota, idf1 = compute_mota(all_preds, all_targets)
        print(f"MOTA: {mota:.4f} | IDF1: {idf1:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="coco")
    parser.add_argument("--task", type=str, choices=["detection", "segmentation", "tracking"], default="detection")
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()
    evaluate(args)
