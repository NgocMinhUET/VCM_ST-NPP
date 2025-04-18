import argparse
import torch
from torch.utils.data import DataLoader
from models.combined_model import CombinedModel
from utils.model_utils import save_checkpoint, load_checkpoint
from utils.data_utils import get_dataloader
from utils.loss_utils import compute_total_loss

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CombinedModel().to(device)
    
    if args.resume:
        load_checkpoint(model, args.resume)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    train_loader = get_dataloader(args.dataset, batch_size=args.batch_size, split='train')

    model.train()
    for epoch in range(args.epochs):
        total_loss = 0
        for batch in train_loader:
            inputs, targets, qp = batch['input'].to(device), batch['target'].to(device), batch['qp'].to(device)

            optimizer.zero_grad()
            output = model(inputs, qp)
            loss = compute_total_loss(output, targets, inputs, qp)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        print(f"[Epoch {epoch+1}/{args.epochs}] Loss: {total_loss / len(train_loader):.4f}")
        save_checkpoint(model, f"{args.checkpoint_dir}/stnpp_qal_epoch{epoch+1}.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="coco", help="Dataset name")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    train(args)
