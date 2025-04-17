# Train pipeline for ST-NPP + QAL with proxy codec and task network
from models.combined_model import CombinedModel
from utils.model_utils import save_checkpoint
from utils.common_utils import load_data, train_loop

def main():
    model = CombinedModel()
    train_loader, val_loader = load_data()
    train_loop(model, train_loader, val_loader, epochs=20)
    save_checkpoint(model)

if __name__ == "__main__":
    main()
