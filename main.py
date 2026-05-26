import os 
import json
import torch
import argparse

from src.engine.Trainer import ViTGSOMTrainer
from src.models.ViTGSOM import ViTGSOM
from src.config import config_mnist, config_fashion_mnist, config_usps
from data.data_loaders import load_mnist, load_fashion_mnist, load_usps


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.dataset == "mnist":
        loaded_dataset = load_mnist()
        loaded_config = config_mnist
    elif args.dataset == "fashion":
        loaded_dataset = load_fashion_mnist()
        loaded_config = config_fashion_mnist
    elif args.dataset == "usps":
        loaded_dataset = load_usps()
        loaded_config = config_usps
    else:
        raise ValueError("Dataset must be 'mnist', 'fashion', or 'usps'")
    
    loader = torch.utils.data.DataLoader(dataset=loaded_dataset, batch_size=32, shuffle=True)
    
    model = ViTGSOM(config=loaded_config).to(device)
    
    trainer = ViTGSOMTrainer(model, config=loaded_config, device=device, loader=loader)
    
    histories = trainer.train_full_pipeline()

    os.makedirs("results", exist_ok=True)
    with open(f"results/history_{args.dataset}.json", "w") as f:
        json.dump(histories, f)
    print(f"Training finished. Results saved to results/history_{args.dataset}.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ViT-GSOM")
    parser.add_argument("--dataset", type=str, default="mnist", help="mnist, fashion, or usps")
    
    args = parser.parse_args()
    print(f"Loading {args.dataset} dataset.")
    
    main(args)