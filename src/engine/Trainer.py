import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np

from src.utils.helpers import get_grid_coords, decay_exponential, calculate_QE_TE_Purity
from src.losses.ViTloss import ViTLoss 
from src.losses.SOMloss import SomLoss
from src.models.ViTGSOM import ViTGSOM

class ViTGSOMTrainer:
    """
    Trainer class for ViT-GSOM.
    """
    def __init__(self,
                 model: ViTGSOM,
                 config: dict,
                 device: torch.device,
                 loader: torch.utils.data.DataLoader):
        """
        Constructor for ViT-GSOM Trainer
        :param model: ViT-GSOM Autoencoder
        :param config: Configuration dictionary
        :param device: The torch device for training
        :param loader: The data loader
        """
        self.model = model
        self.config = config
        self.device = device
        self.loader = loader

        self.criterionViT = ViTLoss()
        self.criterionSOM = SomLoss()

    def train_phase_1(self) -> dict[str: list[float]]:
        """
        Method for Phase 1 training.
        :return: History dictionary
        """
        print("\n--- Starting Phase 1: ViT Pre-training ---")

        self.model.som_weights.requires_grad = False
        for param in self.model.encoder.parameters(): param.requires_grad = True
        for param in self.model.decoder.parameters(): param.requires_grad = True

        paramsViT = list(self.model.encoder.parameters()) + list(self.model.decoder.parameters())
        optimizerViT = optim.AdamW(paramsViT, lr=self.config['lr'])
        scheduler = CosineAnnealingLR(optimizerViT, T_max=self.config['epochs_phase1'])

        history = {'MSE': []}

        for epoch in range(self.config['epochs_phase1']):
            running_mse = 0.0

            for images, _ in self.loader:
                images = images.to(self.device)

                reconstructed, latent = self.model(images)
                som_weights = self.model.get_som_weights()

                l_nn = self.criterionViT(images, reconstructed)

                optimizerViT.zero_grad()
                l_nn.backward()
                optimizerViT.step()
                running_mse += l_nn.item()

            scheduler.step()
            history['MSE'].append(running_mse / len(self.loader))

            print(f"Epoch {epoch + 1}/{self.config['epochs_phase1']}: MSE Loss {running_mse / len(self.loader):.8f}")

        return history


    def train_phase_2(self) -> dict[str: list[float]]:
        """
        Method for Phase 2 training.
        :return: History dictionary
        """
        print("\n--- Starting Phase 2: SOM Growth ---")

        self.model.som_weights.requires_grad = True
        for param in self.model.encoder.parameters(): param.requires_grad = False
        for param in self.model.decoder.parameters(): param.requires_grad = False

        paramsSOM = self.model.som_weights
        optimizerSOM = optim.AdamW([paramsSOM], lr=self.config['lr'])
        scheduler = CosineAnnealingLR(optimizerSOM, T_max=self.config['epochs_phase2'])

        sigma_start = self.model.get_sigma()
        sigma_end = 0.3
        beta = (sigma_end / sigma_start) ** (1 / self.config['grow_after_epochs'])
        epochs_since_reset = 0

        rows, cols = self.model.get_som_shape() 
        grid_coords = get_grid_coords(rows, cols, self.device)

        history = {'som': [], 'purity': [], 'QE': [], 'TE': []}

        unique_labels = set()

        for epoch in range(self.config['epochs_phase2']):

            running_som = 0.0

            sigma_t = decay_exponential(sigma_start, beta, epochs_since_reset)

            for images, labels in self.loader:
                images = images.to(self.device)
                unique_labels.update(labels.tolist())

                reconstructed, latent = self.model(images)
                som_weights = self.model.get_som_weights()

                l_som = self.criterionSOM(latent, som_weights, grid_coords, sigma_t)

                optimizerSOM.zero_grad()
                l_som.backward()
                optimizerSOM.step()

                running_som += l_som.item()

            # updating learning rule through CosineAnnealingLR
            scheduler.step()

            metrics = calculate_QE_TE_Purity(self.model, self.loader, self.device)

            avg_som = running_som / len(self.loader)

            history['som'].append(avg_som)
            history['purity'].append(metrics["Purity"])
            history['QE'].append(metrics["QE"])
            history['TE'].append(metrics["TE"])

            print(
                f"Epoch {epoch + 1}/{self.config['epochs_phase2']} | Sigma: {sigma_t:.3f} | SOM: {avg_som:.8f} | QE: {metrics['QE']:.4f} | TE: {metrics['TE']:.4f} | Purity: {metrics['Purity']:.5f}")

            if epoch > 0 and (epoch + 1) % self.config['grow_after_epochs'] == 0:

                if metrics["Purity"] > self.config['stop_growth_purity']:
                    print(f"Purity is high enough: {metrics['Purity']}, entering Phase 3 - Fine-tuning")
                    break

                epochs_since_reset += 1
                self.model.start_growth(self.loader, self.device)
                paramsSOM = self.model.get_som_weights()
                optimizerSOM = optim.AdamW([paramsSOM], lr=self.config['lr'])

                for param_group in optimizerSOM.param_groups: param_group['initial_lr'] = self.config['lr']

                scheduler = CosineAnnealingLR(optimizerSOM, T_max=self.config['epochs_phase2'], last_epoch=epoch)
                grid_coords = get_grid_coords(self.model.current_row_num, self.model.current_col_num, self.device)

                sigma_start = self.model.get_sigma()
                sigma_start = max(sigma_start, 2.0)
                beta = (sigma_end / sigma_start) ** (1 / max(1, self.config['grow_after_epochs']))
                epochs_since_reset = 0

            else:
                epochs_since_reset += 1

        return history

    def train_phase_3(self) -> dict[str: list[float]]:
        """
        Method for Phase 3 training.
        :return: History dictionary
        """
        print("\n--- Starting Phase 3: SOM Fine-Tuning ---")

        self.model.som_weights.requires_grad = True
        for param in self.model.encoder.parameters():
            param.requires_grad = False
        for param in self.model.decoder.parameters():
            param.requires_grad = False

        paramsSOM = self.model.get_som_weights()
        optimizerSOM = optim.AdamW([paramsSOM], lr=self.config['lr'])
        scheduler = CosineAnnealingLR(optimizerSOM, T_max=self.config['epochs_phase3'], last_epoch=-1)

        sigma_start = 2
        sigma_end = 0.01
        beta = (sigma_end / sigma_start) ** (1 / self.config['epochs_phase3'])

        grid_coords = get_grid_coords(self.model.current_row_num, self.model.current_col_num, self.device)

        history = {'som': [], 'purity': [], 'QE': [], 'TE': []}
        best_purity = 0.0

        for epoch in range(self.config['epochs_phase3']):

            running_som = 0.0

            sigma_t = decay_exponential(sigma_start, beta, epoch)

            for images, _ in self.loader:
                images = images.to(self.device)

                reconstructed, latent = self.model(images)
                som_weights = self.model.get_som_weights()

                l_som = self.criterionSOM(latent, som_weights, grid_coords, sigma_t)

                optimizerSOM.zero_grad()
                l_som.backward()
                optimizerSOM.step()

                running_som += l_som.item()

            # updating learning rule through CosineAnnealingLR
            scheduler.step()

            metrics = calculate_QE_TE_Purity(self.model, self.loader, self.device)

            avg_som = running_som / len(self.loader)

            history['som'].append(avg_som)
            history['purity'].append(metrics["Purity"])
            history['QE'].append(metrics["QE"])
            history['TE'].append(metrics["TE"])

            if metrics["Purity"] > best_purity: best_purity = metrics["Purity"]

            print(f"Epoch {epoch + 1}/{self.config['epochs_phase3']} | Sigma: {sigma_t:.3f} | SOM: {avg_som:.8f} | QE: {metrics['QE']:.4f} | TE: {metrics['TE']:.4f} | Purity: {metrics['Purity']:.5f}")

        return history, best_purity


    def train_full_pipeline(self) -> dict[str: dict[str, list[float]]]:
        """
        Method that executes Phase 1, 2, and 3 sequentially.
        :return: dictionary of all histories
        """
        print("==========================================")
        print("  INITIATING FULL VIT-GSOM PIPELINE  ")
        print("==========================================")

        hist_p1 = self.train_phase_1()
        hist_p2 = self.train_phase_2()
        hist_p3, purity = self.train_phase_3()

        print("\n==========================================")
        print("  PIPELINE COMPLETE  ")
        print(f"  Final Grid Size: {self.model.current_row_num} x {self.model.current_col_num}")
        print(f"  Best Purity: {purity:.5f}")
        print("==========================================")

        return {
            "phase_1": hist_p1,
            "phase_2": hist_p2,
            "phase_3": hist_p3
        }