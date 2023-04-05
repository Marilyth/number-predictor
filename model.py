from torchvision import datasets
from torchvision.transforms import ToTensor
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
import tqdm
from typing import *
import time


class Trainer:
    def __init__(self):
        self.generator = torch.Generator().manual_seed(42)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(1, 16, 5, 1, 2),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(16),
            torch.nn.MaxPool2d(2),

            torch.nn.Conv2d(16, 32, 5, 1, 2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),

            torch.nn.Flatten(),
            torch.nn.Linear(1568, 10)
        )

        self.model.to(self.device)

    @staticmethod
    def visualise(data_point: torch.Tensor):
        plt.imshow(data_point, cmap='gray')
        plt.show()

    def train(self, batch_size: int = 64) -> List[float]:
        """Trains the model until the test loss increases.

        Args:
            batch_size (int, optional): The amount of data points to send in at once. Defaults to 64.

        Returns:
            List[float]: The test losses.
        """
        self.train_set = datasets.MNIST("data", True, download=True, transform=ToTensor())
        self.test_set = datasets.MNIST("data", False, transform=ToTensor())
        
        sampler = DataLoader(self.train_set, batch_size, shuffle=True, generator=self.generator)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)

        losses = []
        last_state = {}

        while True:
            self.model.train()
            for batch_x, batch_y in tqdm.tqdm(sampler, total=len(sampler)):
                optimizer.zero_grad()

                logits: torch.Tensor = self.model.forward(batch_x.to(self.device))
                loss = torch.nn.functional.cross_entropy(logits, batch_y.to(self.device))
                loss.backward()
                optimizer.step()
            
            losses.append(self.test(self.test_set))
            print(losses[-1])

            gain = (losses[-2] - losses[-1]) if len(losses) > 1 else 1
            if gain < 0:
                self.model.load_state_dict(last_state)
                break

            last_state = self.model.state_dict()

        return losses
    
    @torch.no_grad()
    def test(self, test_set, batch_size: int = 1024) -> float:
        """Iterates over the test_set and calculates the loss.

        Args:
            test_set (_type_): The test data set.
            batch_size (int, optional): The amount of data points to send in at once. Defaults to 1024.

        Returns:
            float: The average loss over the test_set.
        """
        self.model.eval()

        sampler = DataLoader(test_set, batch_size, shuffle=True, generator=self.generator)
        avg_loss = 0
        batches = len(sampler)
        for batch_x, batch_y in tqdm.tqdm(sampler, total=len(sampler), desc=f"Testing"):
            logits: torch.Tensor = self.model.forward(batch_x.to(self.device))
            avg_loss += torch.nn.functional.cross_entropy(logits, batch_y.to(self.device)).item() / batches

        return avg_loss
    
    @torch.no_grad()
    def predict(self, image: torch.Tensor) -> List[float]:
        """Uses the current model to predict the provided image.

        Args:
            image (torch.Tensor): A 28x28 grayscale image to be predicted.

        Returns:
            List[float]: The probabilities for every class.
        """
        self.model.eval()
        rescaled_image = image.view(1, 1, 28, 28)
        logits: torch.Tensor = self.model.forward(rescaled_image.to(self.device))
        return torch.nn.functional.softmax(logits).to("cpu").tolist()[0]
    
    def save_model(self):
        torch.save(self.model.state_dict(), "./CNN.pkl")

    def load_model(self):
        state_dict = torch.load("./CNN.pkl")
        self.model.load_state_dict(state_dict)
