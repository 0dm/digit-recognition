import argparse
from time import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch import nn, optim
from torchvision import datasets, transforms


""" Workshop Exercise 1: Fix the ReLU activation function """


class ReLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        zero_tensor = torch.zeros_like(x)
        # TODO: Fix the ReLU activation function (hint: use torch.max)


""" Workshop Exercise 2: How many neurons should we have in the input layer? """
INPUT_N = 1

""" Workshop Exercise 3: How many neurons should we have in the output layer? """
OUTPUT_N = 1

""" Workshop Exercise 4: How many times should we go through the training data? """
EPOCHS = 0

"""
additional exercises (see extract.py):
5. Save more images from the training data
6. Name the images according to their labels, preferably using the model's predictions
"""


def view_classify(img: torch.Tensor, ps: torch.Tensor) -> None:
    """Function for viewing an image and it's predicted classes.

    Arguments:
        img (tensor): image tensor
        ps (tensor): predicted class probabilities
    """

    ps = ps.data.numpy().squeeze()
    fig, (ax1, ax2) = plt.subplots(figsize=(6, 9), ncols=2)
    ax1.imshow(img.resize_(1, 28, 28).numpy().squeeze())
    ax1.axis("off")
    ax2.barh(np.arange(OUTPUT_N), ps)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(OUTPUT_N))
    ax2.set_yticklabels(np.arange(OUTPUT_N))
    ax2.set_title("Class Probability")
    ax2.set_xlim(0, 1.1)
    plt.tight_layout()


def load_data(train: bool = True) -> torch.utils.data.DataLoader:
    """Load the MNIST dataset.

    Arguments:
        train (bool): whether to load the training or testing set

    Returns:
        dataloader (DataLoader): the dataloader for the specified set
    """

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )

    dataset = datasets.MNIST(
        "trainset" if train else "testset",
        download=True,
        train=train,
        transform=transform,
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
    return dataloader


def train_model(
    model: nn.Sequential,
    trainloader: torch.utils.data.DataLoader,
    epochs: int = EPOCHS,
    lr: float = 0.003,
    momentum: float = 0.9,
) -> None:
    """Train the model.

    Arguments:
        model (Sequential): the model to train
        trainloader (DataLoader): the dataloader for the training set
        epochs (int): the number of epochs to train the model for
        lr (float): the learning rate
        momentum (float): the momentum
    """

    criterion = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    time0 = time()
    for e in range(epochs):
        running_loss = 0
        for images, labels in trainloader:
            images = images.view(images.shape[0], -1)
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print("Epoch {} - Training loss: {}".format(e, running_loss / len(trainloader)))

    print("\nTraining Time (in minutes) =", (time() - time0) / 60)


def test_model(model: nn.Sequential, valloader: torch.utils.data.DataLoader) -> None:
    """Test the model.

    Arguments:
        model (Sequential): the model to test
        valloader (DataLoader): the dataloader for the testing set
    """

    correct_count, all_count = 0, 0
    with torch.no_grad():
        for images, labels in valloader:
            for i in range(len(labels)):
                img = images[i].view(1, 784)
                logps = model(img)
                ps = torch.exp(logps)
                probab = list(ps.numpy()[0])
                pred_label = probab.index(max(probab))
                true_label = labels.numpy()[i]
                if true_label == pred_label:
                    correct_count += 1
                all_count += 1
    view_classify(img.view(1, 28, 28), ps)
    print("Number Of Images Tested =", all_count)
    print("\nModel Accuracy =", (correct_count / all_count))


def main() -> None:
    """Main function."""

    parser = argparse.ArgumentParser(description="MNIST Model Training and Testing")

    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--test", action="store_true", help="Test the model")
    parser.add_argument("--image_path", type=str, help="Path to the image for testing")

    args = parser.parse_args()

    # Create the model
    input_size = INPUT_N
    hidden_sizes = [128, 64]
    output_size = OUTPUT_N

    model = nn.Sequential(
        nn.Linear(input_size, hidden_sizes[0]),
        ReLU(),
        nn.Linear(hidden_sizes[0], hidden_sizes[1]),
        ReLU(),
        nn.Linear(hidden_sizes[1], output_size),
        nn.LogSoftmax(dim=1),
    )

    if args.train:
        # Load the training data and train the model
        trainloader = load_data(train=True)
        train_model(model, trainloader)

        # Save the trained model
        torch.save(model, "./mnist_model.pt")

    if args.test:
        # Load the testing data and test the model
        valloader = load_data(train=False)
        model = torch.load("./mnist_model.pt")

        if args.image_path:
            # If an image path is provided, test the model on that image
            image_path = args.image_path
            img = Image.open(image_path).convert("RGB")
            transform = transforms.Compose(
                [
                    transforms.Resize((28, 28)),
                    transforms.Grayscale(num_output_channels=1),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,)),
                ]
            )
            img = transform(img).unsqueeze(0)
            img = img.view(1, -1)
            with torch.no_grad():
                logps = model(img)
            ps = torch.exp(logps)
            probab = list(ps.numpy()[0])
            predicted_digit = probab.index(max(probab))
            view_classify(img.view(1, 28, 28), ps)
            print("Predicted Digit =", predicted_digit)
        else:
            # Test the model on the entire test set
            test_model(model, valloader)
        plt.show()


if __name__ == "__main__":
    main()
