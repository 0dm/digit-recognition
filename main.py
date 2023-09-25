import argparse
import torch
from torch import nn, optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from time import time
from PIL import Image


def view_classify(img, ps):
    ps = ps.data.numpy().squeeze()
    fig, (ax1, ax2) = plt.subplots(figsize=(6, 9), ncols=2)
    ax1.imshow(img.resize_(1, 28, 28).numpy().squeeze())
    ax1.axis("off")
    ax2.barh(np.arange(10), ps)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(10))
    ax2.set_yticklabels(np.arange(10))
    ax2.set_title("Class Probability")
    ax2.set_xlim(0, 1.1)
    plt.tight_layout()


def load_data(train=True):
    # Function to load MNIST dataset
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


def train_model(model, trainloader, epochs=15, lr=0.003, momentum=0.9):
    # Function to train the model
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


def test_model(model, valloader):
    # Function to test the model
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

    print("Number Of Images Tested =", all_count)
    print("\nModel Accuracy =", (correct_count / all_count))


def main():
    parser = argparse.ArgumentParser(description="MNIST Model Training and Testing")

    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--test", action="store_true", help="Test the model")
    parser.add_argument("--image_path", type=str, help="Path to the image for testing")

    args = parser.parse_args()

    # Create the model
    input_size = 784
    hidden_sizes = [128, 64]
    output_size = 10

    model = nn.Sequential(
        nn.Linear(input_size, hidden_sizes[0]),
        nn.ReLU(),
        nn.Linear(hidden_sizes[0], hidden_sizes[1]),
        nn.ReLU(),
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
            print("Predicted Digit =", predicted_digit)
        else:
            # Test the model on the entire test set
            test_model(model, valloader)


if __name__ == "__main__":
    main()
