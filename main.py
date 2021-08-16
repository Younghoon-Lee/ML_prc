import torch
from torchvision import datasets, transforms

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

mnist_train = datasets.MNIST(
    root="./data/", train=True, transform=transforms.ToTensor(), download=True
)
mnist_test = datasets.MNIST(
    root="./data/", train=False, transform=transforms.ToTensor(), download=True
)
BATCH_SIZE = 256
train_iter = torch.utils.data.DataLoader(
    mnist_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=1
)
test_iter = torch.utils.data.DataLoader(
    mnist_test, batch_size=BATCH_SIZE, shuffle=True, num_workers=1
)
print("Done")
print(train_iter)
print(test_iter)
