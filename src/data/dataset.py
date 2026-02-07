from torchvision.datasets import ImageFolder

class EmotionDataset(ImageFolder):
    def __init__(self, root, transform=None):
        super().__init__(root=root, transform=transform)

    

