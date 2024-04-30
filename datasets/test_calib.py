from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import datasets, transforms
from tqdm import tqdm

## IMPORTANT
# The batch_size parameter in DataLoaders might raise exceptions due to differences in tensor sizes.

# load val  ImageNet-1k
all_data = datasets.ImageNet(root='/shared/sets/datasets/vision/ImageNet/', split='val',  transform=transforms.ToTensor())

# BEE CLASS INDEX
bee = 309

# Divide val by indexes for each class
class_indices = {}
for i in tqdm(range(len(all_data))):
    _, label = all_data[i]
    if label not in class_indices:
        class_indices[label] = []
    class_indices[label].append(i)


split_ratio = 0.5
test_indices = []
calib_indices = []
bee_indices = []
for label, indices in tqdm(class_indices.items()):
    split = int(len(indices) * split_ratio)
    test_indices.extend(indices[:split])
    if label == bee:
        bee_indices.extend((indices[:split]))

    calib_indices.extend(indices[split:])




# samplers
test_sampler = SubsetRandomSampler(test_indices)
calib_sampler = SubsetRandomSampler(calib_indices)
bee_sampler = SubsetRandomSampler(bee_indices)

# DataLoaders
test_loader = DataLoader(all_data, batch_size=64, sampler=test_sampler)
calib_loader = DataLoader(all_data, batch_size=64, sampler=calib_sampler)
bee_loader = DataLoader(all_data, batch_size=64, sampler=bee_sampler)


