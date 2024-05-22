from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import datasets, transforms, models
from tqdm import tqdm
import pickle

## IMPORTANT
# The batch_size parameter in DataLoaders might raise exceptions due to differences in tensor sizes.

transform = models.EfficientNet_B0_Weights.IMAGENET1K_V1.transforms()


# load val  ImageNet-1k
all_data = datasets.ImageNet(root='./', split='val',  transform=transform)

# BEE CLASS INDEX
pred_class = 309

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
    if label == pred_class:
        bee_indices.extend(indices)

    calib_indices.extend(indices[split:])




# samplers
test_sampler = SubsetRandomSampler(test_indices)
calib_sampler = SubsetRandomSampler(calib_indices)
bee_sampler = SubsetRandomSampler(bee_indices)

# DataLoaders
test_loader = DataLoader(all_data, batch_size=64, sampler=test_sampler)
calib_loader = DataLoader(all_data, batch_size=64, sampler=calib_sampler)
bee_loader = DataLoader(all_data, batch_size=64, sampler=bee_sampler)

with open('test_loader.pkl', 'wb') as f:
    pickle.dump(test_loader, f)

with open('calib_loader.pkl', 'wb') as f:
    pickle.dump(calib_loader, f)

with open('bee_loader.pkl', 'wb') as f:
    pickle.dump(bee_loader, f)