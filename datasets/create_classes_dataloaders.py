from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import datasets, transforms, models
from tqdm import tqdm
import pickle

## IMPORTANT
# The batch_size parameter in DataLoaders might raise exceptions due to differences in tensor sizes.

transform = models.EfficientNet_B0_Weights.IMAGENET1K_V1.transforms()


# load val  ImageNet-1k
all_data = datasets.ImageNet(root='./', split='val',  transform=transform)

"""
# Divide val by indexes for each class
class_indices = {}
for i in tqdm(range(len(all_data))):
    _, label = all_data[i]
    if label not in class_indices:
        class_indices[label] = []
    class_indices[label].append(i)

with open('class_indices.pkl', 'wb') as f:
    pickle.dump(class_indices, f) """

with open('class_indices.pkl', 'rb') as f:
    class_indices = pickle.load(f)
k = 0
for label, indices in tqdm(class_indices.items()):
    my_class_indices = []
    my_class_indices.extend(indices)
    my_class_sampler = SubsetRandomSampler(my_class_indices)
    my_class_loader = DataLoader(all_data, batch_size=64, sampler=my_class_sampler)

    with open('classes1/'+str(k)+'_loader.pkl', 'wb') as f:
        pickle.dump(my_class_loader, f)

    k += 1