import pickle
import torch
from torchvision.models import efficientnet_b1
import csv
from tqdm import tqdm

with open('calib_loader.pkl', 'rb') as f:
    calib_dataloader = pickle.load(f)

model = efficientnet_b1(pretrained=True).to(device='cuda')
model.eval()
predicted_labels = []
predicted_values = []
true_labels = []
for images, labels in tqdm(calib_dataloader):
    images = images.to(device='cuda')
    with torch.no_grad():
        outputs = model(images)

    softmax_outputs = torch.softmax(outputs, dim=1)
    value, predicted = torch.max(softmax_outputs, 1)
    predicted_labels.extend(predicted.cpu().tolist())
    predicted_values.extend(value.cpu().tolist())
    true_labels.extend(labels.tolist())


file_path = "calib_results.csv"

pairs = zip(predicted_values, predicted_labels, true_labels)

with open(file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["val", "pred", "true"])
    for val, predicted, true in pairs:
        writer.writerow([val, predicted, true])

print("Saved in: ", file_path)

