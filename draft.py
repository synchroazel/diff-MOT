import torch
from torch_geometric.transforms import ToDevice
from tqdm import tqdm

from model import Net
from motclass import MotDataset, build_graph
from utilities import get_best_device, save_graph

device = get_best_device()

mot20_path = "/media/dmmp/vid+backup/Data/MOT20"




def train(model, train_loader, loss_function, optimizer, epochs, device):
    model = model.to(device)
    model.train()

    pbar = tqdm(range(epochs))

    for epoch in pbar:

        epoch_loss = 0
        for i, data in enumerate(train_loader):

            data = ToDevice(device.type)(data)
            # data = data.to(device)

            # Forward pass
            pred_edges = model(data)  # Get the predicted edge labels
            gt_edges = data.y  # Get the true edge labels

            loss = loss_function(pred_edges, gt_edges)

            print("Loss computed - " + str(loss.item()))

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5)

            epoch_loss += loss.item()

        print(f'Epoch {epoch + 1}, Loss: {epoch_loss / i}')


# Hyperparameters
backbone = 'resnet50'
l_size = 128
epochs = 10
learning_rate = 0.001

model = Net(backbone, l_size, dtype=torch.float32)

loss_function = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

mo20_train_dl = MotDataset(dataset_path=mot20_path,
                           split='train',
                           subtrack_len=50,
                           linkage_window=12,
                           detections_file_folder='gt',
                           detections_file_name='gt.txt',
                           device=device,
                           dl_mode=True,
                           dtype=torch.float32)

train(model, mo20_train_dl, loss_function, optimizer, epochs, device)