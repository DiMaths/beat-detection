import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.io import wavfile
from tqdm import tqdm
import os
import librosa
from sklearn.metrics import accuracy_score, f1_score, recall_score


class CNN_Model(nn.Module):
    """
    Creates a CNN model according to paper 
    IMPROVED MUSICAL ONSET DETECTION
    WITH CONVOLUTIONAL NEURAL NETWORKS 
    By Jan Schlüter and Sebastian Böck
    """
    def __init__(self):

        super(CNN_Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=(3, 7))
        self.conv2 = nn.Conv2d(10, 20, kernel_size=(3,3))
        self.fc1 = nn.Linear(20 * 8 * 7, 256)
        self.fc2 = nn.Linear(256, 124)
        self.fc3 = nn.Linear(124, 1)
        self.flat = nn.Flatten()

    def forward(self, x, istraining=False):
        x = F.max_pool2d(F.relu(self.conv1(x)), (3, 1))
        x = F.max_pool2d(F.relu(self.conv2(x)), (3, 1))

        x = F.dropout(self.flat(x),p=0.25,training=istraining)

        # Fully connected layers with ReLU activations
        x = F.dropout(F.relu(self.fc1(x)),p=0.25,training=istraining)  # Output: (256)
        x =  F.dropout(F.relu(self.fc2(x)),p=0.25,training=istraining)  # Output: (120)

        # Sigmoid output layer
        x = self.fc3(x)  # Output: (1)

        x = torch.sigmoid(x)  # Output: (1)
        
        return x
    

class CNN_dataset(torch.utils.data.Dataset):
    """
    Creates a dataset according to paper 
    IMPROVED MUSICAL ONSET DETECTION
    WITH CONVOLUTIONAL NEURAL NETWORKS 
    By Jan Schlüter and Sebastian Böck
    @param infiles: list - list of files for
    @param GT: dict - ways to the files
    @param is_training: bool - True - training, False - testing (no ground truth)
    """
    def __init__(self, infiles, GT, options, training = True):
        self.GT = GT
        self.is_training = training
        self.spectrogram_fps = options.spectrogram_fps
        data_df = [np.zeros((3,80,7),dtype=float)] # padding of 7
        if self.is_training:
            labels_df = []

        for filename in tqdm(infiles):
            if self.is_training:
                data, label = self.get_np_data(filename)
                labels_df.append(label)
            else:
                data = self.get_np_data(filename)
            data_df.append(data)

        data_df.append(np.zeros((3,80,7),dtype=float)) # padding of 7

        self.data_df = np.concatenate(data_df,axis=2)
        if self.is_training: 
            self.labels_df = np.concatenate(labels_df,axis=0).reshape(-1,1)
            print(self.labels_df.shape)


    def __len__(self):
        return self.data_df.shape[2]-14

    def __getitem__(self, idx: int):
        if self.is_training:
            label = torch.Tensor(self.labels_df[idx])
            item = torch.Tensor(self.data_df[:,:,idx:idx+15])
            return item, label
        else:
            item = torch.Tensor(self.data_df[:,:,idx:idx+15])
            return item
    
    def get_np_data(self,filename):
        sample_rate, signal = wavfile.read(filename)

        # convert from integer to float
        if signal.dtype.kind == 'i':
            signal = signal / np.iinfo(signal.dtype).max

        # convert from stereo to mono (just in case)
        if signal.ndim == 2:
            signal = signal.mean(axis=-1)

        hop_length_ = sample_rate // self.spectrogram_fps

        base_name_with_extension = os.path.basename(filename)
        base_name = os.path.splitext(base_name_with_extension)[0]

        melspects_3 = []
        
        n_ffts = [1024,2048,4096]
        for n_fft in n_ffts:

            spect_ = librosa.stft(signal, n_fft=n_fft, hop_length=hop_length_, window='hann')

            # Only keep the magnitude
            magspect_ = np.abs(spect_)

            # compute a mel spectrogram
            melspect_ = librosa.feature.melspectrogram(S=magspect_, sr=sample_rate, n_mels=80, fmin=27.5, fmax=16000)
            
            # compress magnitudes logarithmically
            melspect_ = np.log1p(100 * melspect_)
            melspects_3.append(melspect_)


        melspects_3 = np.array(melspects_3)

        if self.is_training:
            onsets_ground_t = np.around((np.array(self.GT[base_name]["onsets"])*100)).astype(int)

            # basic deletion of 1st and last element
            if onsets_ground_t[0] < 1:
                onsets_ground_t = onsets_ground_t[1:]
            if onsets_ground_t[-1] > melspects_3.shape[2]-1:
                onsets_ground_t = onsets_ground_t[:-1]

            ground_truth_onsets = np.zeros(melspects_3.shape[2])
            ground_truth_onsets[onsets_ground_t] = 1

            for i in range(0,melspects_3.shape[2]-1):
                if ground_truth_onsets[i] == 1:
                    # working under assumption that onset is not fisrt or last element
                    # and that there cannot be 2 frames with onsets in a row
                    ground_truth_onsets[i-1] = 0.25
                    ground_truth_onsets[i+1] = 0.25

            return melspects_3, ground_truth_onsets
        
        return melspects_3



# Functions (Training loop and training steps)


def training_step(network, optimizer, data, targets, loss_fn):
    optimizer.zero_grad()
    output = network(data,istraining=True)
    labels_processed = targets.flatten().float().reshape(-1,1)
    loss = loss_fn(output, labels_processed)
    loss.backward()
    optimizer.step()
    return loss.item()

    


def print_eval(network, test_dataloader, loss_fn, threshold = 0.60) -> None:
    network.eval().to('cuda')
    running_loss = 0.
    accuracy = 0.
    counter = 0
    recall = 0.
    f1_scores = 0.
    for i, data in tqdm(enumerate(test_dataloader)):
        input, true_labels = data
        input = input.to('cuda')
        true_labels = true_labels.to('cuda')
        output = network(input,istraining=False)
        labels_processed = true_labels.flatten().float().reshape(-1,1)
        loss = loss_fn(output, labels_processed)
        running_loss += loss.item()
        
        output = output.detach().cpu().numpy()
        output = (output>threshold).astype(int)
 
        labels_processed = labels_processed.detach().cpu().numpy()
        labels_processed = (labels_processed>threshold).astype(int)
        acc = accuracy_score(labels_processed, output)
        f1 = f1_score(labels_processed, output,average="weighted")
        recall_c = recall_score(labels_processed, output,average="weighted")
        f1_scores += f1
        accuracy += acc
        recall += recall_c
        counter += 1
    print('\n')
    print('Loss =', running_loss / counter)
    print('Accuracy = ', 100 * (accuracy / counter))
    print('F1_score = ', 100 * (f1_scores / counter))
    print('Recall = ', 100 * (recall / counter))
    print('\n\n\n')


def train_model(
        network: torch.nn.Module,
        train_dataloader: torch.utils.data.dataloader.DataLoader,
        test_dataloader: torch.utils.data.dataloader.DataLoader,
        num_epochs: int,
        show_progress: bool = True):
    
    loss_fn = nn.MSELoss()
    device = torch.device("cuda")

    if not torch.cuda.is_available():
        print("CUDA IS NOT AVAILABLE")
        device = torch.device("cpu")

    print("Working on device",torch.cuda.get_device_name(0))

    
    optimizer = torch.optim.SGD(network.parameters(), lr=0.04, momentum=0.75)

    for _ in tqdm(range(num_epochs), desc="Epoch", position=0, disable= (not show_progress)):
        network.train().to('cuda')
        for _, data in tqdm(enumerate(train_dataloader), desc="Minibatch", position=1, leave=False, disable= (not show_progress)):
            inputs, targets = data
            inputs = inputs.to(device=device)
            targets = targets.to(device=device)
            training_step(network, optimizer, inputs, targets, loss_fn)
        
        print_eval(network, test_dataloader, loss_fn)
            
    return network
