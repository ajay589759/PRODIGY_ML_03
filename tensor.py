import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2 as cv
import os
import pickle

''' Global parameters '''
IMAGE_SIZE = 100
LEARNING_RATE = 1e-4
TRAIN_STEP = 10000
TRAIN_SIZE = 100
TEST_STEP = 100
TEST_SIZE = 50

IS_TRAIN = True

SAVE_PATH = './model/'

data_dir = './batch_files'
pic_path = './data/test1'


def load_data(filename):
    '''Load image data from batch files'''
    with open(filename, 'rb') as f:
        data = pickle.load(f, encoding='iso-8859-1')
        return data['data'], data['label'], data['filenames']


class InputData:
    def __init__(self, filenames, need_shuffle):
        all_data = []
        all_labels = []
        all_names = []
        for file in filenames:
            data, labels, filename = load_data(file)
            all_data.append(data)
            all_labels.append(labels)
            all_names += filename

        self._data = np.vstack(all_data)
        self._labels = np.hstack(all_labels)
        print(self._data.shape)
        print(self._labels.shape)

        self._filenames = all_names

        self._num_examples = self._data.shape[0]
        self._need_shuffle = need_shuffle
        self._indicator = 0
        if self._indicator:
            self._shuffle_data()

    def _shuffle_data(self):
        # Shuffle the data
        p = np.random.permutation(self._num_examples)
        self._data = self._data[p]
        self._labels = self._labels[p]

    def next_batch(self, batch_size):
        '''Return a batch of data'''
        end_indicator = self._indicator + batch_size
        if end_indicator > self._num_examples:
            if self._need_shuffle:
                self._shuffle_data()
                self._indicator = 0
                end_indicator = batch_size
            else:
                raise Exception('No more examples')
        if end_indicator > self._num_examples:
            raise Exception('Batch size is larger than the number of examples')
        batch_data = self._data[self._indicator:end_indicator]
        batch_labels = self._labels[self._indicator:end_indicator]
        batch_filenames = self._filenames[self._indicator:end_indicator]
        self._indicator = end_indicator
        return batch_data, batch_labels, batch_filenames


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv_net = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc_net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 6 * 6, 512),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        x = self.conv_net(x)
        x = self.fc_net(x)
        return x


class MyTensor:
    def __init__(self):
        train_filenames = [os.path.join(data_dir, 'train_batch_%d' % i) for i in range(1, 101)]
        test_filenames = [os.path.join(data_dir, 'test_batch')]
        self.batch_train_data = InputData(train_filenames, True)
        self.batch_test_data = InputData(test_filenames, True)

        self.model = MyModel()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)

    def train(self):
        self.model.train()
        for i in range(TRAIN_STEP):
            train_data, train_label, _ = self.batch_train_data.next_batch(TRAIN_SIZE)
            train_data = torch.tensor(train_data, dtype=torch.float32).permute(0, 3, 1, 2)
            train_label = torch.tensor(train_label, dtype=torch.long)

            self.optimizer.zero_grad()
            output = self.model(train_data)
            loss = self.criterion(output, train_label)
            loss.backward()
            self.optimizer.step()

            if (i + 1) % 100 == 0:
                print(f'Step [{i + 1}/{TRAIN_STEP}], Loss: {loss.item():.4f}')

            if (i + 1) % 1000 == 0:
                self.evaluate()

        torch.save(self.model.state_dict(), SAVE_PATH + 'my_model.pth')

    def evaluate(self):
        self.model.eval()
        test_acc = 0
        with torch.no_grad():
            for _ in range(TEST_STEP):
                test_data, test_label, _ = self.batch_test_data.next_batch(TEST_SIZE)
                test_data = torch.tensor(test_data, dtype=torch.float32).permute(0, 3, 1, 2)
                test_label = torch.tensor(test_label, dtype=torch.long)

                output = self.model(test_data)
                _, predicted = torch.max(output.data, 1)
                test_acc += (predicted == test_label).sum().item() / TEST_SIZE

        print(f'Test Accuracy: {test_acc / TEST_STEP:.4f}')

    def classify(self):
        self.model.load_state_dict(torch.load(SAVE_PATH + 'my_model.pth'))
        self.model.eval()
        with torch.no_grad():
            for filename in os.listdir(pic_path):
                img_data = self.get_img_data(os.path.join(pic_path, filename))
                img_data = torch.tensor(img_data, dtype=torch.float32).unsqueeze(0).permute(0, 3, 1, 2)
                output = self.model(img_data)
                _, predicted = torch.max(output.data, 1)
                print(f'Image: {filename}, Predicted: {predicted.item()}')

    def get_img_data(self, img_name):
        img = cv.imread(img_name)
        resized_img = cv.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        img_data = np.array(resized_img)
        return img_data


if __name__ == '__main__':
    mytensor = MyTensor()
    if IS_TRAIN:
        mytensor.train()
    else:
        mytensor.classify()
    print('Hello world')
