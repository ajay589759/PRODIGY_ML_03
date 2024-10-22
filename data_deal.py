import cv2 as cv
import os
import numpy as np

import random
import pickle

import time

start_time = time.time()

data_dir = './data'
batch_save_path = './batch_files'

os.makedirs(batch_save_path, exist_ok=True)

all_data_files = os.listdir(os.path.join(data_dir, 'train/'))

random.shuffle(all_data_files)

all_train_files = all_data_files[:20000]
all_test_files = all_data_files[20000:]

train_data = []
train_label = []
train_filenames = []

test_data = []
test_label = []
test_filenames = []

for each in all_train_files:
    img = cv.imread(os.path.join(data_dir,'train/',each),1)
    resized_img = cv.resize(img, (100,100))

    img_data = np.array(resized_img)
    train_data.append(img_data)
    if 'cat' in each:
        train_label.append(0)
    elif 'dog' in each:
        train_label.append(1)
    else:
        raise Exception('%s is wrong train file'%(each))
    train_filenames.append(each)

for each in all_test_files:
    img = cv.imread(os.path.join(data_dir,'train/',each), 1)
    resized_img = cv.resize(img, (100,100))

    img_data = np.array(resized_img)
    test_data.append(img_data)
    if 'cat' in each:
        test_label.append(0)
    elif 'dog' in each:
        test_label.append(1)
    else:
        raise Exception('%s is wrong test file'%(each))
    test_filenames.append(each)

print(len(train_data), len(test_data))

start = 0
end = 200
for num in range(1, 101):
    batch_data = train_data[start: end]
    batch_label = train_label[start: end]
    batch_filenames = train_filenames[start: end]
    batch_name = 'training batch {} of 15'.format(num)

    all_data = {
        'data':batch_data,
        'label':batch_label,
        'filenames':batch_filenames,
        'name':batch_name
    }

    with open(os.path.join(batch_save_path, 'train_batch_{}'.format(num)), 'wb') as f:
        pickle.dump(all_data, f)

    start += 200
    end += 200

all_test_data = {
    'data':test_data,
    'label':test_label,
    'filenames':test_filenames,
    'name':'test batch 1 of 1'
}

with open(os.path.join(batch_save_path, 'test_batch'), 'wb') as f:
    pickle.dump(all_test_data, f)


end_time = time.time()
print(''.format(end_time - start_time))