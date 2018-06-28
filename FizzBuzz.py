import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data

# hyperparameters
HIDDEN_UNITS = 1000
LEARNING_RATE = 0.001
EPOCH = 250
BATCH_SIZE = 15


# Using 2 hidden layers
class DNN(nn.Module):
    def __init__(self):
        super(DNN, self).__init__()
        self.linear_1 = nn.Linear(10, HIDDEN_UNITS)
        self.linear_2 = nn.Linear(HIDDEN_UNITS, 4)

    def forward(self, x):
        x = F.relu(self.linear_1(x))
        # using softmax to output the probabilities of each class
        x = F.softmax(self.linear_2(x), dim=0)

        return x


# encode the number into binary form list
#   lower bit --------> higher bit
#   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
def encode_data(x):
    code = np.zeros(10)
    x_bin = list(bin(x))[::-1]
    x_len = len(x_bin) - 2
    for i in range(10):
        if i < x_len:
            code[i] = float(x_bin[i])

    return code


# the answer we want to get:
#   0 -- nothing  (can't divided by 3 and 5)
#   1 -- Fizz     (can be divided by 3 but not 5)
#   2 -- Buzz     (can be divided by 5 but not 3)
#   3 -- FizzBuzz (can be divided by 3 and 5)
def encode_label(x):
    if (x % 3 == 0 and x % 5 == 0):
        return 3
    elif (x % 3 != 0 and x % 5 == 0):
        return 2
    elif (x % 3 == 0 and x % 5 != 0):
        return 1
    else:
        return 0


# use first 100 numbers as test set, and other 900 numbers as traing set
def genarate_data(device):
    train_data = []
    train_label = []
    test_data = []
    test_label = []
    for i in range(1000):
        if i < 100:
            test_data.append(encode_data(i))
            test_label.append(encode_label(i))
        else:
            train_data.append(encode_data(i))
            train_label.append(encode_label(i))

    test_data = torch.Tensor(test_data).to(device)
    test_label = torch.LongTensor(test_label).to(device)
    train_data = torch.Tensor(train_data).to(device)
    train_label = torch.LongTensor(train_label).to(device)

    return test_data, test_label, train_data, train_label


def main():
    # we want to use GPU if we have one
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_data, test_label, train_data, train_label = genarate_data(device)

    # prepare the data loader
    training_set = Data.TensorDataset(train_data,
                                      train_label)

    training_loader = Data.DataLoader(dataset=training_set,
                                      batch_size=BATCH_SIZE,
                                      shuffle=True)

    testing_set = Data.TensorDataset(test_data,
                                     test_label)

    testing_loader = Data.DataLoader(dataset=testing_set,
                                     batch_size=BATCH_SIZE,
                                     shuffle=False)

    model = DNN().to(device)

    # using crossentropy loss on classification problem
    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCH):
        correct_train = 0
        total_train = 0
        for (data, label) in training_loader:
            pred_label = model(data)

            loss = criterion(pred_label, label)
            optim.zero_grad()
            loss.backward()
            optim.step()

            _, answer = torch.max(pred_label.data, 1)
            total_train += label.size(0)
            correct_train += (answer == label).sum()

        print('Epoch {:3d} Accuracy on training data: {}% ({}/{})'
              .format(epoch, (100 * correct_train / total_train), correct_train, total_train))

        # pytorch 0.4 feature, not calculate grad on test set
        with torch.no_grad():
            correct_test = 0
            total_test = 0
            for (data, label) in testing_loader:
                pred_label = model(data)
                _, answer = torch.max(pred_label.data, 1)
                total_test += label.size(0)
                correct_test += (answer == label).sum()

            print('          Accuracy on testing data: {}% ({}/{})'
                  .format((100 * correct_test / total_test), correct_test, total_test))


if __name__ == '__main__':
    main()
