import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import TrainingData, TestData
from model import PolynomialModel

from multiprocessing import Pool
from os import path

import ray
import time


ray.init()


def log_preds(order, epoch, x, y, pred_file):
    for a, b in zip(x, y):
        pred_file.write(
            "{}, {}, {}, {}\n".format(order, epoch, a.item(), b.item())
        )

@ray.remote
def train_and_log_models(order):
    if path.exists("./logs/preds_{}.csv".format(order)):
        return
    epochs = 200000
    model = PolynomialModel(order=order + 1)
    training = TrainingData()
    train_x, train_y = training.get_all_data()
    test = TestData()
    test_x, test_y = test.get_all_data()
    criterion = torch.nn.MSELoss(reduction="sum")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    data_log = []
    print("Starting training, {}".format(order))
    start_time = time.time()
    for epoch in range(epochs):
        train_pred = model(train_x)
        loss = criterion(train_pred, train_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_err = loss.item()
        with torch.no_grad():
            pred = model(test_x)
            loss = criterion(pred, test_y)
            data_log.append((epoch, train_err, loss.item()))
        if epoch % 10000 == 9999:
            elapsed_time = time.time() - start_time
            print(
                "order: {}, epoch: {} took {} seconds". format(
                    order,
                    epoch + 1,
                    elapsed_time,
                )
            )
            start_time = time.time()
    
    log_file_format = "./logs/run_log{}.csv"
    log_file = open(log_file_format.format(model.order), "w")
    log_file.write("order, epoch, train_err, test_err\n")
    for epoch, train_err, test_err in data_log:
        log_file.write(
            "{}, {}, {}, {}\n".format(
                model.order,
                epoch,
                train_err,
                test_err,
            )
        )
    log_file.close()
    pred_file = open("./logs/preds_{}.csv".format(model.order), "w")
    pred_file.write("order, epoch, x, y\n")
    pred = model(test_x)
    log_preds(model.order, epoch, test_x, pred, pred_file)
    pred_file.close()
        

def main():
    ray.get([train_and_log_models.remote(order) for order in range(1000)])


if __name__ == "__main__":
    main()

