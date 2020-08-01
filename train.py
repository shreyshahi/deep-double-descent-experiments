import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import TrainingData, TestData
from model import PolynomialModel

import time

def log_preds(order, epoch, x, y, pred_file):
    for a, b in zip(x, y):
        pred_file.write(
            "{}, {}, {}, {}\n".format(order, epoch, a.item(), b.item())
        )


def train_and_log_models(epochs, models, training, test, log_file):
    test_x, test_y = test.get_all_data()
    train_x, train_y = training.get_all_data()
    criterion = torch.nn.MSELoss(reduction="mean")
    optimizers = [
        torch.optim.Adam(model.parameters(), lr=1e-3)
        for model in models
    ]
    start_time = time.time()
    for epoch in range(epochs):
        train_err = {
            model.order: 0
            for model in models
        }
        for model, optimizer in zip(models, optimizers):
            train_pred = model(train_x)
            loss = criterion(train_pred, train_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_err[model.order] += loss.item()
        #print("Epoch {} done".format(epoch))
        with torch.no_grad():
            for model in models:
                pred = model(test_x)
                loss = criterion(pred, test_y)
                log_file.write(
                    "{}, {}, {}, {}\n".format(
                        model.order,
                        epoch,
                        train_err[model.order],
                        loss.item(),
                    )
                )
        if epoch % 1000 == 999:
            elapsed_time = time.time() - start_time
            print("Order {} took {} seconds".format(model.order, elapsed_time))
            start_time = time.time()
        

def main():
    models = [
        PolynomialModel(order=80),
    ]

    epochs = 10000

    training_data = TrainingData()
    test_data = TestData()

    with open("./run_log.csv", "w") as f1:
        f1.write("order, epoch, train_err, test_err\n")
        train_and_log_models(
            epochs,
            models,
            training_data,
            test_data,
            f1,
        )

if __name__ == "__main__":
    main()

