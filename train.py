import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import TrainingData, TestData
from model import PolynomialModel


def log_preds(order, epoch, x, y, pred_file):
    for a, b in zip(x, y):
        pred_file.write(
            "{}, {}, {}, {}\n".format(order, epoch, a.item(), b.item())
        )


def train_and_log_models(epochs, models, training, test, log_file, pred_file):
    train_data_loader = DataLoader(
        training, batch_size=100, shuffle=True
    )
    test_x, test_y = test.get_all_data()
    criterion = torch.nn.MSELoss(reduction="sum")
    optimizers = [
        torch.optim.Adam(model.parameters(), lr=0.0001)
        for model in models
    ]
    for epoch in range(epochs):
        train_err = {
            model.order: 0
            for model in models
        }
        for train_x, train_y in iter(train_data_loader):
            for model, optimizer in zip(models, optimizers):
                train_pred = model(train_x)
                loss = criterion(train_pred, train_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_err[model.order] += loss.item()
        print("Epoch {} done".format(epoch))
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
                log_preds(model.order, epoch, test_x, pred, pred_file)
        print("Epoch {} logged".format(epoch))
        

def main():
    models = [
        PolynomialModel(order=1),
        PolynomialModel(order=2),
        PolynomialModel(order=3),
    ]

    epochs = 10000

    training_data = TrainingData()
    test_data = TestData()

    with open("./run_log.csv", "w") as f1, open("./preds.csv", "w") as f2:
        f1.write("order, epoch, train_err, test_err\n")
        f2.write("order, epoch, x, y\n")
        train_and_log_models(
            epochs,
            models,
            training_data,
            test_data,
            f1,
            f2,
        )

if __name__ == "__main__":
    main()

