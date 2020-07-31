import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import TrainingData, TestData
from model import PolynomialModel

from multiprocessing import Pool


def log_preds(order, epoch, x, y, pred_file):
    for a, b in zip(x, y):
        pred_file.write(
            "{}, {}, {}, {}\n".format(order, epoch, a.item(), b.item())
        )


def train_and_log_models(order):
    epochs = 500000
    model = PolynomialModel(order=order + 1)
    log_file_format = "./logs/run_log{}.csv"
    log_file = open(log_file_format.format(model.order), "w")
    log_file.write("order, epoch, train_err, test_err\n")
    training = TrainingData()
    test = TestData()
    train_data_loader = DataLoader(
        training, batch_size=4, shuffle=True
    )
    test_x, test_y = test.get_all_data()
    criterion = torch.nn.MSELoss(reduction="sum")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    for epoch in range(epochs):
        train_err = 0
        for train_x, train_y in iter(train_data_loader):
            train_pred = model(train_x)
            loss = criterion(train_pred, train_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_err += loss.item()
        if epoch % 1000 == 999:
            print("Epoch {}, {} done".format(epoch + 1, model.order))
        with torch.no_grad():
            pred = model(test_x)
            loss = criterion(pred, test_y)
            log_file.write(
                "{}, {}, {}, {}\n".format(
                    model.order,
                    epoch,
                    train_err,
                    loss.item(),
                )
            )
    log_file.close()
    pred_file = open("./logs/preds_{}.csv".format(model.order), "w")
    pred_file.write("order, epoch, x, y\n")
    pred = model(test_x)
    log_preds(model.order, epoch, test_x, pred, pred_file)
    pred_file.close()
        

def main():
    with Pool() as p:
        p.map(train_and_log_models, iter(range(100)))


if __name__ == "__main__":
    main()

