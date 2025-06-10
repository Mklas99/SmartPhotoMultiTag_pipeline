import matplotlib.pyplot as plt

def save_loss_plot(train_loss: list, val_loss: list, path: str):
    plt.figure()
    plt.plot(train_loss, label="train")
    plt.plot(val_loss, label="val")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(path)
    plt.close()
