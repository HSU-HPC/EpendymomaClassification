import torchvision.transforms as transforms
import utils.MILDataset as MD
from torch.utils.data import random_split
import CLAM
from utils.clam_training import train
import matplotlib.pyplot as plt


def test_clam(encoder_dim, num_classes, num_samples_per_class, bag_size, repr_inst, subtyping, std=1.0):
    transformations = transforms.Compose([transforms.ToTensor(), lambda x: x.float(), lambda x: x.squeeze()])
    if subtyping:
        dataset = MD.MIL_Test_Subtyping_Dataset(feature_dim=encoder_dim, bag_size=bag_size,
                                                num_samples_per_class=num_samples_per_class,
                                                num_classes=num_classes, repr_inst=repr_inst, std=std,
                                                transform=transformations)
    elif num_classes == 2:
        dataset = MD.MIL_Test_Dataset(feature_dim=encoder_dim, bag_size=bag_size, num_samples=num_samples_per_class * 2,
                                      repr_inst=repr_inst, transform=transformations)
    else:
        raise ValueError("Subtypig=False may only be combined with num_classes=2")
    train_dataset, test_dataset, val_dataset = random_split(dataset, lengths=[
        int(len(dataset) * 0.6),
        int(len(dataset) * 0.0),
        int(len(dataset) * 0.4)])
    model = CLAM.CLAM(encoded_dim=encoder_dim, input_dim_attention=encoder_dim//2,
                      projection_dim_attention=encoder_dim//4, dropout=True, n_classes=num_classes, n_inst=repr_inst,
                      subtyping=subtyping)
    model, accuracy_logger, loss_logger = train(train_dataset, val_dataset, model, learning_rate=1e-5,
                                                weight_decay=1e-5, max_epochs=20, device="cpu", bag_weight=0.7)


    fig, axs = plt.subplots(ncols=2, nrows=1)
    accuracy_logger.plot(axs[0], "Accuracies")
    loss_logger.plot(axs[1], "Loss")
    plt.show()
