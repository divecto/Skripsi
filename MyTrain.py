import torch
import argparse
import csv
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import numpy as np
from Src.SINet import SINet_ResNet101  # import ResNet-101
from Src.utils.Dataloader import get_loader
from Src.utils.trainer import trainer, adjust_lr
from apex import amp

# Kelas EarlyStopping
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001, mode='min', restore_best_weights=True, save_best_model_path='./'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        self.save_best_model_path = save_best_model_path
        self.best_score = np.Inf if mode == 'min' else -np.Inf
        self.wait = 0
        self.stopped_epoch = 0
        self.best_weights = None

    def on_epoch_end(self, epoch, current_score, model):
        if current_score is None:
            print("Warning: current_score is None. Skipping early stopping check.")
            return False

        if self.mode == 'min':
            if current_score < self.best_score - self.min_delta:
                self.best_score = current_score
                self.wait = 0
                if self.restore_best_weights:
                    self.best_weights = model.state_dict()
                    torch.save(self.best_weights, os.path.join(self.save_best_model_path, 'best_model.pth'))
            else:
                self.wait += 1
                if self.wait >= self.patience:
                    self.stopped_epoch = epoch
                    if self.restore_best_weights:
                        model.load_state_dict(self.best_weights)
                    return True
        elif self.mode == 'max':
            if current_score > self.best_score + self.min_delta:
                self.best_score = current_score
                self.wait = 0
                if self.restore_best_weights:
                    self.best_weights = model.state_dict()
                    torch.save(self.best_weights, os.path.join(self.save_best_model_path, 'best_model.pth'))
            else:
                self.wait += 1
                if self.wait >= self.patience:
                    self.stopped_epoch = epoch
                    if self.restore_best_weights:
                        model.load_state_dict(self.best_weights)
                    return True
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=70, help='epoch number, default=30')
    parser.add_argument('--lr', type=float, default=1e-4, help='init learning rate, try `lr=1e-4`')
    parser.add_argument('--batchsize', type=int, default=18, help='training batch size (Note: ~500MB per img in GPU)')
    parser.add_argument('--trainsize', type=int, default=352, help='the size of training image, try small resolutions for speed (like 256)')
    parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
    parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate per decay step')
    parser.add_argument('--decay_epoch', type=int, default=30, help='every N epochs decay lr')
    parser.add_argument('--gpu', type=int, default=0, help='choose which gpu you use')
    parser.add_argument('--save_epoch', type=int, default=10, help='every N epochs save your trained snapshot')
    parser.add_argument('--save_model', type=str, default='./Snapshot/2020-CVPR-SINet/')
    parser.add_argument('--train_img_dir', type=str, default='./Dataset/TrainDataset/Imgs/')
    parser.add_argument('--train_gt_dir', type=str, default='./Dataset/TrainDataset/GT/')
    parser.add_argument('--csv_path', type=str, default='./training_log.csv', help='path to save training log CSV')
    parser.add_argument('--summary_path', type=str, default='./summary.txt', help='path to save training summary TXT')
    opt = parser.parse_args()

    # Check available GPUs and validate the selected GPU
    num_gpus = torch.cuda.device_count()
    if opt.gpu >= num_gpus or opt.gpu < 0:
        raise ValueError(f"Invalid GPU id {opt.gpu}. Available GPUs: {num_gpus}")

    torch.cuda.set_device(opt.gpu)

    # Use deeper network for better performance with channel=64
    model_SINet = SINet_ResNet101(channel=32).cuda()
    print('-' * 30, model_SINet, '-' * 30)

    optimizer = torch.optim.Adam(model_SINet.parameters(), opt.lr)
    LogitsBCE = torch.nn.BCEWithLogitsLoss()

    net, optimizer = amp.initialize(model_SINet, optimizer, opt_level='O1')  # NOTES: Ox not 0x

    train_loader = get_loader(opt.train_img_dir, opt.train_gt_dir, batchsize=opt.batchsize,
                              trainsize=opt.trainsize, num_workers=12)
    total_step = len(train_loader)

    summary = '-' * 30 + "\n[Training Dataset INFO]\nimg_dir: {}\ngt_dir: {}\nLearning Rate: {}\nBatch Size: {}\n" \
              "Training Save: {}\ntotal_num: {}\n".format(opt.train_img_dir, opt.train_gt_dir, opt.lr,
                                                         opt.batchsize, opt.save_model, total_step) + '-' * 30
    print(summary)

    # Save summary to TXT file
    with open(opt.summary_path, mode='w') as summary_file:
        summary_file.write(summary)

    # Initialize EarlyStopping
    early_stopping = EarlyStopping(patience=5, mode='min', restore_best_weights=True, save_best_model_path=opt.save_model)

    # Open CSV file to write
    with open(opt.csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Loss'])  # Write header

        # Setup for live plotting
        fig, ax = plt.subplots()
        epochs = []
        losses = []
        line, = ax.plot(epochs, losses, label='Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss Over Epochs')
        ax.legend()

        def update_plot(epoch, loss):
            epochs.append(epoch)
            losses.append(loss)
            line.set_xdata(epochs)
            line.set_ydata(losses)
            ax.relim()
            ax.autoscale_view()
            plt.draw()
            plt.pause(0.1)  # Adjust the pause time as needed

        for epoch_iter in range(1, opt.epoch + 1):
            adjust_lr(optimizer, epoch_iter, opt.decay_rate, opt.decay_epoch)
            loss = trainer(train_loader=train_loader, model=model_SINet,
                           optimizer=optimizer, epoch=epoch_iter,
                           opt=opt, loss_func=LogitsBCE, total_step=total_step)

            # Write data to CSV
            writer.writerow([epoch_iter, loss])

            # Update plot
            update_plot(epoch_iter, loss)

            # Check early stopping
            if early_stopping.on_epoch_end(epoch_iter, loss, model_SINet):
                print(f"Early stopping triggered at epoch {epoch_iter}")
                break

    plt.savefig('training_loss.png')  # Save the plot as a PNG file
    plt.show()
