import torch
import argparse
import csv
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from Src.SINet import SINet_ResNet101  # Pastikan Anda mengimpor model yang benar
from Src.utils.Dataloader import get_loader
from Src.utils.trainer import trainer, adjust_lr
from apex import amp

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=40,
                        help='epoch number, default=30')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='init learning rate, try `lr=1e-4`')
    parser.add_argument('--batchsize', type=int, default=36,
                        help='training batch size (Note: ~500MB per img in GPU)')
    parser.add_argument('--trainsize', type=int, default=352,
                        help='the size of training image, try small resolutions for speed (like 256)')
    parser.add_argument('--clip', type=float, default=0.5,
                        help='gradient clipping margin')
    parser.add_argument('--decay_rate', type=float, default=0.1,
                        help='decay rate of learning rate per decay step')
    parser.add_argument('--decay_epoch', type=int, default=30,
                        help='every N epochs decay lr')
    parser.add_argument('--gpu', type=int, default=0,
                        help='choose which gpu you use')
    parser.add_argument('--save_epoch', type=int, default=10,
                        help='every N epochs save your trained snapshot')
    parser.add_argument('--save_model', type=str, default='./Snapshot/2020-CVPR-SINet/')
    parser.add_argument('--train_img_dir', type=str, default='./Dataset/TrainDataset/Imgs/')
    parser.add_argument('--train_gt_dir', type=str, default='./Dataset/TrainDataset/GT/')
    parser.add_argument('--csv_path', type=str, default='./training_log.csv', 
                        help='path to save training log CSV')
    parser.add_argument('--summary_path', type=str, default='./summary.txt', 
                        help='path to save training summary TXT')
    opt = parser.parse_args()

    # Check available GPUs and validate the selected GPU
    num_gpus = torch.cuda.device_count()
    if opt.gpu >= num_gpus or opt.gpu < 0:
        raise ValueError(f"Invalid GPU id {opt.gpu}. Available GPUs: {num_gpus}")

    torch.cuda.set_device(opt.gpu)

    # TIPS: you also can use deeper network for better performance like channel=64
    model_SINet = SINet_ResNet101(channel=32).cuda()  # Ganti SINet_ResNet50 dengan SINet_ResNet101
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

    plt.savefig('training_loss.png')  # Save the plot as a PNG file
    plt.show()
