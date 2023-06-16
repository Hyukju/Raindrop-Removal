import os 
import argparse
import utils 
from model.find_model import find_model
from torch.utils.data import DataLoader
from datetime import datetime
from time import time 

def train(args):

    ckpt_dir = os.path.join(args.ckpt_dir, args.model)
    os.makedirs(ckpt_dir, exist_ok=True)

    utils.print_args(args)
    utils.save_log(ckpt_dir, 'train', args)
    
    model, dataset = find_model(model_name=args.model, phase='train', dataset_dir=args.dataset_dir, use_transform=True, learning_rate=args.learning_rate)
    data_loader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
   
    max_batchs = len(data_loader)

    st_epoch = 1
    # check continuous training
    if args.train_continue:
        last_epoch = model.load(ckpt_dir)
        if last_epoch > 1:
            st_epoch = last_epoch + 1
            print(f'Continued Training from EPOCH {last_epoch}!!')

    
    for epoch in range(st_epoch, args.max_epochs + 1):
        start_t = time()
        losses_list = []
        for batch, data in enumerate(data_loader, 1):
            # load training data
            model.set_inputs(data)
            # run training
            model.train_on_batch()

            # print loss for each batch
            if batch == 1 or batch % 10 == 0 or batch == max_batchs:
                losses_list.append(model.get_losses())
                utils.print_losses(epoch,  args.max_epochs, batch, max_batchs, losses_list, title=f'[{args.model}] TRAIN', mode='last')

        # print loss for Each Epoch (average of batch losses)
        delay_t = time() - start_t
        utils.print_losses(epoch, args.max_epochs, batch, max_batchs, losses_list, title=f'[{args.model}: {delay_t:.3f}s] TRAIN MEAN LOSS', mode='mean')
        utils.save_losses(ckpt_dir, epoch, losses_list, mode='mean')

        # save images 
        utils.save_outputs(
            save_dir = os.path.join(args.save_dir, args.model, 'train'),
            filename = f'{epoch:04d}.png',
            outputs = model.get_outputs(),
            max_display = 3
            )
        
        if epoch % 50 == 0:
            model.save(ckpt_dir, epoch)
    print(f'Training Finished!!: {datetime.now()}')

if __name__=='__main__':
    parser = argparse.ArgumentParser(prog = 'DeRainDrop')             
    
    parser.add_argument('--train_continue', action='store_true', dest='train_continue')
    parser.add_argument('--model', default='proposed', type=str, dest='model')
    parser.add_argument('--max_epochs', default=2000, type=int, dest='max_epochs')
    parser.add_argument('--batch_size', default=16, type=int, dest='batch_size')
    parser.add_argument('--num_workers', default=8, type=int, dest='num_workers')
    parser.add_argument('--lr', default=0.0002, type=float, dest='learning_rate')
    parser.add_argument('--ckpt_dir', default='./checkpoint/', type=str, dest='ckpt_dir')    
    parser.add_argument('--dataset_dir', default='./dataset/train', type=str, dest='dataset_dir')    
    parser.add_argument('--save_dir', default='./result', type=str, dest='save_dir')
    
    args = parser.parse_args()
    train(args)


