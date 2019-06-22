import argparse
import torch
from tqdm import tqdm
from data_loader.data_loader_new import Dataset
from torch.utils.data import DataLoader

#import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
import cv2
import numpy as np


def main(config, resume):
    logger = config.get_logger('test')

    # setup data_loader instances
    '''data_loader = getattr(module_data, config['data_loader']['type'])(
        config['data_loader']['args']['data_dir'],
        batch_size=512,
        shuffle=False,
        validation_split=0.0,
        training=False,
        num_workers=2
    )'''
    data_dir = config.config['data_loader']['args']['data_dir']
    validation_set = Dataset(data_dir + "/validation")
    valid_data_loader = DataLoader(validation_set, batch_size=2)

    # build model architecture
    model = config.initialize('arch', module_arch)
    logger.info(model)

    # get function handles of loss and metrics
    loss_fn = getattr(module_loss, config['loss'])
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    logger.info('Loading checkpoint: {} ...'.format(resume))
    checkpoint = torch.load(resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    model.double()

    total_loss = 0.0
    total_metrics = torch.zeros(len(metric_fns))

    with torch.no_grad():
        for i, (data, target) in enumerate(tqdm(valid_data_loader)):
            data, target = data.to(device), target.to(device)
            
            output = model(data)

            #
            # save sample images, or do something with output here
            #
                       
            for sample_idx in range(2):
                img = data[sample_idx,0]  

                
                for i in range(0,output.shape[1]):
                    #cv2.imshow('Image with generated blobs', output[sample_idx,i].numpy())
                    data[sample_idx,0] = torch.max(data[sample_idx,0], output[sample_idx,i])
                    #cv2.waitKey(0)
                
                cv2.imshow('Image with generated blobs', data[sample_idx,0].numpy())
                cv2.imshow('Original', img.numpy())
                
                cv2.waitKey()

            # computing loss, metrics on test set
            loss = loss_fn(output, target)
            batch_size = data.shape[0]
            total_loss += loss.item() * batch_size
            
            #for i, metric in enumerate(metric_fns):
            #    total_metrics[i] += metric(output, target) * batch_size

    #n_samples = len(data_loader.sampler)
    log = {'loss': total_loss / 2}
    '''log.update({
        met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)
    })'''
    logger.info(log)
    model.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Template')

    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='path to latest checkpoint (default: None)')
    parser.add_argument('-d', '--device', default=None, type=str,
                        help='indices of GPUs to enable (default: all)')

    args = parser.parse_args()
    config = ConfigParser(args, True)
    main(config, args.resume)
    cv2.destroyAllWindows()
