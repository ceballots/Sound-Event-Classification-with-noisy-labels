from torch import nn
from copy import deepcopy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import tqdm
import os
import numpy as np
import time
import math
from torch.autograd import Variable

from storage_utils import save_statistics,save_parameters,load_statistics
import losses as CustomLosses
import mixup as MixUp 

class ExperimentBuilder(nn.Module):
    def __init__(self, network_model, experiment_name, num_epochs, train_data, val_data,
                 test_data,batch_size, weight_decay_coefficient, use_gpu,training_instances,
                 test_instances,val_instances,image_height, image_width,eps_smooth,num_classes,loss_function,use_cluster,args,gpu_id,consider_manual = False,q_=0.8,continue_from_epoch=-1):
        """
        Initializes an ExperimentBuilder object. Such an object takes care of running training and evaluation of a deep net
        on a given dataset. It also takes care of saving per epoch models and automatically inferring the best val model
        to be used for evaluating the test set metrics.
        :param network_model: A pytorch nn.Module which implements a network architecture.
        :param experiment_name: The name of the experiment. This is used mainly for keeping track of the experiment and creating and directory structure that will be used to save logs, model parameters and other.
        :param num_epochs: Total number of epochs to run the experiment
        :param train_data: An object of the DataProvider type. Contains the training set.
        :param val_data: An object of the DataProvider type. Contains the val set.
        :param test_data: An object of the DataProvider type. Contains the test set.
        :param weight_decay_coefficient: A float indicating the weight decay to use with the adam optimizer.
        :param use_gpu: A boolean indicating whether to use a GPU or not.
        :param continue_from_epoch: An int indicating whether we'lll start from scrach (-1) or whether we'll reload a previously saved model of epoch 'continue_from_epoch' and continue training from there.
        """
        super(ExperimentBuilder, self).__init__()
        if torch.cuda.is_available() and use_gpu:  # checks whether a cuda gpu is available and whether the gpu flag is True
            if use_cluster:
                if "," in gpu_id:
                    self.device = [torch.device('cuda:{}'.format(idx)) for idx in gpu_id.split(",")]  # sets device to be cuda
                else:
                    self.device = torch.device('cuda:{}'.format(gpu_id))  # sets device to be cuda

                os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id  # sets the main GPU to be the one at index 0 (on multi gpu machines you can choose which one you want to use by using the relevant GPU ID)
                print("use GPU")
                print("GPU ID {}".format(gpu_id))
            else:
                self.device = torch.device('cuda')  # sets device to be cuda
                os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # sets the main GPU to be the one at index 0 (on multi gpu machines you can choose which one you want to use by using the relevant GPU ID)
                print("use GPU")
        else:
            print("use CPU")
            self.device = torch.device('cpu')  # sets the device to be CPU

        self.experiment_name = experiment_name
        self.model = network_model
        self.model.reset_parameters()
        if type(self.device) is list:
            self.model.to(self.device[0])
            self.model = nn.DataParallel(module=self.model, device_ids=self.device)
            self.device = self.device[0]
        else:
            self.model.to(self.device)  # sends the model from the cpu to the gpu
          # re-initialize network parameters
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.batch_size = batch_size
        self.training_instances = training_instances
        self.test_instances = test_instances
        self.val_instances = val_instances
        self.image_height = image_height
        self.image_width = image_width
        self.consider_manual = consider_manual
        self.optimizer = optim.Adam(self.parameters(), amsgrad=False,
                                    weight_decay=weight_decay_coefficient)
        # Generate the directory names
        self.experiment_folder = os.path.abspath(experiment_name)
        self.experiment_logs = os.path.abspath(os.path.join(self.experiment_folder, "result_outputs"))
        self.experiment_saved_models = os.path.abspath(os.path.join(self.experiment_folder, "saved_models"))
        print(self.experiment_folder, self.experiment_logs)
        # Set best models to be at 0 since we are just starting
        self.best_val_model_idx = 0
        self.best_val_model_acc = 0.
        self.loss_function = loss_function
        self.q_ = q_
        self.eps_smooth = eps_smooth
        self.num_classes = num_classes
        self.mixup = args.mixup
        self.alpha = args.alpha
        self.use_gpu = args.use_gpu
        self.stack = args.stack
        self.width = args.image_width
        self.heigth = args.image_height
        self.shuffle = args.shuffle
        if self.stack:
            self.batch_size = 2*self.batch_size
        if not os.path.exists(self.experiment_folder):  # If experiment directory does not exist
            os.mkdir(self.experiment_folder)  # create the experiment directory

        if not os.path.exists(self.experiment_logs):
            os.mkdir(self.experiment_logs)  # create the experiment log directory

        if not os.path.exists(self.experiment_saved_models):
            os.mkdir(self.experiment_saved_models)  # create the experiment saved models directory
            
        # Safe parameters with which are running
        if use_gpu:
            save_parameters(args,self.experiment_logs)
        
        self.num_epochs = num_epochs
        self.criterion = nn.CrossEntropyLoss().to(self.device)  # send the loss computation to the GPU
        if continue_from_epoch == -2:
            try:
                self.best_val_model_idx, self.best_val_model_acc, self.state = self.load_model(
                    model_save_dir=self.experiment_saved_models, model_save_name="train_model",
                    model_idx='latest')  # reload existing model from epoch and return best val model index
                # and the best val acc of that model
                self.starting_epoch = continue_from_epoch
            except:
                print("Model objects cannot be found, initializing a new model and starting from scratch")
                self.starting_epoch = 0
                self.state = dict()

        elif continue_from_epoch != -1:  # if continue from epoch is not -1 then
            self.best_val_model_idx, self.best_val_model_acc, self.state = self.load_model(
                model_save_dir=self.experiment_saved_models, model_save_name="train_model",
                model_idx=continue_from_epoch)  # reload existing model from epoch and return best val model index
            # and the best val acc of that model
            self.starting_epoch = self.state['current_epoch_idx']
        else:
            self.starting_epoch = 0
            self.state = dict()

    def get_num_parameters(self):
        total_num_params = 0
        for param in self.parameters():
            total_num_params += np.prod(param.shape)

        return total_num_params


    def run_train_iter(self, x, y,manual_verified,epoch_number = -1):
        """
        Receives the inputs and targets for the model and runs a training iteration. Returns loss and accuracy metrics.
        :param x: The inputs to the model. A numpy array of shape batch_size, channels, height, width
        :param y: The targets for the model. A numpy array of shape batch_size, num_classes
        :return: the loss and accuracy for this batch
        """
        self.train()  # sets model to training mode (in case batch normalization or other methods have different procedures for training and evaluation)


        if len(y.shape) > 1:
            y = np.argmax(y, axis=1)  # convert one hot encoded labels to single integer labels
            y_no_cuda = y
        #print(type(x))

        if type(x) is np.ndarray:
            x, y = torch.Tensor(x).float().to(device=self.device), torch.Tensor(y).long().to(
            device=self.device)  # send data to device as torch tensors
            y_no_cuda = torch.Tensor(y_no_cuda).long()
        x = x.to(self.device)
        y = y.to(self.device)

        if self.mixup == True:
            inputs, targets_a, targets_b,y_, lam  = MixUp.mixup_data(x, y,y_no_cuda,self.num_classes,
                                                       self.alpha, use_cuda=self.use_gpu)
           # inputs, targets_a, targets_b = map(Variable, (inputs,
           #                                           targets_a, targets_b))
            if self.stack  == True:
                x_stack = torch.stack((x, inputs), 0)
                x_stack = x_stack.view((self.batch_size,1,self.heigth,self.width))
                out = self.model.forward_train(x_stack) 
                loss_mix = MixUp.mixup_criterion(out[:int(self.batch_size/2)],targets_a,targets_b,lam,self.device)
                loss_smooth = CustomLosses.loss_function(out[int(self.batch_size/2):],y,y_no_cuda,self.num_classes,self.device,self.eps_smooth,self.loss_function,
                                          array_manual_label=manual_verified,consider_manual = self.consider_manual)
                loss = (loss_mix + loss_smooth)/2
            else:
                out = self.model.forward_train(x)  # forward the data in the model
                loss = MixUp.mixup_criterion(out, targets_a, targets_b, lam,self.device)
        else:
            out = self.model.forward_train(x)
            loss = CustomLosses.loss_function(out,y,y_no_cuda,self.num_classes,self.device,self.eps_smooth,self.loss_function,
                                          array_manual_label=manual_verified,consider_manual = self.consider_manual)
        
       #if self.loss_function=='CCE':
       #    loss = F.cross_entropy(input=out, target=y)  # compute loss
       #elif self.loss_function=='lq_loss':
       #    loss=CustomLosses.lq_loss(y_true=y,y_pred=out,_q=self.q_)

        self.optimizer.zero_grad()  # set all weight grads from previous training iters to 0
        loss.backward()  # backpropagate to compute gradients for current iter loss

        self.optimizer.step()  # update network parameters
        _, predicted = torch.max(out.data, 1)  # get argmax of predictions
        if self.stack:
            accuracy = np.mean(list(predicted[int(self.batch_size/2):].eq(y.data).cpu())) 
        else:

            accuracy = np.mean(list(predicted.eq(y.data).cpu()))  # compute accuracy
        return loss.data.detach().cpu().numpy(), accuracy

    def run_evaluation_iter(self, x, y):
        """
        Receives the inputs and targets for the model and runs an evaluation iterations. Returns loss and accuracy metrics.
        :param x: The inputs to the model. A numpy array of shape batch_size, channels, height, width
        :param y: The targets for the model. A numpy array of shape batch_size, num_classes
        :return: the loss and accuracy for this batch
        """
        self.eval()  # sets the system to validation mode
        if len(y.shape) > 1:
            y = np.argmax(y, axis=1)  # convert one hot encoded labels to single integer labels
            y_no_cuda = y
        if type(x) is np.ndarray:
            x, y = torch.Tensor(x).float().to(device=self.device), torch.Tensor(y).long().to(
            device=self.device)  # convert data to pytorch tensors and send to the computation device
            y_no_cuda = torch.Tensor(y_no_cuda).long()

        x = x.to(self.device)
        y = y.to(self.device)
        out = self.model.forward(x)  # forward the data in the model
        
       # loss = CustomLosses.loss_function(out,y,y_no_cuda,self.num_classes,self.device,self.eps_smooth,self.loss_function,
       #                                   array_manual_label=None,consider_manual = False)
        
      # if self.loss_function == 'CCE':
        loss = F.cross_entropy(input=out, target=y)  # compute loss
      # elif self.loss_function=='lq_loss':
      #     loss=CustomLosses.lq_loss(y_true=y,y_pred=out,_q=self.q_)
     
        _, predicted = torch.max(out.data, 1)  # get argmax of predictions
        accuracy = np.mean(list(predicted.eq(y.data).cpu()))  # compute accuracy
        return loss.data.detach().cpu().numpy(), accuracy

    def save_model(self, model_save_dir, model_save_name, model_idx, state):
        """
        Save the network parameter state and current best val epoch idx and best val accuracy.
        :param model_save_name: Name to use to save model without the epoch index
        :param model_idx: The index to save the model with.
        :param best_validation_model_idx: The index of the best validation model to be stored for future use.
        :param best_validation_model_acc: The best validation accuracy to be stored for use at test time.
        :param model_save_dir: The directory to store the state at.
        :param state: The dictionary containing the system state.
        """
        state['network'] = self.state_dict()  # save network parameter and other variables.
        torch.save(state, f=os.path.join(model_save_dir, "{}_{}".format(model_save_name, str(
            model_idx))))  # save state at prespecified filepath

    def load_model(self, model_save_dir, model_save_name, model_idx):
        """
        Load the network parameter state and the best val model idx and best val acc to be compared with the future val accuracies, in order to choose the best val model
        :param model_save_dir: The directory to store the state at.
        :param model_save_name: Name to use to save model without the epoch index
        :param model_idx: The index to save the model with.
        :return: best val idx and best val model acc, also it loads the network state into the system state without returning it
        """
        state = torch.load(f=os.path.join(model_save_dir, "{}_{}".format(model_save_name, str(model_idx))))
        self.load_state_dict(state_dict=state['network'])
        return state['best_val_model_idx'], state['best_val_model_acc'], state

    def run_experiment(self):
        """
        Runs experiment train and evaluation iterations, saving the model and best val model and val model accuracy after each epoch
        :return: The summary current_epoch_losses from starting epoch to total_epochs.
        """
        total_losses = {"train_acc": [], "train_loss": [], "val_acc": [],
                        "val_loss": []}  # initialize a dict to keep the per-epoch metrics
        train_number_batches = int(math.ceil(self.training_instances/self.batch_size))
        val_number_batches = int(math.ceil(self.val_instances/self.batch_size))
        if self.stack:
           train_number_batches = 2*train_number_batches
           val_number_batches = 2*val_number_batches
        
        
        for i, epoch_idx in enumerate(range(self.starting_epoch, self.num_epochs)):
            epoch_start_time = time.time()
            current_epoch_losses = {"train_acc": [], "train_loss": [], "val_acc": [], "val_loss": []}
            
            print("num batches",train_number_batches)
            if self.stack:
               total_ = train_number_batches-2
            else:
               total_ = train_number_batches-1
            
            if self.shuffle:
                idx = np.arange(0,self.train_data.inputs.shape[0])
                print("before shuffle", self.train_data.inputs.shape)
                np.random.shuffle(idx)
                self.train_data.inputs = self.train_data.inputs[idx]
                self.train_data.targets = self.train_data.targets[idx]
            with tqdm.tqdm(total=total_) as pbar_train:  # create a progress bar for training
                 for idx in range(total_):                   
                        
                    x,y,manual_verified = self.get_batch(data = self.train_data,
                                             idx = idx, number_batches = total_,train=True)
                    loss, accuracy = self.run_train_iter(x=x, y=y,manual_verified=manual_verified,epoch_number = epoch_idx)  # take a training iter step
                    current_epoch_losses["train_loss"].append(loss)  # add current iter loss to the train loss list
                    current_epoch_losses["train_acc"].append(accuracy)  # add current iter acc to the train acc list
                    pbar_train.update(1)
                    pbar_train.set_description("loss: {:.4f}, accuracy: {:.4f}".format(loss, accuracy))

                        
            with tqdm.tqdm(total=val_number_batches) as pbar_val:  # create a progress bar for validation
                for idx in range(val_number_batches):
                    x,y = self.get_batch(data = self.val_data,
                                             idx = idx, number_batches = val_number_batches) 
                    loss, accuracy = self.run_evaluation_iter(x=x, y=y)  # run a validation iter
                    current_epoch_losses["val_loss"].append(loss)  # add current iter loss to val loss list.
                    current_epoch_losses["val_acc"].append(accuracy)  # add current iter acc to val acc lst.
                    pbar_val.update(1)  # add 1 step to the progress bar
                    pbar_val.set_description("loss: {:.4f}, accuracy: {:.4f}".format(loss, accuracy))
            val_mean_accuracy = np.mean(current_epoch_losses['val_acc'])
            if val_mean_accuracy > self.best_val_model_acc:  # if current epoch's mean val acc is greater than the saved best val acc then
                self.best_val_model_acc = val_mean_accuracy  # set the best val model acc to be current epoch's val accuracy
                self.best_val_model_idx = epoch_idx  # set the experiment-wise best val idx to be the current epoch's idx

            for key, value in current_epoch_losses.items():
                total_losses[key].append(np.mean(
                    value))  # get mean of all metrics of current epoch metrics dict, to get them ready for storage and output on the terminal.

            save_statistics(experiment_log_dir=self.experiment_logs, filename='summary.csv',
                            stats_dict=total_losses, current_epoch=i)  # save statistics to stats file.

            load_statistics(experiment_log_dir=self.experiment_logs, filename='summary.csv') # How to load a csv file if you need to

            out_string = "_".join(
                ["{}_{:.4f}".format(key, np.mean(value)) for key, value in current_epoch_losses.items()])
            # create a string to use to report our epoch metrics
            epoch_elapsed_time = time.time() - epoch_start_time  # calculate time taken for epoch
            epoch_elapsed_time = "{:.4f}".format(epoch_elapsed_time)
            print("Epoch {}:".format(epoch_idx), out_string, "epoch time", epoch_elapsed_time, "seconds")
            self.state['current_epoch_idx']=epoch_idx
            self.state['best_val_model_acc']=self.best_val_model_acc
            self.state['best_val_model_idx']=self.best_val_model_idx
            self.save_model(model_save_dir=self.experiment_saved_models,
                            # save model and best val idx and best val acc, using the model dir, model name and model idx
                            model_save_name="train_model", model_idx=epoch_idx,state=self.state)

            self.save_model(model_save_dir=self.experiment_saved_models,
                            model_save_name="train_model",model_idx='latest',state=self.state)

        print("Generating test set evaluation metrics")
        self.load_model(model_save_dir=self.experiment_saved_models, model_idx=self.best_val_model_idx,
                        # load best validation model
                        model_save_name="train_model")
        current_epoch_losses = {"test_acc": [], "test_loss": []}  # initialize a statistics dict

        test_number_batches = int(math.ceil(self.test_instances/self.batch_size))
        if self.stack:
           test_number_batches = 2*test_number_batches
        with tqdm.tqdm(total=test_number_batches) as pbar_test:  # ini a progress bar
            for idx in range(test_number_batches):  # sample batch
                x,y = self.get_batch(data = self.test_data,
                                         idx = idx, number_batches = test_number_batches) 
                loss, accuracy = self.run_evaluation_iter(x=x,
                                                          y=y)  # compute loss and accuracy by running an evaluation step
                current_epoch_losses["test_loss"].append(loss)  # save test loss
                current_epoch_losses["test_acc"].append(accuracy)  # save test accuracy
                pbar_test.update(1)  # update progress bar status
                pbar_test.set_description(
                    "loss: {:.4f}, accuracy: {:.4f}".format(loss, accuracy))  # update progress bar string output

        test_losses = {key: [np.mean(value)] for key, value in
                       current_epoch_losses.items()}  # save test set metrics in dict format
        save_statistics(experiment_log_dir=self.experiment_logs, filename='test_summary.csv',
                         #save test set metrics on disk in .csv format
                        stats_dict=test_losses, current_epoch=0)

        return total_losses, test_losses

    def get_batch(self, data, idx, number_batches,train=False):
        """
        Get batch data and convert it from h5py to numpy format
        :param data: {train,validation,test} data
        :param idx: current batch number
        :param number_batches: number of batches in set
        """
        if self.stack:
            batch_size =int(self.batch_size/2)
        else:
            batch_size = self.batch_size
        
        if idx == number_batches - 1:
            x_np = data.inputs[idx*batch_size:(idx+1)*batch_size]
            y = data.targets[idx*batch_size:(idx+1)*batch_size]
            if train == True:
                manual = data.manual_verified[idx*batch_size:]
                return x_np,y,manual
            return x_np,y
        else:
            x_np = data.inputs[idx*batch_size:(idx+1)*batch_size]
            y = data.targets[idx*batch_size:(idx+1)*batch_size]
            if train == True:
                manual = data.manual_verified[idx*batch_size:(idx+1)*batch_size]
                return x_np,y,manual
            return x_np,y
