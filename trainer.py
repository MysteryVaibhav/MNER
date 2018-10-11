import torch.utils.data
import torch.nn as nn
from model import MNER
from timeit import default_timer as timer
from util import *
import sys
from tqdm import tqdm
from torchcrf import CRF


def init_xavier(m):
    """
    Sets all the linear layer weights as per xavier initialization
    :param m:
    :return: Nothing
    """
    if type(m) == torch.nn.Linear:
        fan_in = m.weight.size()[1]
        fan_out = m.weight.size()[0]
        std = np.sqrt(6.0 / (fan_in + fan_out))
        m.weight.data.normal_(0, std)
        if m.bias is not None:
            m.bias.data.zero_()


class Trainer:
    def __init__(self, params, data_loader, evaluator):
        self.params = params
        self.data_loader = data_loader
        self.evaluator = evaluator

    def train(self):
        num_of_tags = len(self.data_loader.labelVoc)
        model = MNER(self.params, self.data_loader.word_matrix, num_of_tags)
        model.apply(init_xavier)
        loss_function = CRF(num_of_tags)
        if torch.cuda.is_available():
            model = model.cuda()
            loss_function = loss_function.cuda()
        optimizer = torch.optim.SGD(model.parameters(), lr=self.params.lr, momentum=0.9)
        
        try:
            prev_best = 0
            for epoch in range(self.params.num_epochs):
                iters = 1
                losses = []
                start_time = timer()
                for (x, x_img, y, mask, x_c, lens) in tqdm(self.data_loader.train_data_loader):
                    model.train()
                    optimizer.zero_grad()

                    # forward pass
                    emissions = model(to_variable(x), to_variable(x_img), lens, to_variable(mask), to_variable(x_c))  # seq_len * bs * labels
                    tags = to_variable(y).transpose(0, 1).contiguous()                              # seq_len * bs
                    mask = to_variable(mask).byte().transpose(0, 1)                                 # seq_len * bs

                    # computing crf loss
                    loss = -loss_function(emissions, tags, mask=mask)
                    loss.backward()
                    losses.append(loss.data.cpu().numpy())

                    if self.params.clip_value > 0:
                        torch.nn.utils.clip_grad_norm(model.parameters(), self.params.clip_value)
                    optimizer.step()

                    #tqdm.write("[%d] :: Training Loss: %f   \r" % (iters, np.asscalar(np.mean(losses))))
                    iters += 1

                optim_state = optimizer.state_dict()
                optim_state['param_groups'][0]['lr'] = optim_state['param_groups'][0]['lr'] / (1 + self.params.gamma * (epoch + 1))
                optimizer.load_state_dict(optim_state)

                # Calculate accuracy and save best model
                if (epoch + 1) % self.params.validate_every == 0:
                    acc_dev, f1_dev, p_dev, r_dev = self.evaluator.get_accuracy(model, 'val', loss_function)

                    print("Epoch {} : Training Loss: {:.5f}, Acc: {}, F1: {}, Prec: {}, Rec: {},"
                          "Time elapsed {:.2f} mins"
                          .format(epoch + 1, np.asscalar(np.mean(losses)), acc_dev, f1_dev, p_dev, r_dev,
                                  (timer() - start_time) / 60))
                    if acc_dev > prev_best:
                        print("Accuracy increased....saving weights !!")
                        prev_best = acc_dev
                        torch.save(model.state_dict(),
                                   self.params.model_dir + 'best_model_weights_{}_{:.3f}.t7'.format(epoch + 1, acc_dev))
                else:
                    print("Epoch {} : Training Loss: {:.5f}".format(epoch + 1, np.asscalar(np.mean(losses))))
        except KeyboardInterrupt:
            print("Interrupted.. saving model !!!")
            torch.save(model.state_dict(), self.params.model_dir + '/model_weights_interrupt.t7')


