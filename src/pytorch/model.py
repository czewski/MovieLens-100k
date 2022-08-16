# import torch.nn as nn
# from torch.autograd import Variable


# class MyModel(nn.Module):
#     def __init__(self):
    
    # self.lstm = nn.LSTM(embedding_length, hidden_size)
    # self.label = nn.Linear(hidden_size, output_size)


    # def forward(self):


    # h_0 = Variable(torch.zeros(1, batch_size, self.hidden_size).cuda())
    # c_0 = Variable(torch.zeros(1, batch_size, self.hidden_size).cuda())


    # output, (final_hidden_state, final_cell_state) = self.lstm(input, (h_0, c_0))


    # return self.label(final_hidden_state[-1]) 
