import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from genotypes import STEPS
from utils import mask2d
from utils import LockedDropout
from utils import embedded_dropout
from torch.autograd import Variable
import numpy as np
from tacred_utils import constant

INITRANGE = 0.04


class DARTSCell(nn.Module):
  def __init__(self, ninp, nhid, dropouth, dropoutx, genotype):
    super(DARTSCell, self).__init__()
    """
      This is effectively the Arc component which runs the forward pass of a step in the RNN
      genotype: sample_arc.
    """
    # TODO: Where does genotype come from?
    self.nhid = nhid
    self.dropouth = dropouth
    self.dropoutx = dropoutx
    self.genotype = genotype

    # genotype is None when doing arch search
    steps = len(self.genotype.recurrent) if self.genotype is not None else STEPS
    self._W0 = nn.Parameter(torch.Tensor(ninp+nhid, 2*nhid).uniform_(-INITRANGE, INITRANGE))
    self._Ws = nn.ParameterList([
        nn.Parameter(torch.Tensor(nhid, 2*nhid).uniform_(-INITRANGE, INITRANGE)) for i in range(steps)
    ])

  def forward(self, inputs, hidden):
    T, B = inputs.size(0), inputs.size(1)

    if self.training:
      x_mask = mask2d(B, inputs.size(2), keep_prob=1.-self.dropoutx)
      #print(hidden.shape)
      h_mask = mask2d(B, hidden.size(2), keep_prob=1.-self.dropouth)
    else:
      x_mask = h_mask = None

    hidden = hidden[0]
    hiddens = []
    for t in range(T):
      hidden = self.cell(inputs[t], hidden, x_mask, h_mask)
      hiddens.append(hidden)
    hiddens = torch.stack(hiddens)
    return hiddens, hiddens[-1].unsqueeze(0)

  def _compute_init_state(self, x, h_prev, x_mask, h_mask):
    if self.training:
      #print(x.shape)
      xh_prev = torch.cat([x * x_mask, h_prev * h_mask], dim=-1)
    else:
      xh_prev = torch.cat([x, h_prev], dim=-1)
    c0, h0 = torch.split(xh_prev.mm(self._W0), self.nhid, dim=-1)
    c0 = c0.sigmoid()
    h0 = h0.tanh()
    s0 = h_prev + c0 * (h0-h_prev)
    return s0

  def _get_activation(self, name):
    if name == 'tanh':
      f = F.tanh
    elif name == 'relu':
      f = F.relu
    elif name == 'sigmoid':
      f = F.sigmoid
    elif name == 'identity':
      f = lambda x: x
    else:
      raise NotImplementedError
    return f

  def cell(self, x, h_prev, x_mask, h_mask):
    s0 = self._compute_init_state(x, h_prev, x_mask, h_mask)

    states = [s0]
    for i, (name, pred) in enumerate(self.genotype.recurrent):
      s_prev = states[pred]
      if self.training:
        ch = (s_prev * h_mask).mm(self._Ws[i])
      else:
        ch = s_prev.mm(self._Ws[i])
      c, h = torch.split(ch, self.nhid, dim=-1)
      c = c.sigmoid()
      fn = self._get_activation(name)
      h = fn(h)
      s = s_prev + c * (h-s_prev)
      states += [s]
    output = torch.mean(torch.stack([states[i] for i in self.genotype.concat], -1), -1)
    return output


class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, ntoken, ninp, nhid, nhidlast,
                 dropout=0.5, dropouth=0.5, dropoutx=0.5, dropouti=0.5,
                 dropoute=0.1, nner=None, npos=None, token_emb_path=None,
                 nclasses=None, cell_cls=DARTSCell, genotype=None):
        super(RNNModel, self).__init__()
        # TACRED attributes
        self.nner = nner
        self.npos = npos
        self.nhid = nhid
        self.token_emb_path = token_emb_path
        self.nclasses = nclasses
        # Original attributes
        self.ninp = ninp
        self.nhidlast = nhidlast
        self.dropout = dropout
        self.dropouti = dropouti
        self.dropoute = dropoute
        self.ntoken = ntoken
        self.cell_cls = cell_cls

        self.lockdrop = LockedDropout()
        self.encoder = nn.Embedding(ntoken, ninp, padding_idx=constant.PAD_ID)
        self.peripheral_emb_dim = 0
        if self.nner is not None:
            self.ner_encoder = nn.Embedding(len(constant.NER_TO_ID), self.nner,
                                             padding_idx=constant.PAD_ID)
            self.peripheral_emb_dim += self.nner
        if self.npos is not None:
            self.npos_encoder = nn.Embedding(len(constant.POS_TO_ID), self.npos,
                                             padding_idx=constant.PAD_ID)
            self.peripheral_emb_dim += self.npos
        # If using additional token attributes, need to encode them using smaller size
        if self.peripheral_emb_dim > 0:
            input_dim = self.ninp + self.peripheral_emb_dim
            self.input_aggregator = nn.Linear(in_features=input_dim, out_features=self.ninp)

        assert ninp == nhid == nhidlast
        if cell_cls == DARTSCell:
            assert genotype is not None
            self.rnns = [cell_cls(ninp, nhid, dropouth, dropoutx, genotype)]
        else:
            assert genotype is None
            self.rnns = [cell_cls(ninp, nhid, dropouth, dropoutx)]

        self.rnns = torch.nn.ModuleList(self.rnns)
        self.decoder = nn.Linear(ninp, nclasses)
        # self.decoder.weight = self.encoder.weight
        self.init_weights()

    def init_weights(self):
        # initialize token embedding with pretrained GLoVE embeddings
        if self.token_emb_path is not None:
            token_emb_matrix = np.load(self.token_emb_path)
            token_emb_matrix = torch.from_numpy(token_emb_matrix)
            self.encoder.weight.data.copy_(token_emb_matrix)
        else:
            # keep padding dimension zero
            self.encoder.weight.data[1:, :].uniform_(-INITRANGE, INITRANGE)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-INITRANGE, INITRANGE)
        # Should create further token embedding matrices
        if self.npos is not None:
            self.npos_encoder.weight.data[1:, :].uniform_(-INITRANGE, INITRANGE)
        if self.nner is not None:
            self.ner_encoder.weight.data[1:, :].uniform_(-INITRANGE, INITRANGE)
        if self.peripheral_emb_dim > 0:
            self.input_aggregator.weight.data.uniform_(-INITRANGE, INITRANGE)

    def forward(self, input, hidden, return_h=False):
        tokens = input['tokens']
        masks = input['masks']
        pos = input['pos']
        ner = input['ner']
        #print('tokens shape: {} | pos: {} | ner: {}'.format(tokens.shape, pos.shape, ner.shape))
        batch_size = tokens.size(0)

        # emb = embedded_dropout(self.encoder, tokens, dropout=self.dropoute if self.training else 0)
        # collect all input types
        emb = self.encoder(tokens)
        input_types = [emb]
        if self.nner is not None:
            ner_emb = self.ner_encoder(ner)
            input_types.append(ner_emb)
        if self.npos is not None:
            pos_emb = self.npos_encoder(pos)
            input_types.append(pos_emb)
        combined_input = torch.cat(input_types, dim=2)
        # Reduce inputs to expected size
        emb = self.input_aggregator(combined_input)
        emb = self.lockdrop(emb, self.dropouti)
        emb = torch.transpose(emb, 1, 0)
        raw_output = emb
        new_hidden = []
        raw_outputs = []
        outputs = []
        #print('aggregated emb shape: {}'.format(emb.shape))
        # [B, T] --> [T, B]
        #masks = torch.from_numpy(np.array(masks)).t()
        masks = masks.t()
        #print(masks.shape)
        # [T, B] --> [T, B, 1]
        masks = masks.unsqueeze(2)
        for l, rnn in enumerate(self.rnns):
            # [T, B, E], [B, E]
            raw_output, new_h = rnn(raw_output, hidden)#[l])
            # mask out padded entries
            raw_output = raw_output * masks
            new_hidden.append(new_h)
            raw_outputs.append(raw_output)
        hidden = new_hidden

        output = self.lockdrop(raw_output, self.dropout)
        # [T, B, E] -> [B, E]
        output = torch.mean(output, dim=0)
        outputs.append(output)
        # TODO: This should be replaced by an "aggregation" operation <-- PA attention? and
        #  also should have masking for PADding ends of shorter sequences
        # [B, E] -> [B, C]
        logit = self.decoder(output)
        log_prob = nn.functional.log_softmax(logit, dim=-1)
        model_output = log_prob
        model_output = model_output.view(batch_size, self.nclasses)

        if return_h:
            return model_output, hidden, raw_outputs, outputs
        return model_output, hidden

    def init_hidden(self, bsz):
      weight = next(self.parameters()).data
      return [Variable(weight.new(1, bsz, self.nhid).zero_())]

