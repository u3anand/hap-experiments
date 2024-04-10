import torch
import torchvision
import wikitext.data as data
import sys

def get_model(config, seed=None):
    import models

    if seed is not None:
        torch.manual_seed(seed)

    if config.model_name == 'Tmlp':
        return models.TMLP(nhid=config.emsize, nlayers=config.nlayers, segmentation=config.segmentation)
    if config.model_name == 'Tmlp2':
        return models.TMLP2(nhid=config.emsize, nlayers=config.nlayers, segmentation=config.segmentation)
    if config.model_name == 'Ttransformer':
        return models.TTransformer(emsize=config.emsize, nheads=config.nheads, nhid=config.nhid, dropout=config.dropout, nlayers=config.nlayers, segmentation=config.segmentation)
    if config.model_name == 'Tmoe':
        return models.TMoE(emsize=config.emsize, nheads=config.nheads, nhid=config.nhid, dropout=config.dropout, n_expert=config.n_expert, capacity=config.capacity, nlayers=config.nlayers, segmentation=config.segmentation)

    if config.model_name == 'Rtransformer':
        ntokens, *_ = get_data()
        return models.RTransformer(ntokens=ntokens, seqlen=config.seqlen, emsize=config.emsize, nheads=config.nheads, nhid=config.nhid, dropout=config.dropout, nlayers=config.nlayers, segmentation=config.segmentation)
    if config.model_name == 'Rmoe':
        ntokens, *_ = get_data()
        return models.RMoE(ntokens=ntokens, seqlen=config.seqlen, emsize=config.emsize, nheads=config.nheads, nhid=config.nhid, dropout=config.dropout, n_expert=config.n_expert, capacity=config.capacity, nlayers=config.nlayers, segmentation=config.segmentation)
    if config.model_name == 'Rswitch':
        ntokens, *_ = get_data()
        return models.RSwitch(ntokens=ntokens, seqlen=config.seqlen, emsize=config.emsize, nheads=config.nheads, nhid=config.nhid, dropout=config.dropout, n_expert=config.n_expert, capacity=config.capacity, nlayers=config.nlayers, segmentation=config.segmentation)

    if config.model_name == 'Vtransformer':
        nclasses, *_ = get_data()
        return models.VTransformer(nclasses=nclasses, seqlen=config.seqlen, emsize=config.emsize, nheads=config.nheads, nhid=config.nhid, dropout=config.dropout, nlayers=config.nlayers, segmentation=config.segmentation)
    if config.model_name == 'Vmoe':
        nclasses, *_ = get_data()
        return models.VMoE(nclasses=nclasses, seqlen=config.seqlen, emsize=config.emsize, nheads=config.nheads, nhid=config.nhid, dropout=config.dropout, n_expert=config.n_expert, capacity=config.capacity, nlayers=config.nlayers, segmentation=config.segmentation)
    if config.model_name == 'Vswitch':
        nclasses, *_ = get_data()
        return models.VSwitch(nclasses=nclasses, seqlen=config.seqlen, emsize=config.emsize, nheads=config.nheads, nhid=config.nhid, dropout=config.dropout, n_expert=config.n_expert, capacity=config.capacity, nlayers=config.nlayers, segmentation=config.segmentation)
    if config.model_name == 'Vvgg':
        nclasses, *_ = get_data()
        return models.VVGG(nclasses=nclasses, dropout=config.dropout, segmentation=config.segmentation)
    
def get_data(config):
    if config.model_name.startswith('R'):
        return wikitext2()

    if config.model_name.startswith('V'):
        return cifar10()

    if config.model_name.startswith('T'):
        x = torch.rand(config.batch_size, config.seqlen, config.emsize) / 6
        y = torch.rand(config.batch_size)
        def rep():
            while True:
                yield x, y
        return 0, rep()
    
def wikitext2(config):
    sys.path.insert(1, f"{config.rootpath}/wikitext")
    
    corpus = data.Corpus(f"{config.rootpath}/wikitext")
    train_data = data.segmentify(data.batchify(corpus.train, config.batch_size), config.seqlen)
    test_data = data.segmentify(data.batchify(corpus.test, config.batch_size), config.seqlen)
    valid_data = data.segmentify(data.batchify(corpus.valid, config.batch_size), config.seqlen)
    ntokens = config.world_size * (len(corpus.dictionary) // config.world_size + 1) # we have to ensure that it is dividable
    return ntokens, train_data, test_data, valid_data

def cifar10(config):
    def it(data):
        loader = torch.utils.data.DataLoader(data, batch_size=config.batch_size, drop_last=True)
        while True:
            yield from iter(loader)
    train_data = torchvision.datasets.CIFAR10(f"{config.rootpath}/cifar10", train=True, transform=torchvision.transforms.ToTensor()) #, download=True
    test_data = torchvision.datasets.CIFAR10(f"{config.rootpath}/cifar10", train=False, transform=torchvision.transforms.ToTensor()) #, download=True
    return 10, it(train_data), it(test_data)

def input_shape(config):
    if config.model_name.startswith('R'):
        return { 'x': (config.batch_size, config.seqlen), 'y': (config.batch_size, config.seqlen) }
    if config.model_name.startswith('V'):
        return { 'x': (config.batch_size, 3, 32, 32), 'y': (config.batch_size,) }
    if config.model_name.startswith('T'):
        return { 'x': (config.batch_size, config.seqlen, config.emsize), 'y': (config.batch_size,) }
