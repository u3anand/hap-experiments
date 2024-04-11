import torch
import torchvision
import wikitext.data as data
import sys
from torch.distributed.algorithms._checkpoint import checkpoint_wrapper

def get_model(config, seed=None):
    import models

    if seed is not None:
        torch.manual_seed(seed)

    if config.model_name.startswith('Tmlp'):
        return models.TMLP(nhid=config.emsize, nlayers=config.nlayers, segmentation=config.segmentation)
    if config.model_name.startswith('Tmlp2'):
        return models.TMLP2(nhid=config.emsize, nlayers=config.nlayers, segmentation=config.segmentation)
    if config.model_name.startswith('Ttransformer'):
        return models.TTransformer(emsize=config.emsize, nheads=config.nheads, nhid=config.nhid, dropout=config.dropout, nlayers=config.nlayers, segmentation=config.segmentation)
    if config.model_name.startswith('Tmoe'):
        return models.TMoE(emsize=config.emsize, nheads=config.nheads, nhid=config.nhid, dropout=config.dropout, n_expert=config.n_expert, capacity=config.capacity, nlayers=config.nlayers, segmentation=config.segmentation)

    if config.model_name.startswith('Rtransformer'):
        ntokens, *_ = get_data(config)
        return models.RTransformer(ntokens=ntokens, seqlen=config.seqlen, emsize=config.emsize, nheads=config.nheads, nhid=config.nhid, dropout=config.dropout, nlayers=config.nlayers, segmentation=config.segmentation)
    if config.model_name.startswith('Rmoe'):
        ntokens, *_ = get_data(config)
        return models.RMoE(ntokens=ntokens, seqlen=config.seqlen, emsize=config.emsize, nheads=config.nheads, nhid=config.nhid, dropout=config.dropout, n_expert=config.n_expert, capacity=config.capacity, nlayers=config.nlayers, segmentation=config.segmentation)
    if config.model_name.startswith('Rswitch'):
        ntokens, *_ = get_data(config)
        return models.RSwitch(ntokens=ntokens, seqlen=config.seqlen, emsize=config.emsize, nheads=config.nheads, nhid=config.nhid, dropout=config.dropout, n_expert=config.n_expert, capacity=config.capacity, nlayers=config.nlayers, segmentation=config.segmentation)

    if config.model_name.startswith('Vtransformer'):
        nclasses, *_ = get_data(config)
        return models.VTransformer(nclasses=nclasses, seqlen=config.seqlen, emsize=config.emsize, nheads=config.nheads, nhid=config.nhid, dropout=config.dropout, nlayers=config.nlayers, segmentation=config.segmentation, image_size=config.image_size, patch_size=config.patch_size)
    if config.model_name.startswith('Vmoe'):
        nclasses, *_ = get_data(config)
        return models.VMoE(nclasses=nclasses, seqlen=config.seqlen, emsize=config.emsize, nheads=config.nheads, nhid=config.nhid, dropout=config.dropout, n_expert=config.n_expert, capacity=config.capacity, nlayers=config.nlayers, segmentation=config.segmentation)
    if config.model_name.startswith('Vswitch'):
        nclasses, *_ = get_data(config)
        return models.VSwitch(nclasses=nclasses, seqlen=config.seqlen, emsize=config.emsize, nheads=config.nheads, nhid=config.nhid, dropout=config.dropout, n_expert=config.n_expert, capacity=config.capacity, nlayers=config.nlayers, segmentation=config.segmentation)
    if config.model_name.startswith('Vvgg'):
        nclasses, *_ = get_data(config)
        return models.VVGG(nclasses=nclasses, dropout=config.dropout, segmentation=config.segmentation)
    
def get_data(config):
    if config.model_name.startswith('R'):
        return wikitext2(config)

    if config.model_name.startswith('V'):
        # return cifar10(config)
        return get_image_dataset(config)

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

def get_image_dataset(config):
    def it(data):
        loader = torch.utils.data.DataLoader(data, batch_size=config.batch_size, drop_last=True)
        while True:
            yield from iter(loader)
    num_channels = 3  # RGB
    image_size = config.image_size
    num_batches = config.run_iter * config.batch_size
    size = (num_batches, num_channels, image_size, image_size)
    images = torch.randn(size, device="cuda")
    targets = torch.randint(
        low=0, high=2, size=(num_batches,), device="cuda"
    )
    images2 = torch.randn(size, device="cuda")
    targets2 = torch.randint(
        low=0, high=2, size=(num_batches,), device="cuda"
    )
    dataset = torch.utils.data.TensorDataset(images, targets)
    dataset2 = torch.utils.data.TensorDataset(images2, targets2)
    return 10, it(dataset), it(dataset2)

def input_shape(config):
    if config.model_name.startswith('R'):
        return { 'x': (config.batch_size, config.seqlen), 'y': (config.batch_size, config.seqlen) }
    if config.model_name.startswith('V'):
        return { 'x': (config.batch_size, 3, 32, 32), 'y': (config.batch_size,) }
    if config.model_name.startswith('T'):
        return { 'x': (config.batch_size, config.seqlen, config.emsize), 'y': (config.batch_size,) }
    
def wrap_model_layers(model):
    for i in range(len(model.layers)):
        model.layers[i] = checkpoint_wrapper.CheckpointWrapper(
                            model.layers[i],
                            checkpoint_impl=checkpoint_wrapper.CheckpointImpl.NO_REENTRANT,
                            preserve_rng_state=False,
                          )
