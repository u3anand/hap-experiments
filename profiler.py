import os
from sys import argv

import math
from configure import Config
import torch
import torch.fx
import collectives
import time
import hap
from argparser import parse_args
import json

from utils import get_data, get_model, input_shape, wrap_model_layers

def eprint(*args, **kwargs):
    import sys
    print(*args, file=sys.stderr, **kwargs)

class FlopsProfiler:
    def __init__(self, model: torch.fx.GraphModule, config, *input_data) -> None:
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-8)

        for _ in range(11):
            loss = model(*input_data)
            loss.backward()
            optimizer.step()
        torch.cuda.synchronize()

        start_time = time.time()
        loss = model(*input_data)
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        duration = time.time() - start_time

        flops = hap.stat(model, {
            "input_shape": input_shape(config)
        })

        eprint(f"Profiling finished. Total flops: {flops}, wall time: {duration}")
        self.device_flops = math.floor(flops / duration)
        eprint("device flops: ", self.device_flops)

class BandwidthProfiler:
    def __init__(self, config, ranks, skewness) -> None:
        self.bandwidth = {}
        self.skewness = skewness

        for op in (
            collectives.all_gather,
            collectives.all_gather_by_group_call,
            collectives.all_reduce,
            collectives.reduce_scatter,
            collectives.reduce_scatter_by_group_call,
            collectives.all_to_all
        ):
            estimation = []
            for size in (4*1024*1024, 16*1024*1024, 64*1024*1024, 256*1024*1024):
                ts = [ self.run_collective(config, ranks, op, size) for _ in range(5) ]
                eprint((size, sorted(ts)))
                estimation.append(size / sorted(ts)[2])
            self.bandwidth[op.__name__] = math.floor(sum(estimation) / len(estimation))
        eprint(self.bandwidth)

    def run_collective(self, config, ranks, op, size: int) -> float:
        import os
        os.environ['MASTER_ADDR'] = str(config.master_addr)
        os.environ['MASTER_PORT'] = str(config.master_port)
        os.environ['WORLD_SIZE'] = str(config.world_size)

        import torch.multiprocessing as mp
        ctx = mp.get_context('spawn')
        queue = ctx.Queue(1)

        for local_rank, global_rank in enumerate(ranks):
            ctx.Process(target=_run_collective_worker, args=(op, size, self.skewness, queue, global_rank, local_rank, config.world_size)).start()

        for p in mp.active_children():
            p.join()

        return queue.get()

def _run_collective_worker(op, size: int, skewness: float, queue, global_rank: int, local_rank: int, world_size: int):
    import torch.distributed as dist
    dist.init_process_group('nccl', rank=global_rank)

    if op is collectives.all_reduce:
        tensor = torch.rand(256, size // 1024).to(local_rank)
        op_args = ()

    if op in (collectives.all_gather, collectives.all_gather_by_group_call):
        total_length = size // 1024
        sharding_lengths = [skewness] + [1] * (world_size - 1)
        sharding_lengths = [ x / sum(sharding_lengths) for x in sharding_lengths]
        hap.sharding_round(total_length, sharding_lengths)

        tensor = torch.rand(256, sharding_lengths[global_rank]).to(local_rank)
        op_args = (1, sharding_lengths, global_rank)

    if op in (collectives.reduce_scatter, collectives.reduce_scatter_by_group_call):
        total_length = size // 1024
        sharding_lengths = [skewness] + [1] * (world_size - 1)
        sharding_lengths = [ x / sum(sharding_lengths) for x in sharding_lengths]
        hap.sharding_round(total_length, sharding_lengths)

        tensor = torch.rand(256, total_length).to(local_rank)
        op_args = (1, sharding_lengths, global_rank)

    if op is collectives.all_to_all:
        total_length = size // 1024
        split_sharding_lengths = [skewness] + [1] * (world_size - 1)
        split_sharding_lengths = [ x / sum(split_sharding_lengths) for x in split_sharding_lengths]
        hap.sharding_round(256, split_sharding_lengths)

        cat_sharding_lengths = [skewness] + [1] * (world_size - 1)
        cat_sharding_lengths = [ x / sum(cat_sharding_lengths) for x in cat_sharding_lengths]
        hap.sharding_round(total_length, cat_sharding_lengths)

        tensor = torch.rand(256, cat_sharding_lengths[global_rank]).to(local_rank)
        op_args = (0, 1, split_sharding_lengths, cat_sharding_lengths, global_rank)

    for _ in range(5): # 4 warmup rounds
        start_time = time.time()
        op(tensor, *op_args)
        torch.cuda.synchronize(local_rank)
        duration = time.time() - start_time

    if local_rank == 0:
        queue.put(duration)
        

def save_results(machine_name, model_name, batch_size, data, is_flops=False):
    """
    Save results to ./profiler_data/{}
    """
    results = {}
    if is_flops:
        file_path = f"/profiler_data/flops_config.json"
    else:
        file_path = f"/profiler_data/bandwidth_config.json"
    
    # Attempt to load existing configurations, if the file already exists
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            results = json.load(f)
    
    if not is_flops:
        results[machine_name] = data
    else:
        if machine_name not in results:
            results[machine_name] = {}
        
        if model_name not in results[machine_name]:
            results[machine_name][model_name] = {}
        
        results[machine_name][model_name][batch_size] = data

    with open(file_path, 'w') as f:
        json.dump(results, f, indent=4)
        print(f"Configuration saved to {file_path}")


if __name__ == '__main__':
    args = parse_args()
    ranks = args.ranks
    machine = args.machine
    config_file = args.config_file
    config = Config.from_json(config_file)
    
    if args.profile_bandwidth:
        skewness = args.skewness
        profiler = BandwidthProfiler(config, ranks, skewness)
        save_results(args.machine, config.model_name, config.batch_size, data=profiler.bandwidth)
        
    if args.profile_flops:
        flop_results = []
        for device_id in range(torch.cuda.device_count()):
            torch.cuda.set_device(device_id)
            model = get_model(config)
            # for profiling we only care about ratio of compute, so don't need to profile whole model
            # delete layers so we don't run OOM
            del model.layers[5:]
            if args.use_checkpointing:
                wrap_model_layers(model)
            model = hap.trace(model).cuda(device_id)
            x, y = next(get_data(config)[1])
            x, y = x.cuda(device_id), y.cuda(device_id)
            profiler = FlopsProfiler(model, config, x, y)
            flop_results.append(profiler.device_flops)
        save_results(args.machine, config.model_name, config.batch_size, data=flop_results, is_flops=True)
