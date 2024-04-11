from argparser import parse_args
import os
import json
import time
from configure import Config
from profiler import FlopsProfiler, save_results
import torch
import torch.fx
from torch.profiler import profile, record_function, ProfilerActivity
import numpy as np
import torch.multiprocessing as mp
import hap
import torch.distributed as dist
from utils import get_data, get_model, input_shape, wrap_model_layers

def eprint(*args, **kwargs):
    import sys
    print(*args, file=sys.stderr, **kwargs)

def get_device_flops_for_machine(machine_name, model_name, batch_size):
    file_path = "/profiler_data/flops_config.json"
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            results = json.load(f)
    
    try:
        flops = results[machine_name][model_name][str(batch_size)]
    except Exception as e:
        eprint("Flops data not found, please profile flops data")
        raise SystemExit
            
    return flops

def get_comm_bandwidth_for_machine(machine_name):
    file_path = "/profiler_data/bandwidth_config.json"
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            results = json.load(f)
            
    try:
        comm_data = results[machine_name]
    except Exception as e:
        eprint("Communication Data not found, please profile comm data")
        raise SystemExit
    
    return comm_data


def run(global_rank, local_rank, model, dgraph, config):
    dist.init_process_group('nccl', rank=global_rank)

    dmodel = torch.fx.GraphModule(model, dgraph).cuda(local_rank)
    del model

    optimizer = torch.optim.Adam(dmodel.parameters(), lr=config.lr)
    train_data = get_data(config)[1]

    result_times = []
    strat_time = last_iter_time = time.time()
    total_loss = 0
    tokens_processed = 0

    x, y = next(train_data)
    x = x.cuda(local_rank)
    y = y.cuda(local_rank)

    for iter in range(config.run_iter):
        optimizer.zero_grad()

        loss = dmodel(x, y)

        aggregated_loss = loss.detach().clone()
        dist.reduce(aggregated_loss, 0)

        if global_rank == 0:
            total_loss += aggregated_loss.cpu().numpy() / config.batch_size / config.seqlen
            if iter % config.log_iter == 0:
                eprint(f"loss (log ppl) {iter}: {total_loss / config.log_iter:.3f}, wall clock: {time.time() - strat_time:.3f}")
                total_loss = 0
        # dist.barrier(device_ids=[global_rank])

        loss.backward()
        torch.nn.utils.clip_grad_norm_(dmodel.parameters(), 0.5)
        # torch.cuda.synchronize()
        optimizer.step()
        # dist.barrier()
        if config.report_per_iter_time and local_rank == 0:
            iter_duration = time.time() - last_iter_time
            result_times.append(iter_duration)
            last_iter_time += iter_duration
            tokens_processed = config.batch_size * config.seqlen / iter_duration
            eprint("iter time: ", iter_duration)
            eprint("avg±std:", np.mean(result_times[-config.avg_iter:]), np.std(result_times[-config.avg_iter:]))
            eprint("Training Throughput: ", tokens_processed)
            tokens_processed = 0

    # for epoch in range(config.epoch):
    #     total_loss = 0.
    #     start_time = time.time()
    #     for batch, offset in enumerate(range(0, train_data.size(1) - config.seqlen, config.seqlen)):
    #         loss = model(
    #             x = train_data[:, offset:offset+config.seqlen],
    #             y = train_data[:, offset+1:offset+1+config.seqlen]
    #         ) / config.batch_size / config.seqlen

    #         total_loss += loss.detach()
    #         if batch % config.log_iterval == 0 and batch > 0:
    #             dist.reduce(total_loss, 0)
    #             if global_rank == 0:
    #                 avg_loss = total_loss / config.log_iterval
    #                 elapsed = time.time() - start_time
    #                 eprint(f"epoch {epoch:3d} | batch {batch:3d} | ppl {math.exp(avg_loss):02.2f} | ms/batch {elapsed*1000/config.log_iterval:5.2f}")
    #             total_loss = 0.
    #             start_time = time.time()

    #         torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)

    #         loss.backward()
    #         optimizer.step()

    if not config.trace:
        return

    # x, y = next(train_data)
    # x = x.cuda(local_rank)
    # y = y.cuda(local_rank)
    with profile(
        activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA],
        # record_shapes = True,
        # profile_memory = True,
        schedule = torch.profiler.schedule(wait=1, warmup=10, active=4)
    ) as prof:
        for _ in range(15):
            with record_function("forward"):
                loss = dmodel(x, y)
            with record_function("backward"):
                loss.backward()
                torch.cuda.synchronize()
            with record_function("update"):
                optimizer.step()
            dist.barrier()
            prof.step()

    if local_rank == 0:
        # eprint(prof.key_averages().table(sort_by="cuda_time_total"))
        prof.export_chrome_trace("trace.json")


def run_multiprocessing_setup(args, config):
    # set environment variables
    os.environ['MASTER_ADDR'] = str(config.master_addr)
    os.environ['MASTER_PORT'] = str(config.master_port)
    os.environ['WORLD_SIZE'] = str(config.world_size)
    
    models_for_trace = [get_model(config, seed=39) for _ in range(len(args.ranks))]
    if main_args.use_checkpointing:
        for i in range(len(args.ranks)):
            wrap_model_layers(models_for_trace[i])
    models = [hap.trace(m) for m in models_for_trace]
    # model = hap.trace(model_for_trace)
    device_flops = get_device_flops_for_machine(args.machine, config.model_name, config.batch_size)
    communication_bandwidth = get_comm_bandwidth_for_machine(args.machine)
    dgraphs = []
    for i, rank in enumerate(args.ranks):
        dgraph = hap.main(models[i], {
            "input_shape": input_shape(config),
            "device_flops": device_flops,
            "all_gather_bandwidth": communication_bandwidth["all_gather"],
            "all_gather_by_group_call_bandwidth": communication_bandwidth["all_gather_by_group_call"],
            "all_reduce_bandwidth": communication_bandwidth["all_reduce"],
            "reduce_scatter_bandwidth": communication_bandwidth["reduce_scatter"],
            "reduce_scatter_by_group_call_bandwidth": communication_bandwidth["reduce_scatter_by_group_call"],
            "all_to_all_bandwidth": communication_bandwidth["all_to_all"],
            "extra_ps": False,
            "group_collective": False,
            "rank": rank,
            # "sharding_ratios": [ 0.32745167869976854 ] * 2 + [ 0.17254832130023143 ] * 2,
        })
        dgraphs.append(dgraph)
        
    # launch processes    
    mp.set_start_method('spawn')

    # for local_rank, global_rank in enumerate(ranks):
    #     mp.Process(target=run, args=(global_rank, local_rank, model, config, main_args)).start()
    for local_rank, global_rank in enumerate(args.ranks):
        mp.Process(target=run, args=(global_rank, local_rank, models[local_rank], dgraphs[local_rank], config)).start()

    for p in mp.active_children():
        p.join()

if __name__ == '__main__':
    main_args = parse_args()
    config = Config.from_json(main_args.config_file)
    run_multiprocessing_setup(main_args, config)