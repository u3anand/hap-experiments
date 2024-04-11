import math
import json

class Config:
    def __init__(self, config_dict):
        self.model_name = config_dict.get('model_name', 'Rtransformer')
        self.rootpath = config_dict.get('rootpath', "/home/u3anand/fall2023/hap")
        self.world_size = config_dict.get('world_size', 4)
        self.nlayers = config_dict.get('nlayers', 36)
        self.batch_size = config_dict.get('batch_size', 128) * self.world_size # Default 32*world_size
        self.image_size = config_dict.get('image_size', 32)
        self.patch_size = config_dict.get('patch_size', 4)
        self.seqlen = config_dict.get('seqlen', 512)
        self.emsize = config_dict.get('emsize', 1536)
        self.dropout = config_dict.get('dropout', 0.1)
        self.nheads = config_dict.get('nheads', 24)
        self.lr = config_dict.get('lr', 5e-4)
        self.run_iter = config_dict.get('run_iter', 100)
        self.avg_iter = config_dict.get('avg_iter', 50)
        self.log_iter = config_dict.get('log_iter', 100)
        self.segmentation = config_dict.get('segmentation', False)
        self.trace = config_dict.get('trace', False)
        self.report_per_iter_time = config_dict.get('report_per_iter_time', True)
        self.n_expert = 2 * self.world_size
        self.capacity_factor = 1.25 if not self.model_name.endswith('moe') else 2.5
        self.capacity = math.ceil(self.seqlen / self.n_expert * self.capacity_factor)
        self.nhid = self.emsize * 4
        self.master_addr = "127.0.0.1"
        self.master_port = 39266
            
    @staticmethod
    def from_json(json_input):
        if isinstance(json_input, str):
            try:
                # Assuming json_input is a file path
                with open(json_input, 'r') as file:
                    config_dict = json.load(file)
            except FileNotFoundError:
                # Assuming json_input is a JSON string
                config_dict = json.loads(json_input)
        else:
            raise ValueError("Invalid input: json_input must be a file path or JSON string")
        
        return Config(config_dict)
