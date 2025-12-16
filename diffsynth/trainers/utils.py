import imageio, os, torch, warnings, torchvision, argparse, json
from peft import LoraConfig, inject_adapter_in_model
from PIL import Image
import pandas as pd
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
import glob
import hashlib
import tempfile
import subprocess
import re
# from data.pyslam.test.thirdparty.test_superpoint_lightflue import video_path


class ImageDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_path=None, metadata_path=None,
        max_pixels=1920*1080, height=None, width=None,
        height_division_factor=16, width_division_factor=16,
        data_file_keys=("image",),
        image_file_extension=("jpg", "jpeg", "png", "webp"),
        repeat=1,
        args=None,
    ):
        if args is not None:
            base_path = args.dataset_base_path
            metadata_path = args.dataset_metadata_path
            height = args.height
            width = args.width
            max_pixels = args.max_pixels
            data_file_keys = args.data_file_keys.split(",")
            repeat = args.dataset_repeat
            
        self.base_path = base_path
        self.max_pixels = max_pixels
        self.height = height
        self.width = width
        self.height_division_factor = height_division_factor
        self.width_division_factor = width_division_factor
        self.data_file_keys = data_file_keys
        self.image_file_extension = image_file_extension
        self.repeat = repeat

        if height is not None and width is not None:
            print("Height and width are fixed. Setting `dynamic_resolution` to False.")
            self.dynamic_resolution = False
        elif height is None and width is None:
            print("Height and width are none. Setting `dynamic_resolution` to True.")
            self.dynamic_resolution = True
            
        if metadata_path is None:
            print("No metadata. Trying to generate it.")
            metadata = self.generate_metadata(base_path)
            print(f"{len(metadata)} lines in metadata.")
            self.data = [metadata.iloc[i].to_dict() for i in range(len(metadata))]
        elif metadata_path.endswith(".json"):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            self.data = metadata
        elif metadata_path.endswith(".jsonl"):
            metadata = []
            with open(metadata_path, 'r') as f:
                for line in tqdm(f):
                    metadata.append(json.loads(line.strip()))
            self.data = metadata
        else:
            metadata = pd.read_csv(metadata_path)
            self.data = [metadata.iloc[i].to_dict() for i in range(len(metadata))]


    def generate_metadata(self, folder):
        image_list, prompt_list = [], []
        file_set = set(os.listdir(folder))
        for file_name in file_set:
            if "." not in file_name:
                continue
            file_ext_name = file_name.split(".")[-1].lower()
            file_base_name = file_name[:-len(file_ext_name)-1]
            if file_ext_name not in self.image_file_extension:
                continue
            prompt_file_name = file_base_name + ".txt"
            if prompt_file_name not in file_set:
                continue
            with open(os.path.join(folder, prompt_file_name), "r", encoding="utf-8") as f:
                prompt = f.read().strip()
            image_list.append(file_name)
            prompt_list.append(prompt)
        metadata = pd.DataFrame()
        metadata["image"] = image_list
        metadata["prompt"] = prompt_list
        return metadata
    
    
    def crop_and_resize(self, image, target_height, target_width):
        width, height = image.size
        scale = max(target_width / width, target_height / height)
        image = torchvision.transforms.functional.resize(
            image,
            (round(height*scale), round(width*scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        image = torchvision.transforms.functional.center_crop(image, (target_height, target_width))
        return image
    
    
    def get_height_width(self, image):
        if self.dynamic_resolution:
            width, height = image.size
            if width * height > self.max_pixels:
                scale = (width * height / self.max_pixels) ** 0.5
                height, width = int(height / scale), int(width / scale)
            height = height // self.height_division_factor * self.height_division_factor
            width = width // self.width_division_factor * self.width_division_factor
        else:
            height, width = self.height, self.width
        return height, width
    
    
    def load_image(self, file_path):
        image = Image.open(file_path).convert("RGB")
        image = self.crop_and_resize(image, *self.get_height_width(image))
        return image
    
    
    def load_data(self, file_path):
        return self.load_image(file_path)


    def __getitem__(self, data_id):
        data = self.data[data_id % len(self.data)].copy()
        for key in self.data_file_keys:
            if key in data:
                if isinstance(data[key], list):
                    path = [os.path.join(self.base_path, p) for p in data[key]]
                    data[key] = [self.load_data(p) for p in path]
                else:
                    path = os.path.join(self.base_path, data[key])
                    data[key] = self.load_data(path)
                if data[key] is None:
                    warnings.warn(f"cannot load file {data[key]}.")
                    return None
        return data
    

    def __len__(self):
        return len(self.data) * self.repeat



class VideoDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_path=None, metadata_path=None,
        num_frames=81,
        time_division_factor=4, time_division_remainder=1,
        max_pixels=1920*1080, height=None, width=None,
        height_division_factor=16, width_division_factor=16,
        data_file_keys=("video",),
        image_file_extension=("jpg", "jpeg", "png", "webp"),
        video_file_extension=("mp4", "avi", "mov", "wmv", "mkv", "flv", "webm"),
        cache_file_extension=("pth"),
        repeat=1,
        args=None,
        accelerator=None
    ):
        if args is not None:
            base_path = args.dataset_base_path
            metadata_path = args.dataset_metadata_path
            height = args.height
            width = args.width
            max_pixels = args.max_pixels
            num_frames = args.num_frames
            data_file_keys = args.data_file_keys.split(",")
            repeat = args.dataset_repeat
        
        self.base_path = base_path
        self.num_frames = num_frames
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        self.max_pixels = max_pixels
        self.height = height
        self.width = width
        self.height_division_factor = height_division_factor
        self.width_division_factor = width_division_factor
        self.data_file_keys = data_file_keys
        self.image_file_extension = image_file_extension
        self.video_file_extension = video_file_extension
        self.cache_file_extension = cache_file_extension
        self.repeat = repeat

        self.args = args

        if height is not None and width is not None:
            print("Height and width are fixed. Setting `dynamic_resolution` to False.")
            self.dynamic_resolution = False
        elif height is None and width is None:
            print("Height and width are none. Setting `dynamic_resolution` to True.")
            self.dynamic_resolution = True

        if args.task == 'train' and args.use_cache:
            data = []
            if args.from_cos:

                # def run_with_retry(cmd, max_retries=10, delay=2):
                #     import time
                #     for attempt in range(max_retries):
                #         try:
                #             # 执行命令
                #             result = subprocess.run(
                #                 cmd,
                #                 capture_output=True,
                #                 text=True,
                #                 check=True
                #             )
                #             # 成功执行则返回结果
                #             return result
                #         except subprocess.CalledProcessError as e:
                #             # 打印错误信息和重试提示
                #             print(f"尝试 {attempt + 1}/{max_retries} 失败:")
                #             print(f"命令: {e.cmd}")
                #             print(f"返回码: {e.returncode}")
                #             print(f"错误输出: {e.stderr}")
                #
                #             # 如果不是最后一次尝试，等待后重试
                #             if attempt < max_retries - 1:
                #                 print(f"将在 {delay} 秒后重试...\n")
                #                 time.sleep(delay)
                #
                #     # 超过最大重试次数
                #     raise subprocess.CalledProcessError(
                #         returncode=e.returncode,
                #         cmd=e.cmd,
                #         output=e.output,
                #         stderr=e.stderr
                #     )
                #
                cmd = [
                    'coscli',
                    'ls',
                    f'{args.cos_root}/{self.base_path}',
                    '-r',
                    '--include', '.*\.pth$',
                    '--limit', '100000000',
                ]
                # rank = int(os.environ.get('RANK', 0))
                rank = accelerator.process_index
                print(f"***********************rank: {rank}")
                if rank == 0:
                    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                    pth_pattern = r'data/.*?\.pth'
                    cache_paths = re.findall(pth_pattern, result.stdout)
                else:
                    cache_paths = None

                    # 广播数据到所有进程
                if rank == 0:
                    # 发送方：准备要广播的对象列表
                    obj_list = [cache_paths]
                    torch.distributed.broadcast_object_list(obj_list, src=0)
                else:
                    # 接收方：创建接收列表
                    obj_list = [None]
                    torch.distributed.broadcast_object_list(obj_list, src=0)
                    cache_paths = obj_list[0]
                    # 验证接收结果
                    assert cache_paths is not None, f"Rank {rank} 未收到数据"
                    print(f"Rank {rank} 成功接收 {len(cache_paths)} 个文件路径")

                # result = run_with_retry(cmd, max_retries=10)
                # result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                # pth_pattern = r'data/.*?\.pth'
                # cache_paths = re.findall(pth_pattern, result.stdout)
            else:
                cache_paths = glob.glob(f"{self.base_path}/**/*.pth", recursive=True)
            cache_relpaths = [os.path.relpath(path, self.base_path) for path in cache_paths]
            for cache_relpath in cache_relpaths:
                _ = {'video': cache_relpath, 'prompt': ''}
                data.append(_)
            self.data = data
            print(f"==> using {len(self.data)} files for training.")
            pass
        else:
            if metadata_path is None:
                print("No metadata. Trying to generate it.")
                metadata = self.generate_metadata(base_path)
                print(f"{len(metadata)} lines in metadata.")
                self.data = [metadata.iloc[i].to_dict() for i in range(len(metadata))]
            elif metadata_path.endswith(".json"):
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                self.data = metadata
            else:
                metadata = pd.read_csv(metadata_path)
                self.data = [metadata.iloc[i].to_dict() for i in range(len(metadata))]

            # data_process
            if args.task == 'data_process':
                print(f"==> find {len(self.data)} files for data processing.")
                self.data = self.data[args.idx_s:] if args.idx_e == -1 else self.data[args.idx_s:args.idx_e]
                print(f"==> using {len(self.data)} files for data processing.")
                data = []
                if args.sovled_video_json is not None:
                    with open(args.sovled_video_json, 'r', encoding='utf-8') as f:
                        cache_relpaths = json.load(f)
                        print(f"==> find {len(cache_relpaths)} solved files from {args.sovled_video_json}.")
                else:
                    cache_paths = glob.glob(f"{args.data_process_output_path}/**/*.pth", recursive=True)
                    cache_relpaths = [os.path.relpath(path, args.data_process_output_path) for path in cache_paths]
                    cache_relpaths = set(cache_relpaths)
                    print(f"==> find {len(cache_relpaths)} solved files.")
                for _ in self.data:
                    if f"{_['video'].split('.')[0]}_{hashlib.md5(_['prompt'].encode()).hexdigest()}.pth" in cache_relpaths:
                        # print(f"File {_['video'].split('.')[0]}_{hashlib.md5(_['prompt'].encode()).hexdigest()}.pth already exists. Skipping processing.")
                        continue
                    data.append(_)
                self.data = data
                print(f"==> using {len(self.data)} unsolved files for data processing.")
                # self.data.sort()
                # print(f"==> find {len(self.data)} files for data processing.")
                # self.data = self.data[args.idx_s:] if args.idx_e == -1 else self.data[args.idx_s:args.idx_e]
                # self.data = [f"{_['video'].split('.')[0]}_{hashlib.md5(_['prompt'].encode()).hexdigest()}.pth" for _ in
                #              self.data]
                # print(f"****************1 {self.data[0]}")
                # print(f"==> using {len(self.data)} files for data processing.")
                # data = []
                # cache_paths = glob.glob(f"{args.data_process_output_path}/**/*.pth", recursive=True)
                # cache_relpaths = [os.path.relpath(path, args.data_process_output_path) for path in cache_paths]
                # print(f"****************2 {cache_relpaths[0]}")
                # print(f"==> find {len(cache_relpaths)} solved files.")
                # # for _ in self.data:
                # #     if _ in cache_relpaths:
                # #         print(f"{_} already exists. Skipping processing.")
                # #         continue
                # #     data.append(_)
                # # self.data = data
                # cache_set = set(cache_relpaths)
                # self.data = [item for item in self.data if item not in cache_set]
                # print(f"==> using {len(self.data)} unsolved files for data processing.")
            elif args.task == 'train':
                print(f"==> using {len(self.data)} files for training.")
            else:
                pass

            # elif args.task == 'train':
            #     if args.use_cache:
            #         data = []
            #         cache_paths = glob.glob(f"{self.base_path}/**/*.pth", recursive=True)
            #         cache_relpaths = [os.path.relpath(path, self.base_path) for path in cache_paths]
            #         for _ in self.data:
            #             if f"{_['video'].split('.')[0]}_{hashlib.md5(_['prompt'].encode()).hexdigest()}.pth" in cache_relpaths:
            #                 _['video'] = f"{_['video'].split('.')[0]}_{hashlib.md5(_['prompt'].encode()).hexdigest()}.pth"
            #                 data.append(_)
            #         self.data = data
            #         pass

    def generate_metadata(self, folder):
        video_list, prompt_list = [], []
        file_set = set(os.listdir(folder))
        for file_name in file_set:
            if "." not in file_name:
                continue
            file_ext_name = file_name.split(".")[-1].lower()
            file_base_name = file_name[:-len(file_ext_name)-1]
            if file_ext_name not in self.image_file_extension and file_ext_name not in self.video_file_extension:
                continue
            prompt_file_name = file_base_name + ".txt"
            if prompt_file_name not in file_set:
                continue
            with open(os.path.join(folder, prompt_file_name), "r", encoding="utf-8") as f:
                prompt = f.read().strip()
            video_list.append(file_name)
            prompt_list.append(prompt)
        metadata = pd.DataFrame()
        metadata["video"] = video_list
        metadata["prompt"] = prompt_list
        return metadata
        
        
    def crop_and_resize(self, image, target_height, target_width):
        width, height = image.size
        scale = max(target_width / width, target_height / height)
        image = torchvision.transforms.functional.resize(
            image,
            (round(height*scale), round(width*scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        image = torchvision.transforms.functional.center_crop(image, (target_height, target_width))
        return image
    
    
    def get_height_width(self, image):
        if self.dynamic_resolution:
            width, height = image.size
            if width * height > self.max_pixels:
                scale = (width * height / self.max_pixels) ** 0.5
                height, width = int(height / scale), int(width / scale)
            height = height // self.height_division_factor * self.height_division_factor
            width = width // self.width_division_factor * self.width_division_factor
        else:
            height, width = self.height, self.width
        return height, width
    
    
    # def get_num_frames(self, reader):
    #     num_frames = self.num_frames
    #     if int(reader.count_frames()) < num_frames:
    #         num_frames = int(reader.count_frames())
    #         while num_frames > 1 and num_frames % self.time_division_factor != self.time_division_remainder:
    #             num_frames -= 1
    #     return num_frames
    def get_num_frames(self, reader):
        num_frames = self.num_frames
        if int(reader.count_frames()) <= num_frames:
            num_frames = int(reader.count_frames())
            while num_frames > 1 and num_frames % self.time_division_factor != self.time_division_remainder:
                num_frames -= 1
            return 0, num_frames
        else:
            start_frame_id = torch.randint(0, int(reader.count_frames()) - self.num_frames, (1,))[0]
            return start_frame_id, num_frames + start_frame_id
            # return 0, num_frames


    def load_video(self, file_path):
        reader = imageio.get_reader(file_path)
        start_frame_id, num_frames = self.get_num_frames(reader)
        frames = []
        for frame_id in range(start_frame_id, num_frames , 1):
            frame = reader.get_data(frame_id)
            frame = Image.fromarray(frame)
            frame = self.crop_and_resize(frame, *self.get_height_width(frame))
            frames.append(frame)
        reader.close()
        return frames
    
    
    def load_image(self, file_path):
        image = Image.open(file_path).convert("RGB")
        image = self.crop_and_resize(image, *self.get_height_width(image))
        frames = [image]
        return frames

    def load_cache(self, file_path):
        cache = torch.load(file_path)
        return cache
    
    def is_image(self, file_path):
        file_ext_name = file_path.split(".")[-1]
        return file_ext_name.lower() in self.image_file_extension
    
    
    def is_video(self, file_path):
        file_ext_name = file_path.split(".")[-1]
        return file_ext_name.lower() in self.video_file_extension
    
    def is_cache(self, file_path):
        file_ext_name = file_path.split(".")[-1]
        return file_ext_name.lower() in self.cache_file_extension

    def load_data(self, file_path):
        if self.is_image(file_path):
            return self.load_image(file_path)
        elif self.is_video(file_path):
            return self.load_video(file_path)
        elif self.is_cache(file_path):
            return self.load_cache(file_path)
        else:
            return None


    def __getitem__(self, data_id):
        try:
            data = self.data[data_id % len(self.data)].copy()
            for key in self.data_file_keys:
                if key in data:
                    if self.args.from_cos:
                        with tempfile.TemporaryDirectory(prefix=self.args.cos_tmp_dir) as temp_dir:
                            path = f'{temp_dir}/{self.base_path}/{data[key]}'
                            cmd = [
                                'coscli',
                                'cp',
                                f'{self.args.cos_root}/{self.base_path}/{data[key]}',
                                path,
                            ]
                            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                            data[f"{key}_path"] = data[key]
                            data[key] = self.load_data(path)
                            if data[key] is None:
                                warnings.warn(f"cannot load file {data[key]}.")
                                return None
                    else:
                        path = os.path.join(self.base_path, data[key])
                        data[f"{key}_path"] = data[key]
                        data[key] = self.load_data(path)
                        if data[key] is None:
                            warnings.warn(f"cannot load file {data[key]}.")
                            return None
            return data
        except Exception as e:
            return None

    def __len__(self):
        return len(self.data) * self.repeat



class DiffusionTrainingModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        
    def to(self, *args, **kwargs):
        for name, model in self.named_children():
            model.to(*args, **kwargs)
        return self
        
        
    def trainable_modules(self):
        trainable_modules = filter(lambda p: p.requires_grad, self.parameters())
        return trainable_modules
    
    
    def trainable_param_names(self):
        trainable_param_names = list(filter(lambda named_param: named_param[1].requires_grad, self.named_parameters()))
        trainable_param_names = set([named_param[0] for named_param in trainable_param_names])
        return trainable_param_names
    
    
    def add_lora_to_model(self, model, target_modules, lora_rank, lora_alpha=None):
        if lora_alpha is None:
            lora_alpha = lora_rank
        lora_config = LoraConfig(r=lora_rank, lora_alpha=lora_alpha, target_modules=target_modules)
        model = inject_adapter_in_model(lora_config, model)
        return model
    
    
    def export_trainable_state_dict(self, state_dict, remove_prefix=None):
        trainable_param_names = self.trainable_param_names()
        state_dict = {name: param for name, param in state_dict.items() if name in trainable_param_names}
        if remove_prefix is not None:
            state_dict_ = {}
            for name, param in state_dict.items():
                if name.startswith(remove_prefix):
                    name = name[len(remove_prefix):]
                state_dict_[name] = param
            state_dict = state_dict_
        return state_dict


class ModelLogger:
    def __init__(self, output_path, remove_prefix_in_ckpt=None, state_dict_converter=lambda x:x, num_steps=0):
        self.output_path = output_path
        self.remove_prefix_in_ckpt = remove_prefix_in_ckpt
        self.state_dict_converter = state_dict_converter
        self.num_steps = num_steps


    def on_step_end(self, accelerator, model, save_steps=None):
        self.num_steps += 1
        if save_steps is not None and self.num_steps % save_steps == 0:
            self.save_model(accelerator, model, f"step-{self.num_steps}.safetensors")


    def on_epoch_end(self, accelerator, model, epoch_id):
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            state_dict = accelerator.get_state_dict(model)
            state_dict = accelerator.unwrap_model(model).export_trainable_state_dict(state_dict, remove_prefix=self.remove_prefix_in_ckpt)
            state_dict = self.state_dict_converter(state_dict)
            os.makedirs(self.output_path, exist_ok=True)
            path = os.path.join(self.output_path, f"epoch-{epoch_id}.safetensors")
            accelerator.save(state_dict, path, safe_serialization=True)


    def on_training_end(self, accelerator, model, save_steps=None):
        if save_steps is not None and self.num_steps % save_steps != 0:
            self.save_model(accelerator, model, f"step-{self.num_steps}.safetensors")


    def save_model(self, accelerator, model, file_name):
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            state_dict = accelerator.get_state_dict(model)
            state_dict = accelerator.unwrap_model(model).export_trainable_state_dict(state_dict, remove_prefix=self.remove_prefix_in_ckpt)
            state_dict = self.state_dict_converter(state_dict)
            os.makedirs(self.output_path, exist_ok=True)
            path = os.path.join(self.output_path, file_name)
            accelerator.save(state_dict, path, safe_serialization=True)


def launch_training_task(
    # dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    num_workers: int = 8,
    save_steps: int = None,
    num_epochs: int = 1,
    gradient_accumulation_steps: int = 1,
    find_unused_parameters: bool = False,
    use_cache: bool = False,
    args=None
):
    print(f'***********gradient_accumulation_steps: {gradient_accumulation_steps}')
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=find_unused_parameters)],
    )
    dataset = VideoDataset(args=args, accelerator=accelerator)
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, collate_fn=lambda x: x[0], num_workers=num_workers)
    model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)

    import torch.distributed as dist
    def all_ranks_ok(local_ok: int, device):
        """
        local_ok: 1 表示本 rank 正常，0 表示本 rank 出问题
        返回 True 当且仅当所有 rank 都为 1
        """
        tensor = torch.tensor(local_ok, device=device, dtype=torch.int)
        if dist.is_initialized():
            dist.all_reduce(tensor, op=dist.ReduceOp.MIN)
        return bool(tensor.item())

    def any_rank_bad_int(tensor_int, device):
        # tensor_int: 0/1 flag on this rank -> return True if ANY rank has 0
        t = torch.tensor(tensor_int, device=device, dtype=torch.int)
        if dist.is_initialized():
            dist.all_reduce(t, op=dist.ReduceOp.MIN)
        return t.item() == 0

    def any_grad_is_finite(params):
        # 本地检查梯度是否全部 finite
        for p in params:
            if p.grad is None:
                continue
            if not torch.isfinite(p.grad).all():
                return False
        return True


    device = accelerator.device
    for epoch_id in range(num_epochs):
        # import random
        for data_id, data in enumerate(tqdm(dataloader)):
            with accelerator.accumulate(model):

                # # 统一先设定一个 flag，确保不会在某些 rank 上提前退出
                # # 坏数据或异常统一由 dummy_loss 替代
                # loss = None
                # try:
                #     if data is None:
                #         # 如果没有数据，构造与模型参数同设备的 dummy loss
                #         params = [p for p in model.parameters() if p.requires_grad]
                #         if params:
                #             dummy_loss = torch.tensor(0.0, device=device)
                #             for p in params:
                #                 # 保持图的依赖，避免不同步
                #                 dummy_loss = dummy_loss + 0.0 * p.sum()
                #             loss = dummy_loss
                #         else:
                #             loss = torch.tensor(0.0, device=device, requires_grad=True)
                #     else:
                #         # 正常前向
                #         loss = model(None, data['video']) if use_cache else model(data)
                # except Exception as e:
                #     # 捕获前向/数据处理中的异常，记录并构造 dummy loss，
                #     # 但不要在部分 rank 上抛出异常导致不同步
                #     if accelerator.is_main_process:
                #         print(
                #             f"[WARN] epoch:{epoch_id} Iter:{data_id + 1}/{len(dataloader)} forward error: {e}. Using dummy loss.")
                #     params = [p for p in model.parameters() if p.requires_grad]
                #     if params:
                #         dummy_loss = torch.tensor(0.0, device=device)
                #         for p in params:
                #             dummy_loss = dummy_loss + 0.0 * p.sum()
                #         loss = dummy_loss
                #     else:
                #         loss = torch.tensor(0.0, device=device, requires_grad=True)
                #
                # # loss 可能是 tensor 或者已被替换为 tensor，确保是 scalar
                # if torch.is_tensor(loss) and loss.dim() != 0:
                #     loss = loss.mean()
                #
                # # 进一步检查数值有效性
                # if not torch.is_tensor(loss) or not torch.isfinite(loss).all():
                #     if accelerator.is_main_process:
                #         print(
                #             f"[WARN] epoch:{epoch_id} Iter:{data_id + 1}/{len(dataloader)} got NaN/Inf or non-tensor loss, replacing with dummy 0 loss.")
                #     params = [p for p in model.parameters() if p.requires_grad]
                #     if params:
                #         dummy_loss = torch.tensor(0.0, device=device)
                #         for p in params:
                #             dummy_loss = dummy_loss + 0.0 * p.sum()
                #         loss = dummy_loss
                #     else:
                #         loss = torch.tensor(0.0, device=device, requires_grad=True)
                #
                # accelerator.backward(loss)
                # accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
                # optimizer.step()
                # scheduler.step()
                # optimizer.zero_grad()
                #
                # if accelerator.sync_gradients:
                #     model_logger.on_step_end(accelerator, model, save_steps)





                # ================================ 仍会出现NaN
                # optimizer.zero_grad()
                # loss = model(None, data['video']) if use_cache else model(data)
                # accelerator.backward(loss)
                # for param in model.parameters():
                #     if param.grad is not None:
                #         torch.nan_to_num_(param.grad, nan=0.0)  # 将 nan 替换为 0
                # accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
                # optimizer.step()
                # if accelerator.sync_gradients:
                #     model_logger.on_step_end(accelerator, model, save_steps)
                # scheduler.step()





                # 1) 前向前先假定本 rank 正常
                local_forward_ok = 1
                loss = None

                try:
                    if data is None:
                        # 构造与模型参数同设备的 dummy loss (带图依赖)
                        params = [p for p in model.parameters() if p.requires_grad]
                        if params:
                            dummy_loss = torch.tensor(0.0, device=device)
                            for p in params:
                                dummy_loss = dummy_loss + 0.0 * p.sum()
                            loss = dummy_loss
                        else:
                            loss = torch.tensor(0.0, device=device, requires_grad=True)
                    else:
                        # 正常前向（根据你的模型接口）
                        loss = model(None, data['video']) if use_cache else model(data)

                except Exception as e:
                    # 捕获本 rank 的前向异常，但不要直接 raise
                    local_forward_ok = 0
                    if accelerator.is_main_process:
                        print(
                            f"[WARN] epoch:{epoch_id} Iter:{data_id + 1}/{len(dataloader)} forward error: {e}. Will use dummy loss after global sync.")
                    # 不在这里立即构造 dummy loss，先做全局 sync 判断

                # 2) 全局同步前向是否都成功：如果任一 rank 失败，则所有 rank 都应该使用 dummy loss
                global_all_forward_ok = all_ranks_ok(local_forward_ok, device)
                if not global_all_forward_ok:
                    # 构造 dummy loss（所有 rank 都走这里）
                    params = [p for p in model.parameters() if p.requires_grad]
                    if params:
                        dummy_loss = torch.tensor(0.0, device=device)
                        for p in params:
                            dummy_loss = dummy_loss + 0.0 * p.sum()
                        loss = dummy_loss
                    else:
                        loss = torch.tensor(0.0, device=device, requires_grad=True)
                    # 可在主进程打印一次
                    if accelerator.is_main_process:
                        print(
                            f"[WARN] epoch:{epoch_id} Iter:{data_id + 1}/{len(dataloader)} using globally-synced dummy loss due to forward error on some rank.")

                # 3) 确保 loss 是 scalar
                if torch.is_tensor(loss) and loss.dim() != 0:
                    loss = loss.mean()

                # 4) 检查 loss 是否为 finite，在所有 rank 上同步检查
                local_loss_finite = 1 if (torch.is_tensor(loss) and torch.isfinite(loss).all()) else 0
                global_loss_finite = all_ranks_ok(local_loss_finite, device)
                if not global_loss_finite:
                    # 任一 rank 的 loss 非 finite，统一使用 dummy loss
                    params = [p for p in model.parameters() if p.requires_grad]
                    if params:
                        dummy_loss = torch.tensor(0.0, device=device)
                        for p in params:
                            dummy_loss = dummy_loss + 0.0 * p.sum()
                        loss = dummy_loss
                    else:
                        loss = torch.tensor(0.0, device=device, requires_grad=True)
                    if accelerator.is_main_process:
                        print(
                            f"[WARN] epoch:{epoch_id} Iter:{data_id + 1}/{len(dataloader)} global loss non-finite -> using dummy loss and skipping optimizer step later.")

                # 5) backward
                accelerator.backward(loss)

                # 6) 在同步梯度前检查本地梯度是否为 finite，并在全局上汇总
                params = [p for p in model.parameters() if p.requires_grad]
                local_grads_finite = 1 if any_grad_is_finite(params) else 0
                # NOTE: 判断 any grad is finite -> we want 1 if all finite, 0 if any non-finite.
                # all_ranks_ok 实现的是 MIN 汇总，上面 any_grad_is_finite 返回 True if all local grads finite
                global_grads_all_finite = all_ranks_ok(local_grads_finite, device)

                if not global_grads_all_finite:
                    # 如果任一 rank 出现非有限的梯度，所有 rank 都跳过更新（保持参数一致）
                    if accelerator.is_main_process:
                        print(
                            f"[WARN] epoch:{epoch_id} Iter:{data_id + 1}/{len(dataloader)} found non-finite grads on some rank. Skipping optimizer.step(), zeroing grads.")
                    # 统一 zero_grad，跳过 optimizer.step() 与 scheduler.step()
                    optimizer.zero_grad()
                    # 如果使用 AMP/GradScaler，需要让 scaler 调整（accelerator 已经封装，若自己管理 scaler 请在这里处理）
                    # 强制同步并跳过后续更新
                    accelerator.wait_for_everyone()
                    continue

                # 7) 在 grads 都为 finite 的情况下，进行裁剪 / step
                accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                # 8) 可选：在同步梯度点后的回调
                if accelerator.sync_gradients:
                    model_logger.on_step_end(accelerator, model, save_steps)




                rank = accelerator.process_index
                if rank == 0:
                    print(f'epoch:{epoch_id} Iter:{data_id + 1}/{len(dataloader)} {loss.item()}')
        if save_steps is None:
            model_logger.on_epoch_end(accelerator, model, epoch_id)
    model_logger.on_training_end(accelerator, model, save_steps)


def launch_data_process_task(
    # dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    output_path="./models",
    args=None
):
    # dataloader = torch.utils.data.DataLoader(dataset, shuffle=False, collate_fn=lambda x: x[0])
    # accelerator = Accelerator()
    # model, dataloader = accelerator.prepare(model, dataloader)
    # accelerator = Accelerator(gradient_accumulation_steps=1, kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=False)], )
    accelerator = Accelerator()

    print(f"**************num_processes: {accelerator.num_processes}")
    print(f"**************process_index: {accelerator.process_index}")
    print(f"**************local_process_index: {accelerator.local_process_index}")

    dataset = VideoDataset(args=args)
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=False, collate_fn=lambda x: x[0], num_workers=1, drop_last=False)
    model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)
    # model, dataloader = accelerator.prepare(model, dataloader)
    # os.makedirs(os.path.join(output_path, "data_cache"), exist_ok=True)
    for data_id, data in enumerate(tqdm(dataloader)):
        if data is None:
            continue
        with (torch.no_grad()):
            if hasattr(model, 'forward_preprocess'):
                inputs = model.forward_preprocess(data)
                for key in model.no_model_input_keys:
                    if key in inputs:
                        del inputs[key]
            else:
                inputs = model.module.forward_preprocess(data)
                for key in model.module.no_model_input_keys:
                    if key in inputs:
                        del inputs[key]
            # inputs = {key: inputs[key] for key in model.model_input_keys if key in inputs}

            for key, value in inputs.items():
                if isinstance(value, torch.Tensor) and value.device.type == 'cuda':
                    inputs[key] = value.cpu()

            video_path = inputs['video_path']
            cache_path = f"{video_path.split('.')[0]}_{hashlib.md5(inputs['prompt'].encode()).hexdigest()}.pth"
            cache_path_abs = f"{output_path}/{cache_path}"
            os.makedirs(os.path.dirname(cache_path_abs), exist_ok=True)
            try:
                torch.save(inputs, cache_path_abs)
            except Exception as e:
                if os.path.exists(cache_path_abs):
                    os.remove(cache_path_abs)
            print(f'{data_id + 1}/{len(dataloader)}')

def wan_parser():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    # parser.add_argument("--dataset_base_path", type=str, default="/z_data/video/wan_example_video_dataset", help="Base path of the dataset.")
    # parser.add_argument("--dataset_base_path", type=str, default="/z_dataset/dataset/hyvideo", help="Base path of the dataset.")
    parser.add_argument("--dataset_base_path", type=str, default="/z_vtzhang/datasets/hyvideo/cache/cache_704_1280_81", help="Base path of the dataset.")
    # parser.add_argument("--dataset_metadata_path", type=str, default="/z_data/video/wan_example_video_dataset/metadata.csv", help="Path to the metadata file of the dataset.")
    parser.add_argument("--dataset_metadata_path", type=str, default="/z_dataset/private/vtzhang/datasets/hyvideo/csv/1080_1920_81.csv", help="Path to the metadata file of the dataset.")
    parser.add_argument("--max_pixels", type=int, default=832*480, help="Maximum number of pixels per frame, used for dynamic resolution..")
    parser.add_argument("--height", type=int, default=480, help="Height of images or videos. Leave `height` and `width` empty to enable dynamic resolution.")
    parser.add_argument("--width", type=int, default=832, help="Width of images or videos. Leave `height` and `width` empty to enable dynamic resolution.")
    parser.add_argument("--num_frames", type=int, default=81, help="Number of frames per video. Frames are sampled from the video prefix.")
    parser.add_argument("--data_file_keys", type=str, default="image,video", help="Data file keys in the metadata. Comma-separated.")
    parser.add_argument("--dataset_repeat", type=int, default=1, help="Number of times to repeat the dataset per epoch.")
    parser.add_argument("--num_steps", type=int, default=0, help="Number of times to repeat the dataset per epoch.")
    parser.add_argument("--model_paths", nargs='*', default=None, help="Paths to load models. In JSON format.")
    # parser.add_argument("--model_paths", type=str, default=None, help="Paths to load models. In JSON format.")
    parser.add_argument("--model_id_with_origin_paths", type=str, default=None, help="Model ID with origin paths, e.g., Wan-AI/Wan2.1-T2V-1.3B:diffusion_pytorch_model*.safetensors. Comma-separated.")
    # parser.add_argument("--model_id_with_origin_paths", type=str, default="Wan-AI/Wan2.2-TI2V-5B:/z_vtzhang/pretrained/Wan2.2-TI2V-5B/diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.2-TI2V-5B:/z_vtzhang/pretrained/Wan2.2-TI2V-5B/models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.2-TI2V-5B:/z_vtzhang/pretrained/Wan2.2-TI2V-5B/Wan2.2_VAE.pth", help="Model ID with origin paths, e.g., Wan-AI/Wan2.1-T2V-1.3B:diffusion_pytorch_model*.safetensors. Comma-separated.")
    # parser.add_argument("--model_id_with_origin_paths", type=str, default="Wan-AI/Wan2.2-TI2V-5B:/z_vtzhang/pretrained/Wan2.2-TI2V-5B/diffusion_pytorch_model*.safetensors", help="Model ID with origin paths, e.g., Wan-AI/Wan2.1-T2V-1.3B:diffusion_pytorch_model*.safetensors. Comma-separated.")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate.")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs.")
    parser.add_argument("--output_path", type=str, default="./models/tmp", help="Output save path.")
    parser.add_argument("--remove_prefix_in_ckpt", type=str, default="pipe.dit.", help="Remove prefix in ckpt.")
    parser.add_argument("--trainable_models", type=str, default='dit', help="Models to train, e.g., dit, vae, text_encoder.")
    parser.add_argument("--lora_base_model", type=str, default=None, help="Which model LoRA is added to.")
    parser.add_argument("--lora_target_modules", type=str, default="q,k,v,o,ffn.0,ffn.2", help="Which layers LoRA is added to.")
    parser.add_argument("--lora_rank", type=int, default=32, help="Rank of LoRA.")
    parser.add_argument("--extra_inputs", default=None, help="Additional model inputs, comma-separated.")
    # parser.add_argument("--extra_inputs", default="input_image", help="Additional model inputs, comma-separated.")
    parser.add_argument("--use_gradient_checkpointing_offload", default=False, action="store_true", help="Whether to offload gradient checkpointing to CPU memory.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps.")
    parser.add_argument("--max_timestep_boundary", type=float, default=1.0, help="Max timestep boundary (for mixed models, e.g., Wan-AI/Wan2.2-I2V-A14B).")
    parser.add_argument("--min_timestep_boundary", type=float, default=0.0, help="Min timestep boundary (for mixed models, e.g., Wan-AI/Wan2.2-I2V-A14B).")
    parser.add_argument("--find_unused_parameters", default=False, action="store_true", help="Whether to find unused parameters in DDP.")
    parser.add_argument("--save_steps", type=int, default=100, help="Number of checkpoint saving invervals. If None, checkpoints will be saved every epoch.")
    parser.add_argument("--dataset_num_workers", type=int, default=0, help="Number of workers for data loading.")
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay.")
    parser.add_argument("--task", type=str, default="train", help="Number of workers for data loading.")
    # parser.add_argument("--task", type=str, default="data_process", help="Number of workers for data loading.")
    # data_process
    parser.add_argument("--data_process_output_path", type=str, default="/z_dataset/private/vtzhang/datasets/hyvideo/cache", help="Number of workers for data loading.")
    parser.add_argument("--sovled_video_json", type=str, default=None, help="Number of workers for data loading.")
    parser.add_argument("--idx_s", type=int, default=0, help="Gradient accumulation steps.")
    parser.add_argument("--idx_e", type=int, default=-1, help="Gradient accumulation steps.")
    parser.add_argument("--model_input_keys", nargs='*', default=[
        "max_timestep_boundary",
        "min_timestep_boundary",
        "input_latents",
        "noise",
        "latents",
    ], help="Number of workers for data loading.")
    parser.add_argument("--no_model_input_keys", nargs='*', default=[
        "input_video",
        "input_image",
    ], help="no_model_input_keys")
    # parser.add_argument("--use_cache", default=True, action="store_true", help="Number of workers for data loading.")
    parser.add_argument("--use_cache", default=False, action="store_true", help="Number of workers for data loading.")
    parser.add_argument("--from_cos", default=False, action="store_true", help="")
    parser.add_argument("--cos_root", type=str, default='cos://zhongwei', help="")
    parser.add_argument("--cos_tmp_dir", type=str, default='/dev/shm/datalaoder-', help="")
    parser.add_argument("--local_rank", type=int, default=0, help="local rank passed from distributed launcher")

    return parser



def flux_parser():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--dataset_base_path", type=str, default="", required=True, help="Base path of the dataset.")
    parser.add_argument("--dataset_metadata_path", type=str, default=None, help="Path to the metadata file of the dataset.")
    parser.add_argument("--max_pixels", type=int, default=1024*1024, help="Maximum number of pixels per frame, used for dynamic resolution..")
    parser.add_argument("--height", type=int, default=None, help="Height of images. Leave `height` and `width` empty to enable dynamic resolution.")
    parser.add_argument("--width", type=int, default=None, help="Width of images. Leave `height` and `width` empty to enable dynamic resolution.")
    parser.add_argument("--data_file_keys", type=str, default="image", help="Data file keys in the metadata. Comma-separated.")
    parser.add_argument("--dataset_repeat", type=int, default=1, help="Number of times to repeat the dataset per epoch.")
    parser.add_argument("--model_paths", type=str, default=None, help="Paths to load models. In JSON format.")
    parser.add_argument("--model_id_with_origin_paths", type=str, default=None, help="Model ID with origin paths, e.g., Wan-AI/Wan2.1-T2V-1.3B:diffusion_pytorch_model*.safetensors. Comma-separated.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of epochs.")
    parser.add_argument("--output_path", type=str, default="./models", help="Output save path.")
    parser.add_argument("--remove_prefix_in_ckpt", type=str, default="pipe.dit.", help="Remove prefix in ckpt.")
    parser.add_argument("--trainable_models", type=str, default=None, help="Models to train, e.g., dit, vae, text_encoder.")
    parser.add_argument("--lora_base_model", type=str, default=None, help="Which model LoRA is added to.")
    parser.add_argument("--lora_target_modules", type=str, default="q,k,v,o,ffn.0,ffn.2", help="Which layers LoRA is added to.")
    parser.add_argument("--lora_rank", type=int, default=32, help="Rank of LoRA.")
    parser.add_argument("--extra_inputs", default=None, help="Additional model inputs, comma-separated.")
    parser.add_argument("--align_to_opensource_format", default=False, action="store_true", help="Whether to align the lora format to opensource format. Only for DiT's LoRA.")
    parser.add_argument("--use_gradient_checkpointing", default=False, action="store_true", help="Whether to use gradient checkpointing.")
    parser.add_argument("--use_gradient_checkpointing_offload", default=False, action="store_true", help="Whether to offload gradient checkpointing to CPU memory.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps.")
    parser.add_argument("--find_unused_parameters", default=False, action="store_true", help="Whether to find unused parameters in DDP.")
    parser.add_argument("--save_steps", type=int, default=None, help="Number of checkpoint saving invervals. If None, checkpoints will be saved every epoch.")
    parser.add_argument("--dataset_num_workers", type=int, default=0, help="Number of workers for data loading.")
    return parser



def qwen_image_parser():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--dataset_base_path", type=str, default="", required=True, help="Base path of the dataset.")
    parser.add_argument("--dataset_metadata_path", type=str, default=None, help="Path to the metadata file of the dataset.")
    parser.add_argument("--max_pixels", type=int, default=1024*1024, help="Maximum number of pixels per frame, used for dynamic resolution..")
    parser.add_argument("--height", type=int, default=None, help="Height of images. Leave `height` and `width` empty to enable dynamic resolution.")
    parser.add_argument("--width", type=int, default=None, help="Width of images. Leave `height` and `width` empty to enable dynamic resolution.")
    parser.add_argument("--data_file_keys", type=str, default="image", help="Data file keys in the metadata. Comma-separated.")
    parser.add_argument("--dataset_repeat", type=int, default=1, help="Number of times to repeat the dataset per epoch.")
    parser.add_argument("--model_paths", type=str, default=None, help="Paths to load models. In JSON format.")
    parser.add_argument("--model_id_with_origin_paths", type=str, default=None, help="Model ID with origin paths, e.g., Wan-AI/Wan2.1-T2V-1.3B:diffusion_pytorch_model*.safetensors. Comma-separated.")
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Paths to tokenizer.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of epochs.")
    parser.add_argument("--output_path", type=str, default="./models", help="Output save path.")
    parser.add_argument("--remove_prefix_in_ckpt", type=str, default="pipe.dit.", help="Remove prefix in ckpt.")
    parser.add_argument("--trainable_models", type=str, default=None, help="Models to train, e.g., dit, vae, text_encoder.")
    parser.add_argument("--lora_base_model", type=str, default=None, help="Which model LoRA is added to.")
    parser.add_argument("--lora_target_modules", type=str, default="q,k,v,o,ffn.0,ffn.2", help="Which layers LoRA is added to.")
    parser.add_argument("--lora_rank", type=int, default=32, help="Rank of LoRA.")
    parser.add_argument("--extra_inputs", default=None, help="Additional model inputs, comma-separated.")
    parser.add_argument("--use_gradient_checkpointing", default=False, action="store_true", help="Whether to use gradient checkpointing.")
    parser.add_argument("--use_gradient_checkpointing_offload", default=False, action="store_true", help="Whether to offload gradient checkpointing to CPU memory.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps.")
    parser.add_argument("--find_unused_parameters", default=False, action="store_true", help="Whether to find unused parameters in DDP.")
    parser.add_argument("--save_steps", type=int, default=None, help="Number of checkpoint saving invervals. If None, checkpoints will be saved every epoch.")
    parser.add_argument("--dataset_num_workers", type=int, default=0, help="Number of workers for data loading.")
    return parser
