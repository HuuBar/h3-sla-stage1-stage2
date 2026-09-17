import torch, os, argparse, accelerate, warnings
from diffsynth.core import UnifiedDataset
from diffsynth.core.data.operators import LoadAudioWithTorchaudio, ToAbsolutePath
from diffsynth.utils.data.minimax_h3 import MiniMaxH3ReferenceLoader
from diffsynth.pipelines.minimax_h3_audio_video import MiniMaxH3Pipeline, ModelConfig
from diffsynth.diffusion import *
os.environ["TOKENIZERS_PARALLELISM"] = "false"

MINIMAX_H3_FRAME_RATE = 24
MINIMAX_H3_TIME_DIVISION_FACTOR = 17
MINIMAX_H3_TIME_DIVISION_REMAINDER = 5


class MiniMaxH3TrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        model_paths=None, model_id_with_origin_paths=None,
        processor_path=None,
        trainable_models=None,
        lora_base_model=None, lora_target_modules="", lora_rank=32, lora_checkpoint=None,
        preset_lora_path=None, preset_lora_model=None,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        fp8_models=None,
        offload_models=None,
        resume_from_checkpoint=None, remove_prefix_in_ckpt=None,
        device="cpu",
        task="sft",
        use_sla=False, sla_topk=0.05, sla_feature_map="softmax",
        sla_blkq=64, sla_blkk=64, sla_per_layer=None,
        max_timestep_boundary=1.0, min_timestep_boundary=0.0,
        mix_low_noise_ratio=0.0,
        teacher_align=False, align_timesteps=50, init_proj=None,
        timestep_grid=None, sigma_weight_cap=None,
        proj_only=False, proj_lr=None,
    ):
        super().__init__()
        # Warning
        if not use_gradient_checkpointing:
            warnings.warn("Gradient checkpointing is detected as disabled. To prevent out-of-memory errors, the training framework will forcibly enable gradient checkpointing.")
            use_gradient_checkpointing = True

        # Load models
        model_configs = self.parse_model_configs(model_paths, model_id_with_origin_paths, fp8_models=fp8_models, offload_models=offload_models, device=device)
        if use_sla:
            for model_config in model_configs:
                hit = False
                if model_config.model_id is not None and model_config.model_id == "MiniMax/MiniMax-H3":
                    hit = model_config.origin_file_pattern is not None and "transformer" in str(model_config.origin_file_pattern)
                elif model_config.path is not None:
                    hit = "transformer" in str(model_config.path)
                if hit:
                    model_config.extra_kwargs = dict(model_config.extra_kwargs or {})
                    model_config.extra_kwargs.update(
                        use_sla=True, sla_topk=sla_topk, sla_feature_map=sla_feature_map,
                        sla_blkq=sla_blkq, sla_blkk=sla_blkk,
                    )
                    if sla_per_layer:
                        import json as _json
                        model_config.extra_kwargs["sla_per_layer"] = _json.loads(sla_per_layer)
        pipe_kwargs = {}
        if processor_path is not None:
            processor_config = self.parse_path_or_model_id(processor_path)
            pipe_kwargs["processor_config"] = processor_config
        self.pipe = MiniMaxH3Pipeline.from_pretrained(torch_dtype=torch.bfloat16, device=device, model_configs=model_configs, **pipe_kwargs)
        self.pipe = self.split_pipeline_units(
            task, self.pipe, trainable_models, lora_base_model,
            remove_unnecessary_params=True,
            force_remove_params_shared=("video_latents", "audio_latents"),
            force_remove_params_nega=("prompt_embeds", "text_token_tags", "packed"),
        )
        self.resume_from_checkpoint(resume_from_checkpoint, remove_prefix_in_ckpt)
        # Training mode
        self.switch_pipe_to_training_mode(
            self.pipe, trainable_models,
            lora_base_model, lora_target_modules, lora_rank, lora_checkpoint,
            preset_lora_path, preset_lora_model,
            task=task,
        )
        self.pipe.scheduler.set_timesteps(1000, training=True, shift=12.0)
        self.pipe.scheduler_audio.set_timesteps(1000, training=True, shift=3.0)

        # 阶段一 (proj_l-only): 冻结主干, 只留 sla_module.proj_l 可训练。
        # sla_core 读 SLA_PROJ_ONLY 后 detach o_s (主干零漂移, sparse backward 不执行)。
        if proj_only:
            os.environ["SLA_PROJ_ONLY"] = "1"
            n_proj = 0
            for name, param in self.pipe.dit.named_parameters():
                keep = "sla_module.proj_l" in name
                param.requires_grad_(keep)
                n_proj += int(keep)
            print(f"[stage1] proj_l-only: 冻结主干, 仅 {n_proj} 个 proj_l 参数可训练 "
                  f"({sum(p.numel() for p in self.pipe.dit.parameters() if p.requires_grad)} params)")
        self.proj_lr = proj_lr

        # 训练分布 = 推理分布: --timestep-grid N 把训练采样网格从全区间 0-999
        # 换成 N 点推理网格 (与 --teacher-align 的 50 点同构, 但独立开关)。
        # 独立于 teacher 对齐: 老办法 SFT 也可用。
        if timestep_grid is not None and timestep_grid > 0:
            self.pipe.scheduler.set_timesteps(timestep_grid, training=True, shift=12.0)
            self.pipe.scheduler_audio.set_timesteps(timestep_grid, training=True, shift=3.0)
            print(f"[train] timestep_grid: {timestep_grid} 点推理网格 (video shift=12, audio shift=3), "
                  f"sigma[0]={self.pipe.scheduler.sigmas[0]:.3f} sigma[-1]={self.pipe.scheduler.sigmas[-1]:.4f}")

        # 阶段二 teacher 对齐: 训练分布 = 推理分布 (50 点推理网格), SLA_ALIGN_TEACHER
        # 让 sla_core 每层累加 o_sla vs o_full 的 MSE; task 用 "sft:train_align"。
        self.teacher_align = teacher_align
        if teacher_align:
            os.environ["SLA_ALIGN_TEACHER"] = "1"
            if not (timestep_grid is not None and timestep_grid > 0):
                self.pipe.scheduler.set_timesteps(align_timesteps, training=True, shift=12.0)
                self.pipe.scheduler_audio.set_timesteps(align_timesteps, training=True, shift=3.0)
            print(f"[stage2] teacher_align: {align_timesteps} 点推理网格 (video shift=12, audio shift=3), "
                  f"sigma[0]={self.pipe.scheduler.sigmas[0]:.3f} sigma[-1]={self.pipe.scheduler.sigmas[-1]:.4f}")

        # 阶段二 proj_l 初始权重: 已通过 model_paths 注入 (transformer shards + proj_l_step250.safetensors
        # 合并 hash 922fd0b8 匹配 SLA 产物配置, zero3 安全)。zero3 分片下不能用 load_state_dict
        # 手动灌全量权重 (每 rank 只有分片, size mismatch)。

        # sigma 加权 (min-SNR 风格): >0 时 FlowMatchSFT loss 乘 weight=min(1/sigma^2, cap)
        if sigma_weight_cap is not None and sigma_weight_cap > 0:
            self.pipe.scheduler.sigma_weight_cap = sigma_weight_cap
            self.pipe.scheduler_audio.sigma_weight_cap = sigma_weight_cap
            print(f"[train] sigma_weight: cap={sigma_weight_cap} (min-SNR 风格, 低噪端权重大)")

        # Store other configs
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        self.fp8_models = fp8_models
        self.task = task
        self.max_timestep_boundary = max_timestep_boundary
        self.min_timestep_boundary = min_timestep_boundary
        self.mix_low_noise_ratio = mix_low_noise_ratio
        self.task_to_loss = {
            "sft:data_process": lambda pipe, *args: args,
            "sft": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTMiniMaxH3AudioVideoLoss(pipe, **inputs_shared, **inputs_posi),
            "sft:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTMiniMaxH3AudioVideoLoss(pipe, **inputs_shared, **inputs_posi),
        }
        # 阶段二 teacher 对齐 loss (注册必须在 task_to_loss 定义之后)
        if self.teacher_align:
            self.task_to_loss["sft:train_align"] = lambda pipe, inputs_shared, inputs_posi, inputs_nega: TeacherAlignMiniMaxH3AudioVideoLoss(pipe, **inputs_shared, **inputs_posi)

    def get_sla_sparsity(self):
        """训练中监控: 返回 DiT 的实际 SLA 稀疏度 (未启用返回 None)."""
        if getattr(self, "pipe", None) is None or getattr(self.pipe, "dit", None) is None:
            return None
        return self.pipe.dit.get_sparsity()

    def parse_extra_inputs(self, data, extra_inputs, inputs_shared):
        # First/last-frame conditioning is derived from the training video itself, following the
        # `input_image` / `end_image` convention used by the wanvideo series. The H3 pipeline takes
        # them as a `keyframes` list plus `keyframe_indices` in {0, -1}.
        keyframes, keyframe_indices = [], []
        for extra_input in extra_inputs:
            if extra_input == "input_image":
                keyframes.append(data["video"][0])
                keyframe_indices.append(0)
            elif extra_input == "end_image":
                keyframes.append(data["video"][-1])
                keyframe_indices.append(-1)
            else:
                inputs_shared[extra_input] = data[extra_input]
        if keyframes:
            inputs_shared["keyframes"] = keyframes
            inputs_shared["keyframe_indices"] = keyframe_indices
        return inputs_shared

    def get_pipeline_inputs(self, data):
        inputs_posi = {"prompt": data["prompt"]}
        inputs_nega = {}
        inputs_shared = {
            # Assume you are using this pipeline for inference,
            # please fill in the input parameters.
            "input_video": data["video"],
            "height": data["video"][0].size[1],
            "width": data["video"][0].size[0],
            "num_frames": len(data["video"]),
            "keyframes": None,
            "keyframe_indices": None,
            "references": None,
            "ref_image_short_edge": 2048,
            "ref_video_short_edge": 768, "ref_video_max_pixels": 768 * 1344,
            "imgvid_cond_noise_aug": self.pipe.imgvid_cond_noise_aug,
            "audio_cond_noise_aug": self.pipe.audio_cond_noise_aug,
            # Please do not modify the following parameters
            # unless you clearly know what this will cause.
            "cfg_scale": 1,
            "seed": 42,
            "rand_device": "cpu",
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            # 高/低噪两阶段: timestep 采样区间 (loss.py 原生支持, 默认 1/0 = 全区间)
            "max_timestep_boundary": self.max_timestep_boundary,
            "min_timestep_boundary": self.min_timestep_boundary,
        }
        inputs_shared = self.parse_extra_inputs(data, self.extra_inputs, inputs_shared)
        return inputs_shared, inputs_posi, inputs_nega

    def forward(self, data, inputs=None):
        if inputs is None: inputs = self.get_pipeline_inputs(data)
        inputs = self.transfer_data_to_device(inputs, self.pipe.device, self.pipe.torch_dtype)
        # timestep 采样区间: 默认全范围 (1/0); 低噪段强化时以 CLI boundary 为准;
        # mix_low_noise_ratio>0 时按概率混合: 该比例走低噪段, 其余走全范围。
        if self.mix_low_noise_ratio > 0 and torch.rand(1).item() < self.mix_low_noise_ratio:
            inputs[0]["max_timestep_boundary"] = self.max_timestep_boundary
            inputs[0]["min_timestep_boundary"] = self.min_timestep_boundary
        else:
            inputs[0]["max_timestep_boundary"] = 1.0
            inputs[0]["min_timestep_boundary"] = 0.0
        for unit in self.pipe.units:
            inputs = self.pipe.unit_runner(unit, self.pipe, *inputs)
        loss = self.task_to_loss[self.task](self.pipe, *inputs)
        return loss


def minimax_h3_parser():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser = add_general_config(parser)
    parser = add_video_size_config(parser)
    parser.add_argument("--processor_path", type=str, default=None, help="Path or `model_id:pattern` of the Qwen3-VL processor.")
    parser.add_argument("--initialize_model_on_cpu", default=False, action="store_true", help="Whether to initialize models on CPU.")
    parser.add_argument("--use_sla", default=False, action="store_true", help="Enable Sparse-Linear Attention (SLA) on the main DiT blocks.")
    parser.add_argument("--sla_topk", type=float, default=0.05, help="SLA block-level top-k ratio (default 0.05 = ~95% sparse).")
    parser.add_argument("--sla_feature_map", type=str, default="softmax", choices=["softmax", "elu", "relu"], help="SLA linear-attention feature map.")
    parser.add_argument("--sla_blkq", type=int, default=64, help="SLA query block size.")
    parser.add_argument("--sla_blkk", type=int, default=64, help="SLA key block size.")
    parser.add_argument("--sla_per_layer", type=str, default=None, help="逐层稀疏覆盖 JSON, 如 {\"0\":{\"dense\":true},\"48\":{\"topk\":0.1},\"49\":{\"topk\":0.1}}")
    parser.add_argument("--max-timestep-boundary", type=float, default=1.0, help="timestep 采样上界 (比例 0~1, 默认 1.0=全区间)")
    parser.add_argument("--min-timestep-boundary", type=float, default=0.0, help="timestep 采样下界 (比例 0~1, 默认 0.0=全区间)")
    parser.add_argument("--mix-low-noise-ratio", type=float, default=0.0, help=">0 时按该概率走低噪段 boundary, 其余走全范围 (混合采样, 默认 0=纯 boundary)")
    parser.add_argument("--teacher-align", default=False, action="store_true", help="阶段二: 主干+proj_l 联合, loss=每层 o_sla vs o_full 对齐 (50 点推理网格采样)")
    parser.add_argument("--align-timesteps", type=int, default=50, help="teacher-align 网格点数 (默认 50 = 推理网格)")
    parser.add_argument("--timestep-grid", type=int, default=None, help="训练 timestep 从 N 点推理网格采 (独立于 teacher-align; 0/None=全区间 0-999)")
    parser.add_argument("--sigma-weight-cap", type=float, default=None, help=">0 时 loss 按 sigma 加权 (min-SNR 风格, weight=min(1/sigma^2, cap)), 低噪端权重大")
    parser.add_argument("--proj-only", default=False, action="store_true", help="阶段一: 冻结主干, 只训 sla_module.proj_l (SLA_PROJ_ONLY=1, 主干零漂移)")
    parser.add_argument("--proj-lr", type=float, default=None, help="proj_l 单独学习率 (阶段二双 lr: 主干 --learning-rate, proj_l 用此值)")
    parser.add_argument("--init-proj", type=str, default=None, help="proj_l 初始权重 (阶段一产物 proj_l_stepN.pt)")
    return parser


if __name__ == "__main__":
    parser = minimax_h3_parser()
    args = parser.parse_args()
    if args.num_frames % MINIMAX_H3_TIME_DIVISION_FACTOR != MINIMAX_H3_TIME_DIVISION_REMAINDER:
        raise ValueError(
            f"--num_frames must be {MINIMAX_H3_TIME_DIVISION_FACTOR}n+{MINIMAX_H3_TIME_DIVISION_REMAINDER} "
            f"(e.g. 39, 56, 124) so it lands on the video VAE's temporal grouping, got {args.num_frames}."
        )
    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[accelerate.DistributedDataParallelKwargs(find_unused_parameters=args.find_unused_parameters)],
    )
    video_processor = UnifiedDataset.default_video_operator(
        base_path=args.dataset_base_path,
        max_pixels=args.max_pixels,
        height=args.height,
        width=args.width,
        height_division_factor=32,
        width_division_factor=32,
        num_frames=args.num_frames,
        time_division_factor=MINIMAX_H3_TIME_DIVISION_FACTOR,
        time_division_remainder=MINIMAX_H3_TIME_DIVISION_REMAINDER,
        frame_rate=MINIMAX_H3_FRAME_RATE,
        fix_frame_rate=True,
    )
    dataset = UnifiedDataset(
        base_path=args.dataset_base_path,
        metadata_path=args.dataset_metadata_path,
        repeat=args.dataset_repeat,
        data_file_keys=args.data_file_keys.split(","),
        main_data_operator=video_processor,
        special_operator_map={
            "input_audio": ToAbsolutePath(args.dataset_base_path) >> LoadAudioWithTorchaudio(
                num_frames=args.num_frames,
                time_division_factor=MINIMAX_H3_TIME_DIVISION_FACTOR,
                time_division_remainder=MINIMAX_H3_TIME_DIVISION_REMAINDER,
                frame_rate=MINIMAX_H3_FRAME_RATE,
                fix_frame_rate=True,
            ),
            "references": MiniMaxH3ReferenceLoader(
                base_path=args.dataset_base_path,
                height=args.height,
                width=args.width,
                max_pixels=args.max_pixels,
                num_frames=args.num_frames,
                frame_rate=MINIMAX_H3_FRAME_RATE,
            ),
        }
    )
    model = MiniMaxH3TrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        processor_path=args.processor_path,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        preset_lora_path=args.preset_lora_path,
        preset_lora_model=args.preset_lora_model,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        fp8_models=args.fp8_models,
        offload_models=args.offload_models,
        resume_from_checkpoint=args.resume_from_checkpoint,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
        task=args.task,
        device="cpu" if (args.initialize_model_on_cpu or args.enable_model_cpu_offload) else accelerator.device,
        use_sla=args.use_sla,
        sla_topk=args.sla_topk,
        sla_feature_map=args.sla_feature_map,
        sla_blkq=args.sla_blkq,
        sla_blkk=args.sla_blkk,
        sla_per_layer=args.sla_per_layer,
        max_timestep_boundary=args.max_timestep_boundary,
        min_timestep_boundary=args.min_timestep_boundary,
        mix_low_noise_ratio=args.mix_low_noise_ratio,
        teacher_align=args.teacher_align,
        align_timesteps=args.align_timesteps,
        timestep_grid=args.timestep_grid,
        sigma_weight_cap=args.sigma_weight_cap,
        init_proj=args.init_proj,
        proj_only=args.proj_only,
        proj_lr=args.proj_lr,
    )
    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
        enable_tensorboard_log=args.enable_tensorboard_log,
        enable_swanlab_log=args.enable_swanlab_log,
        swanlab_project=args.swanlab_project,
        enable_wandb_log=args.enable_wandb_log,
        wandb_project=args.wandb_project,
    )
    launcher_map = {
        "sft:data_process": launch_data_process_task,
        "sft": launch_training_task,
        "sft:train": launch_training_task,
        "sft:train_align": launch_training_task,
    }
    launcher_map[args.task](accelerator, dataset, model, model_logger, args=args)
