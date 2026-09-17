from .base_pipeline import BasePipeline
import torch


def FlowMatchSFTLoss(pipe: BasePipeline, **inputs):
    if "lora" in inputs:
        # Image-to-LoRA models need to load lora here.
        pipe.clear_lora(verbose=0)
        pipe.load_lora(pipe.dit, state_dict=inputs["lora"], hotload=True, verbose=0)

    max_timestep_boundary = int(inputs.get("max_timestep_boundary", 1) * len(pipe.scheduler.timesteps))
    min_timestep_boundary = int(inputs.get("min_timestep_boundary", 0) * len(pipe.scheduler.timesteps))

    timestep_id = torch.randint(min_timestep_boundary, max_timestep_boundary, (1,))
    timestep = pipe.scheduler.timesteps[timestep_id].to(dtype=pipe.torch_dtype, device=pipe.device)
    
    noise = torch.randn_like(inputs["input_latents"]) * inputs.get("noise_scale", 1.0)
    inputs["latents"] = pipe.scheduler.add_noise(inputs["input_latents"], noise, timestep)
    training_target = pipe.scheduler.training_target(inputs["input_latents"], noise, timestep)
    
    if "first_frame_latents" in inputs:
        inputs["latents"][:, :, 0:1] = inputs["first_frame_latents"]
    
    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    noise_pred = pipe.model_fn(**models, **inputs, timestep=timestep)
    
    if "first_frame_latents" in inputs:
        noise_pred = noise_pred[:, :, 1:]
        training_target = training_target[:, :, 1:]
    
    loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float())
    loss = loss * pipe.scheduler.training_weight(timestep)
    return loss


def FlowMatchSFTAudioVideoLoss(pipe: BasePipeline, **inputs):
    max_timestep_boundary = int(inputs.get("max_timestep_boundary", 1) * len(pipe.scheduler.timesteps))
    min_timestep_boundary = int(inputs.get("min_timestep_boundary", 0) * len(pipe.scheduler.timesteps))

    timestep_id = torch.randint(min_timestep_boundary, max_timestep_boundary, (1,))
    timestep = pipe.scheduler.timesteps[timestep_id].to(dtype=pipe.torch_dtype, device=pipe.device)
    
    # video
    noise = torch.randn_like(inputs["input_latents"])
    inputs["video_latents"] = pipe.scheduler.add_noise(inputs["input_latents"], noise, timestep)
    training_target = pipe.scheduler.training_target(inputs["input_latents"], noise, timestep)
    
    # audio
    if inputs.get("audio_input_latents") is not None:
        audio_noise = torch.randn_like(inputs["audio_input_latents"])
        inputs["audio_latents"] = pipe.scheduler.add_noise(inputs["audio_input_latents"], audio_noise, timestep)
        training_target_audio = pipe.scheduler.training_target(inputs["audio_input_latents"], audio_noise, timestep)

    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    noise_pred, noise_pred_audio = pipe.model_fn(**models, **inputs, timestep=timestep)

    loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float())
    loss = loss * pipe.scheduler.training_weight(timestep)
    if inputs.get("audio_input_latents") is not None:
        loss_audio = torch.nn.functional.mse_loss(noise_pred_audio.float(), training_target_audio.float())
        loss_audio = loss_audio * pipe.scheduler.training_weight(timestep)
        loss = loss + loss_audio
    return loss


def FlowMatchSFTMiniMaxH3AudioVideoLoss(pipe: BasePipeline, **inputs):
    max_timestep_boundary = int(inputs.get("max_timestep_boundary", 1) * len(pipe.scheduler.timesteps))
    min_timestep_boundary = int(inputs.get("min_timestep_boundary", 0) * len(pipe.scheduler.timesteps))

    timestep_id = torch.randint(min_timestep_boundary, max_timestep_boundary, (1,))
    timestep_video = pipe.scheduler.timesteps[timestep_id].to(dtype=pipe.torch_dtype, device=pipe.device)
    timestep_audio = pipe.scheduler_audio.timesteps[timestep_id].to(dtype=pipe.torch_dtype, device=pipe.device)

    noise = torch.randn_like(inputs["input_latents"])
    inputs["video_latents"] = pipe.scheduler.add_noise(inputs["input_latents"], noise, timestep_video)
    training_target = pipe.scheduler.training_target(inputs["input_latents"], noise, timestep_video)

    audio_noise = torch.randn_like(inputs["audio_input_latents"])
    inputs["audio_latents"] = pipe.scheduler_audio.add_noise(inputs["audio_input_latents"], audio_noise, timestep_audio)
    training_target_audio = pipe.scheduler_audio.training_target(inputs["audio_input_latents"], audio_noise, timestep_audio)

    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    noise_pred, noise_pred_audio = pipe.model_fn(
        **models, **inputs,
        t_video=1.0 - float(timestep_video) / pipe.scheduler.num_train_timesteps,
        t_audio=1.0 - float(timestep_audio) / pipe.scheduler_audio.num_train_timesteps,
        device=pipe.device,
    )

    loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float())
    loss = loss * pipe.scheduler.training_weight(timestep_video)
    if pipe.scheduler.sigma_weight_cap is not None and pipe.scheduler.sigma_weight_cap > 0:
        w_sigma = pipe.scheduler.sigma_weight(timestep_video, cap=pipe.scheduler.sigma_weight_cap)
        loss = loss * w_sigma
    loss_audio = torch.nn.functional.mse_loss(noise_pred_audio.float(), training_target_audio.float())
    loss_audio = loss_audio * pipe.scheduler_audio.training_weight(timestep_audio)
    if pipe.scheduler_audio.sigma_weight_cap is not None and pipe.scheduler_audio.sigma_weight_cap > 0:
        w_sigma_a = pipe.scheduler_audio.sigma_weight(timestep_audio, cap=pipe.scheduler_audio.sigma_weight_cap)
        loss_audio = loss_audio * w_sigma_a
    return loss + loss_audio


def TeacherAlignMiniMaxH3AudioVideoLoss(pipe: BasePipeline, **inputs):
    """阶段二: 主干 + proj_l 联合训练, loss = 每层 o_sla vs o_full(全注意力) 的 MSE。

    - timestep 从 50 点推理网格采 (训练分布 = 推理分布); scheduler.timesteps 已被
      MiniMaxH3TrainingModule 在 teacher_align 模式下 set 成 50 点。
    - 加噪与 SFT 一致 (video + audio 双分支), 但最终 loss 是逐层对齐和,
      不是输出级 MSE —— 让 SLA 输出的每一层都逼近全注意力 teacher。
    - 逐层反传 (2026-08-16 改造): sla_core.SparseLinearAttention.forward 在
      SLA_ALIGN_TEACHER=1 时对每层 loss_i 立即 backward (图用完即释放, 层间
      detach 解耦), 显存 O(单层) 而非 O(48层)。这里只负责加噪 + 前向,
      返回各层 loss 均值 (detach, 供日志), 梯度已在每层 backward 时累加。
    - 必须关闭 gradient checkpointing: checkpoint 重算 forward 会二次触发每层
      backward, 梯度重复累加。
    """
    timestep_id = torch.randint(0, len(pipe.scheduler.timesteps), (1,))
    timestep_video = pipe.scheduler.timesteps[timestep_id].to(dtype=pipe.torch_dtype, device=pipe.device)
    timestep_audio = pipe.scheduler_audio.timesteps[timestep_id].to(dtype=pipe.torch_dtype, device=pipe.device)

    noise = torch.randn_like(inputs["input_latents"])
    inputs["video_latents"] = pipe.scheduler.add_noise(inputs["input_latents"], noise, timestep_video)
    audio_noise = torch.randn_like(inputs["audio_input_latents"])
    inputs["audio_latents"] = pipe.scheduler_audio.add_noise(inputs["audio_input_latents"], audio_noise, timestep_audio)

    # 清零各层对齐累加器 (forward 前)
    for blk in pipe.dit.blocks:
        sla = getattr(blk.attn, "sla_module", None)
        if sla is not None:
            sla._align_loss_sum = None

    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    # 逐层反传模式下必须禁用 gradient checkpointing, 否则 checkpoint 重算会二次 backward
    inputs["use_gradient_checkpointing"] = False
    inputs["use_gradient_checkpointing_offload"] = False
    noise_pred, noise_pred_audio = pipe.model_fn(
        **models, **inputs,
        t_video=1.0 - float(timestep_video) / pipe.scheduler.num_train_timesteps,
        t_audio=1.0 - float(timestep_audio) / pipe.scheduler_audio.num_train_timesteps,
        device=pipe.device,
    )
    del noise_pred, noise_pred_audio  # 对齐 loss 不用输出级预测

    # 收集各层 loss 均值 (detach, 仅日志; 梯度已在每层 sla forward 内 backward 累加)
    losses = []
    for blk in pipe.dit.blocks:
        sla = getattr(blk.attn, "sla_module", None)
        if sla is not None and sla._align_loss_sum is not None:
            losses.append(sla._align_loss_sum)
    if not losses:
        raise RuntimeError("TeacherAlignMiniMaxH3AudioVideoLoss: 没有收集到任何层的对齐 loss (SLA 未启用?)")
    # 梯度已在每层 sla forward 内 backward 累加, 这里只需返回一个带 grad_fn 的标量
    # 让 deepspeed 断言 (value.grad_fn is not None) 通过; 0 * proj_l 参数使本处 backward 梯度为 0,
    # 不会干扰层内已累加的梯度。
    align_mean = torch.stack(losses).mean()
    zero_grad_anchor = sum(p.sum() * 0.0 for p in pipe.dit.parameters() if p.requires_grad)
    return align_mean + zero_grad_anchor


def DirectDistillLoss(pipe: BasePipeline, **inputs):
    pipe.scheduler.set_timesteps(inputs["num_inference_steps"])
    pipe.scheduler.training = True
    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    for progress_id, timestep in enumerate(pipe.scheduler.timesteps):
        timestep = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)
        noise_pred = pipe.model_fn(**models, **inputs, timestep=timestep, progress_id=progress_id)
        inputs["latents"] = pipe.step(pipe.scheduler, progress_id=progress_id, noise_pred=noise_pred, **inputs)
    loss = torch.nn.functional.mse_loss(inputs["latents"].float(), inputs["input_latents"].float())
    return loss


class TrajectoryImitationLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.initialized = False
    
    def initialize(self, device):
        import lpips # TODO: remove it
        self.loss_fn = lpips.LPIPS(net='alex').to(device)
        self.initialized = True

    def fetch_trajectory(self, pipe: BasePipeline, timesteps_student, inputs_shared, inputs_posi, inputs_nega, num_inference_steps, cfg_scale):
        trajectory = [inputs_shared["latents"].clone()]

        pipe.scheduler.set_timesteps(num_inference_steps, target_timesteps=timesteps_student)
        models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
        for progress_id, timestep in enumerate(pipe.scheduler.timesteps):
            timestep = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)
            noise_pred = pipe.cfg_guided_model_fn(
                pipe.model_fn, cfg_scale,
                inputs_shared, inputs_posi, inputs_nega,
                **models, timestep=timestep, progress_id=progress_id
            )
            inputs_shared["latents"] = pipe.step(pipe.scheduler, progress_id=progress_id, noise_pred=noise_pred.detach(), **inputs_shared)

            trajectory.append(inputs_shared["latents"].clone())
        return pipe.scheduler.timesteps, trajectory
    
    def align_trajectory(self, pipe: BasePipeline, timesteps_teacher, trajectory_teacher, inputs_shared, inputs_posi, inputs_nega, num_inference_steps, cfg_scale):
        loss = 0
        pipe.scheduler.set_timesteps(num_inference_steps, training=True)
        models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
        for progress_id, timestep in enumerate(pipe.scheduler.timesteps):
            timestep = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)

            progress_id_teacher = torch.argmin((timesteps_teacher - timestep).abs())
            inputs_shared["latents"] = trajectory_teacher[progress_id_teacher]

            noise_pred = pipe.cfg_guided_model_fn(
                pipe.model_fn, cfg_scale,
                inputs_shared, inputs_posi, inputs_nega,
                **models, timestep=timestep, progress_id=progress_id
            )

            sigma = pipe.scheduler.sigmas[progress_id]
            sigma_ = 0 if progress_id + 1 >= len(pipe.scheduler.timesteps) else pipe.scheduler.sigmas[progress_id + 1]
            if progress_id + 1 >= len(pipe.scheduler.timesteps):
                latents_ = trajectory_teacher[-1]
            else:
                progress_id_teacher = torch.argmin((timesteps_teacher - pipe.scheduler.timesteps[progress_id + 1]).abs())
                latents_ = trajectory_teacher[progress_id_teacher]
            
            denom = sigma_ - sigma
            denom = torch.sign(denom) * torch.clamp(denom.abs(), min=1e-6)
            target = (latents_ - inputs_shared["latents"]) / denom
            loss = loss + torch.nn.functional.mse_loss(noise_pred.float(), target.float()) * pipe.scheduler.training_weight(timestep)
        return loss
    
    def compute_regularization(self, pipe: BasePipeline, trajectory_teacher, inputs_shared, inputs_posi, inputs_nega, num_inference_steps, cfg_scale):
        inputs_shared["latents"] = trajectory_teacher[0]
        pipe.scheduler.set_timesteps(num_inference_steps)
        models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
        for progress_id, timestep in enumerate(pipe.scheduler.timesteps):
            timestep = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)
            noise_pred = pipe.cfg_guided_model_fn(
                pipe.model_fn, cfg_scale,
                inputs_shared, inputs_posi, inputs_nega,
                **models, timestep=timestep, progress_id=progress_id
            )
            inputs_shared["latents"] = pipe.step(pipe.scheduler, progress_id=progress_id, noise_pred=noise_pred.detach(), **inputs_shared)

        image_pred = pipe.vae_decoder(inputs_shared["latents"])
        image_real = pipe.vae_decoder(trajectory_teacher[-1])
        loss = self.loss_fn(image_pred.float(), image_real.float())
        return loss

    def forward(self, pipe: BasePipeline, inputs_shared, inputs_posi, inputs_nega):
        if not self.initialized:
            self.initialize(pipe.device)
        with torch.no_grad():
            pipe.scheduler.set_timesteps(8)
            timesteps_teacher, trajectory_teacher = self.fetch_trajectory(inputs_shared["teacher"], pipe.scheduler.timesteps, inputs_shared, inputs_posi, inputs_nega, 50, 2)
            timesteps_teacher = timesteps_teacher.to(dtype=pipe.torch_dtype, device=pipe.device)
        loss_1 = self.align_trajectory(pipe, timesteps_teacher, trajectory_teacher, inputs_shared, inputs_posi, inputs_nega, 8, 1)
        loss_2 = self.compute_regularization(pipe, trajectory_teacher, inputs_shared, inputs_posi, inputs_nega, 8, 1)
        loss = loss_1 + loss_2
        return loss
