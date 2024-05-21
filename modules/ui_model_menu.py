import importlib
import math
import re
import traceback
from functools import partial
from pathlib import Path

import gradio as gr
import psutil
import torch
from transformers import is_torch_npu_available, is_torch_xpu_available

from modules import loaders, shared, ui, utils
from modules.logging_colors import logger
from modules.LoRA import add_lora_to_model
from modules.models import load_model, unload_model
from modules.models_settings import (
    apply_model_settings_to_state,
    get_model_metadata,
    save_instruction_template,
    save_model_settings,
    update_model_parameters
)
from modules.utils import gradio


def create_ui():
    """
    创建模型界面
    """
    mu = shared.args.multi_user

    # 获取 GPU 和 CPU 内存的默认值
    total_mem = []
    if is_torch_xpu_available():
        for i in range(torch.xpu.device_count()):
            total_mem.append(math.floor(torch.xpu.get_device_properties(i).total_memory / (1024 * 1024)))
    elif is_torch_npu_available():
        for i in range(torch.npu.device_count()):
            total_mem.append(math.floor(torch.npu.get_device_properties(i).total_memory / (1024 * 1024)))
    else:
        for i in range(torch.cuda.device_count()):
            total_mem.append(math.floor(torch.cuda.get_device_properties(i).total_memory / (1024 * 1024)))

    default_gpu_mem = []
    if shared.args.gpu_memory is not None and len(shared.args.gpu_memory) > 0:
        for i in shared.args.gpu_memory:
            if 'mib' in i.lower():
                default_gpu_mem.append(int(re.sub('[a-zA-Z ]', '', i)))
            else:
                default_gpu_mem.append(int(re.sub('[a-zA-Z ]', '', i)) * 1000)

    while len(default_gpu_mem) < len(total_mem):
        default_gpu_mem.append(0)

    total_cpu_mem = math.floor(psutil.virtual_memory().total / (1024 * 1024))
    if shared.args.cpu_memory is not None:
        default_cpu_mem = re.sub('[a-zA-Z ]', '', shared.args.cpu_memory)
    else:
        default_cpu_mem = 0

    with gr.Tab("模型", elem_id="model-tab"):
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    with gr.Column():
                        with gr.Row():
                            # 模型下拉菜单
                            shared.gradio['model_menu'] = gr.Dropdown(choices=utils.get_available_models(), value=lambda: shared.model_name, label='模型', elem_classes='slim-dropdown', interactive=not mu)
                            # 刷新模型下拉菜单按钮
                            ui.create_refresh_button(shared.gradio['model_menu'], lambda: None, lambda: {'choices': utils.get_available_models()}, 'refresh-button', interactive=not mu)
                            # 加载模型按钮
                            shared.gradio['load_model'] = gr.Button("加载", visible=not shared.settings['autoload_model'], elem_classes='refresh-button', interactive=not mu)
                            # 卸载模型按钮
                            shared.gradio['unload_model'] = gr.Button("卸载", elem_classes='refresh-button', interactive=not mu)
                            # 重新加载模型按钮
                            shared.gradio['reload_model'] = gr.Button("重新加载", elem_classes='refresh-button', interactive=not mu)
                            # 保存模型设置按钮
                            shared.gradio['save_model_settings'] = gr.Button("保存设置", elem_classes='refresh-button', interactive=not mu)

                    with gr.Column():
                        with gr.Row():
                            # LoRA 下拉菜单
                            shared.gradio['lora_menu'] = gr.Dropdown(multiselect=True, choices=utils.get_available_loras(), value=shared.lora_names, label='LoRA(s)', elem_classes='slim-dropdown', interactive=not mu)
                            # 刷新 LoRA 下拉菜单按钮
                            ui.create_refresh_button(shared.gradio['lora_menu'], lambda: None, lambda: {'choices': utils.get_available_loras(), 'value': shared.lora_names}, 'refresh-button', interactive=not mu)
                            # 应用 LoRA 按钮
                            shared.gradio['lora_menu_apply'] = gr.Button(value='应用 LoRAs', elem_classes='refresh-button', interactive=not mu)

        with gr.Row():
            with gr.Column():
                # 模型加载器下拉菜单
                shared.gradio['loader'] = gr.Dropdown(label="模型加载器", choices=loaders.loaders_and_params.keys(), value=None)
                with gr.Blocks():
                    with gr.Row():
                        with gr.Column():
                            with gr.Blocks():
                                # GPU 内存滑块
                                for i in range(len(total_mem)):
                                    shared.gradio[f'gpu_memory_{i}'] = gr.Slider(label=f"GPU 内存 (MiB) 设备：{i}", maximum=total_mem[i], value=default_gpu_mem[i])

                                # CPU 内存滑块
                                shared.gradio['cpu_memory'] = gr.Slider(label="CPU 内存 (MiB)", maximum=total_cpu_mem, value=default_cpu_mem)

                            with gr.Blocks():
                                # Transformers 模型信息
                                shared.gradio['transformers_info'] = gr.Markdown('4 位加载参数:')
                                # 计算精度下拉菜单
                                shared.gradio['compute_dtype'] = gr.Dropdown(label="计算精度", choices=["bfloat16", "float16", "float32"], value=shared.args.compute_dtype)
                                # 量化类型下拉菜单
                                shared.gradio['quant_type'] = gr.Dropdown(label="量化类型", choices=["nf4", "fp4"], value=shared.args.quant_type)

                            # hqq 后端下拉菜单
                            shared.gradio['hqq_backend'] = gr.Dropdown(label="hqq 后端", choices=["PYTORCH", "PYTORCH_COMPILE", "ATEN"], value=shared.args.hqq_backend)
                            # GPU 层数滑块
                            shared.gradio['n_gpu_layers'] = gr.Slider(label="GPU 层数", minimum=0, maximum=256, value=shared.args.n_gpu_layers, info='必须设置为大于 0 才能使用你的 GPU.')
                            # 上下文长度滑块
                            shared.gradio['n_ctx'] = gr.Slider(minimum=0, maximum=shared.settings['truncation_length_max'], step=256, label="上下文长度", value=shared.args.n_ctx, info='上下文长度。如果加载模型时内存不足，尝试降低此值。')
                            # 张量分割文本框
                            shared.gradio['tensor_split'] = gr.Textbox(label='张量分割', info='在多个 GPU 上分割模型的比例列表。示例：18,17')
                            # 批次大小滑块
                            shared.gradio['n_batch'] = gr.Slider(label="批次大小", minimum=1, maximum=2048, step=1, value=shared.args.n_batch)
                            # 线程数滑块
                            shared.gradio['threads'] = gr.Slider(label="线程数", minimum=0, step=1, maximum=32, value=shared.args.threads)
                            # 批次线程数滑块
                            shared.gradio['threads_batch'] = gr.Slider(label="批次线程数", minimum=0, step=1, maximum=32, value=shared.args.threads_batch)
                            # 位宽下拉菜单
                            shared.gradio['wbits'] = gr.Dropdown(label="位宽", choices=["None", 1, 2, 3, 4, 8], value=shared.args.wbits if shared.args.wbits > 0 else "None")
                            # 组大小下拉菜单
                            shared.gradio['groupsize'] = gr.Dropdown(label="组大小", choices=["None", 32, 64, 128, 1024], value=shared.args.groupsize if shared.args.groupsize > 0 else "None")
                            # 模型类型下拉菜单
                            shared.gradio['model_type'] = gr.Dropdown(label="模型类型", choices=["None"], value=shared.args.model_type or "None")
                            # 预处理层滑块
                            shared.gradio['pre_layer'] = gr.Slider(label="预处理层", minimum=0, maximum=100, value=shared.args.pre_layer[0] if shared.args.pre_layer is not None else 0)
                            # GPU 分割文本框
                            shared.gradio['gpu_split'] = gr.Textbox(label='GPU 分割', info='每个 GPU 要使用的 VRAM (以 GB 为单位) 的逗号分隔列表。示例：20,7,7')
                            # 最大序列长度滑块
                            shared.gradio['max_seq_len'] = gr.Slider(label='最大序列长度', minimum=0, maximum=shared.settings['truncation_length_max'], step=256, info='上下文长度。如果加载模型时内存不足，尝试降低此值。', value=shared.args.max_seq_len)
                            with gr.Blocks():
                                # 位置编码 alpha 系数滑块
                                shared.gradio['alpha_value'] = gr.Slider(label='alpha 系数', minimum=1, maximum=8, step=0.05, info='NTK RoPE 缩放的位置编码 alpha 系数。推荐值 (NTKv1)：1.5 倍上下文为 1.75，2 倍上下文为 2.5。使用此值或 compress_pos_emb，不要同时使用两者。', value=shared.args.alpha_value)
                                # RoPE 频率基数滑块
                                shared.gradio['rope_freq_base'] = gr.Slider(label='RoPE 频率基数', minimum=0, maximum=1000000, step=1000, info='如果大于 0，将用于代替 alpha 系数。这两个值由 rope_freq_base = 10000 * alpha_value ^ (64 / 63) 相关联。', value=shared.args.rope_freq_base)
                                # 位置编码压缩系数滑块
                                shared.gradio['compress_pos_emb'] = gr.Slider(label='位置编码压缩系数', minimum=1, maximum=8, step=1, info='位置编码压缩系数。应设置为 (上下文长度) / (模型的原始上下文长度)。等于 1/rope_freq_scale。', value=shared.args.compress_pos_emb)

                            # AutoGPTQ 模型信息
                            shared.gradio['autogptq_info'] = gr.Markdown('ExLlamav2_HF 是针对从 Llama 派生的模型推荐的，而不是 AutoGPTQ.')
                            # QuIP# 模型信息
                            shared.gradio['quipsharp_info'] = gr.Markdown('QuIP# 目前需要手动安装。')

                        with gr.Column():
                            # 8 位加载复选框
                            shared.gradio['load_in_8bit'] = gr.Checkbox(label="8 位加载", value=shared.args.load_in_8bit)
                            # 4 位加载复选框
                            shared.gradio['load_in_4bit'] = gr.Checkbox(label="4 位加载", value=shared.args.load_in_4bit)
                            # 使用双重量化复选框
                            shared.gradio['use_double_quant'] = gr.Checkbox(label="使用双重量化", value=shared.args.use_double_quant)
                            # 使用 Flash Attention 2 复选框
                            shared.gradio['use_flash_attention_2'] = gr.Checkbox(label="使用 Flash Attention 2", value=shared.args.use_flash_attention_2, info='在加载模型时将 use_flash_attention_2 设置为 True.')
                            # Flash Attention 复选框
                            shared.gradio['flash-attn'] = gr.Checkbox(label="Flash Attention", value=shared.args.flash_attn, info='使用 Flash Attention.')
                            # 自动设备复选框
                            shared.gradio['auto_devices'] = gr.Checkbox(label="自动设备", value=shared.args.auto_devices)
                            # 启用 Tensor Cores 复选框
                            shared.gradio['tensorcores'] = gr.Checkbox(label="启用 Tensor Cores", value=shared.args.tensorcores, info='仅限 NVIDIA：使用使用 Tensor Cores 支持编译的 llama-cpp-python。这会提高 RTX 卡的性能。')
                            # 启用流式 LLM 复选框
                            shared.gradio['streaming_llm'] = gr.Checkbox(label="启用流式 LLM", value=shared.args.streaming_llm, info='(实验性) 激活 StreamingLLM 以避免在删除旧消息时重新评估整个提示。')
                            # Attention Sink 大小数字框
                            shared.gradio['attention_sink_size'] = gr.Number(label="Attention Sink 大小", value=shared.args.attention_sink_size, precision=0, info='StreamingLLM：Sink Token 的数量。仅在修剪后的提示与旧提示不共享前缀时使用。')
                            # 使用 CPU 复选框
                            shared.gradio['cpu'] = gr.Checkbox(label="使用 CPU", value=shared.args.cpu, info='llama.cpp：使用没有 GPU 加速编译的 llama-cpp-python。Transformers：在 CPU 模式下使用 PyTorch。')
                            # 行分割复选框
                            shared.gradio['row_split'] = gr.Checkbox(label="行分割", value=shared.args.row_split, info='在 GPU 上按行分割模型。这可能会提高多 GPU 性能。')
                            # 不卸载 KQV 复选框
                            shared.gradio['no_offload_kqv'] = gr.Checkbox(label="不卸载 KQV", value=shared.args.no_offload_kqv, info='不要将 K、Q、V 卸载到 GPU。这会节省 VRAM，但会降低性能。')
                            # 不使用 MulMat Q 复选框
                            shared.gradio['no_mul_mat_q'] = gr.Checkbox(label="不使用 MulMat Q", value=shared.args.no_mul_mat_q, info='禁用 mulmat 内核。')
                            # 使用 Triton 复选框
                            shared.gradio['triton'] = gr.Checkbox(label="使用 Triton", value=shared.args.triton)
                            # 不注入融合 Attention 复选框
                            shared.gradio['no_inject_fused_attention'] = gr.Checkbox(label="不注入融合 Attention", value=shared.args.no_inject_fused_attention, info='禁用融合 Attention。融合 Attention 提高推理性能，但会使用更多 VRAM。如果 VRAM 不足，请禁用。对于 AutoAWQ 融合层。如果 VRAM 不足，请禁用。')
                            # 不注入融合 MLP 复选框
                            shared.gradio['no_inject_fused_mlp'] = gr.Checkbox(label="不注入融合 MLP", value=shared.args.no_inject_fused_mlp, info='仅影响 Triton。禁用融合 MLP。融合 MLP 提高性能，但会使用更多 VRAM。如果 VRAM 不足，请禁用。')
                            # 不使用 CUDA FP16 复选框
                            shared.gradio['no_use_cuda_fp16'] = gr.Checkbox(label="不使用 CUDA FP16", value=shared.args.no_use_cuda_fp16, info='这可能会在某些系统上使模型更快。')
                            # 使用 Desc Act 复选框
                            shared.gradio['desc_act'] = gr.Checkbox(label="使用 Desc Act", value=shared.args.desc_act, info='\'desc_act\', \'wbits\' 和 \'groupsize\' 用于没有 quantize_config.json 的旧模型。')
                            # 不使用 MMAP 复选框
                            shared.gradio['no_mmap'] = gr.Checkbox(label="不使用 MMAP", value=shared.args.no_mmap)
                            # 使用 MLOCK 复选框
                            shared.gradio['mlock'] = gr.Checkbox(label="使用 MLOCK", value=shared.args.mlock)
                            # 使用 NUMA 复选框
                            shared.gradio['numa'] = gr.Checkbox(label="使用 NUMA", value=shared.args.numa, info='NUMA 支持可能有助于某些具有非均匀内存访问的系统。')
                            # 使用磁盘复选框
                            shared.gradio['disk'] = gr.Checkbox(label="使用磁盘", value=shared.args.disk)
                            # 使用 BF16 复选框
                            shared.gradio['bf16'] = gr.Checkbox(label="使用 BF16", value=shared.args.bf16)
                            # 使用 8 位缓存复选框
                            shared.gradio['cache_8bit'] = gr.Checkbox(label="使用 8 位缓存", value=shared.args.cache_8bit, info='使用 8 位缓存以节省 VRAM。')
                            # 使用 4 位缓存复选框
                            shared.gradio['cache_4bit'] = gr.Checkbox(label="使用 4 位缓存", value=shared.args.cache_4bit, info='使用 Q4 缓存以节省 VRAM。')
                            # 自动分割复选框
                            shared.gradio['autosplit'] = gr.Checkbox(label="自动分割", value=shared.args.autosplit, info='自动将模型张量分割到可用的 GPU 上。')
                            # 不使用 Flash Attention 复选框
                            shared.gradio['no_flash_attn'] = gr.Checkbox(label="不使用 Flash Attention", value=shared.args.no_flash_attn, info='强制 Flash Attention 不被使用。')
                            # 使用 CFG 缓存复选框
                            shared.gradio['cfg_cache'] = gr.Checkbox(label="使用 CFG 缓存", value=shared.args.cfg_cache, info='需要与这个加载器一起使用 CFG。')
                            # 每个 Token 的专家数量数字框
                            shared.gradio['num_experts_per_token'] = gr.Number(label="每个 Token 的专家数量", value=shared.args.num_experts_per_token, info='仅适用于 MoE 模型，例如 Mixtral。')
                            with gr.Blocks():
                                # 信任远程代码复选框
                                shared.gradio['trust_remote_code'] = gr.Checkbox(label="信任远程代码", value=shared.args.trust_remote_code, info='在加载词典/模型时将 trust_remote_code 设置为 True。要启用此选项，请使用 --trust-remote-code 标记启动 Web UI。', interactive=shared.args.trust_remote_code)
                                # 不使用快速加载复选框
                                shared.gradio['no_use_fast'] = gr.Checkbox(label="不使用快速加载", value=shared.args.no_use_fast, info='在加载词典时将 use_fast 设置为 False。')
                                # 输出所有 Logits 复选框
                                shared.gradio['logits_all'] = gr.Checkbox(label="输出所有 Logits", value=shared.args.logits_all, info='需要为困惑度评估与这个加载器一起工作而设置。否则，忽略它，因为它会使提示处理变慢。')

                            # 禁用 ExLlama 复选框
                            shared.gradio['disable_exllama'] = gr.Checkbox(label="禁用 ExLlama", value=shared.args.disable_exllama, info='禁用 GPTQ 模型的 ExLlama 内核。')
                            # 禁用 ExLlamav2 复选框
                            shared.gradio['disable_exllamav2'] = gr.Checkbox(label="禁用 ExLlamav2", value=shared.args.disable_exllamav2, info='禁用 GPTQ 模型的 ExLlamav2 内核。')
                            # GPTQ 模型加载器信息
                            shared.gradio['gptq_for_llama_info'] = gr.Markdown('用于与旧 GPU 兼容的旧加载器。在支持的情况下，ExLlamav2_HF 或 AutoGPTQ 是 GPTQ 模型的首选。')
                            # ExLlamav2 模型加载器信息
                            shared.gradio['exllamav2_info'] = gr.Markdown("ExLlamav2_HF 是推荐的，而不是 ExLlamav2，因为它与扩展的集成更好，并且跨加载器的采样行为更加一致。")
                            # llamacpp_HF 模型加载器信息
                            shared.gradio['llamacpp_HF_info'] = gr.Markdown("llamacpp_HF 将 llama.cpp 加载为 Transformers 模型。要使用它，你需要将你的 GGUF 放在 models/ 的子文件夹中，并包含必要的词典文件。\n\n你可以使用 \"llamacpp_HF 创建者\" 菜单自动执行此操作。")

            with gr.Column():
                with gr.Row():
                    # 自动加载模型复选框
                    shared.gradio['autoload_model'] = gr.Checkbox(value=shared.settings['autoload_model'], label='自动加载模型', info='是否在模型下拉菜单中选择模型后立即加载模型。', interactive=not mu)

                # 下载模型标签页
                with gr.Tab("下载"):
                    # 自定义模型菜单文本框
                    shared.gradio['custom_model_menu'] = gr.Textbox(label="下载模型或 LoRA", info="输入 Hugging Face 用户名/模型路径，例如：facebook/galactica-125m。要指定分支，在末尾加上\":\"字符，例如：facebook/galactica-125m:main。要下载单个文件，在第二个框中输入它的名称。", interactive=not mu)
                    # 下载特定文件文本框
                    shared.gradio['download_specific_file'] = gr.Textbox(placeholder="文件名 (针对 GGUF 模型)", show_label=False, max_lines=1, interactive=not mu)
                    with gr.Row():
                        # 下载模型按钮
                        shared.gradio['download_model_button'] = gr.Button("下载", variant='primary', interactive=not mu)
                        # 获取文件列表按钮
                        shared.gradio['get_file_list'] = gr.Button("获取文件列表", interactive=not mu)

                # llamacpp_HF 创建者标签页
                with gr.Tab("llamacpp_HF 创建者"):
                    with gr.Row():
                        # GGUF 下拉菜单
                        shared.gradio['gguf_menu'] = gr.Dropdown(choices=utils.get_available_ggufs(), value=lambda: shared.model_name, label='选择你的 GGUF', elem_classes='slim-dropdown', interactive=not mu)
                        # 刷新 GGUF 下拉菜单按钮
                        ui.create_refresh_button(shared.gradio['gguf_menu'], lambda: None, lambda: {'choices': utils.get_available_ggufs()}, 'refresh-button', interactive=not mu)

                    # 未量化模型 URL 文本框
                    shared.gradio['unquantized_url'] = gr.Textbox(label="输入原始 (未量化) 模型的 URL", info="示例：https://huggingface.co/lmsys/vicuna-13b-v1.5", max_lines=1)
                    # 创建 llamacpp_HF 按钮
                    shared.gradio['create_llamacpp_hf_button'] = gr.Button("提交", variant="primary", interactive=not mu)
                    # 创建 llamacpp_HF 按钮说明
                    gr.Markdown("这将把你的 gguf 文件移动到 `models` 的子文件夹中，以及必要的词典文件。")

                # 自定义指令模板标签页
                with gr.Tab("自定义指令模板"):
                    with gr.Row():
                        # 自定义模板下拉菜单
                        shared.gradio['customized_template'] = gr.Dropdown(choices=utils.get_available_instruction_templates(), value='None', label='选择所需的指令模板', elem_classes='slim-dropdown')
                        # 刷新自定义模板下拉菜单按钮
                        ui.create_refresh_button(shared.gradio['customized_template'], lambda: None, lambda: {'choices': utils.get_available_instruction_templates()}, 'refresh-button', interactive=not mu)

                    # 自定义模板提交按钮
                    shared.gradio['customized_template_submit'] = gr.Button("提交", variant="primary", interactive=not mu)
                    # 自定义模板提交按钮说明
                    gr.Markdown("这允许你为当前在 \"模型加载器\" 菜单中选择的模型设置一个自定义模板。每当模型加载时，将使用此模板来代替模型元数据中指定的模板，有时这个模板是错误的。")

                with gr.Row():
                    # 模型状态 markdown
                    shared.gradio['model_status'] = gr.Markdown('没有加载模型' if shared.model_name == 'None' else '准备就绪')


def create_event_handlers():
    """
    创建事件处理程序
    """
    # 模型加载器下拉菜单事件处理
    shared.gradio['loader'].change(
        loaders.make_loader_params_visible, gradio('loader'), gradio(loaders.get_all_params())).then(
        lambda value: gr.update(choices=loaders.get_model_types(value)), gradio('loader'), gradio('model_type'))

    # 模型下拉菜单事件处理
    shared.gradio['model_menu'].change(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        apply_model_settings_to_state, gradio('model_menu', 'interface_state'), gradio('interface_state')).then(
        ui.apply_interface_values, gradio('interface_state'), gradio(ui.list_interface_input_elements()), show_progress=False).then(
        update_model_parameters, gradio('interface_state'), None).then(
        load_model_wrapper, gradio('model_menu', 'loader', 'autoload_model'), gradio('model_status'), show_progress=False).success(
        update_truncation_length, gradio('truncation_length', 'interface_state'), gradio('truncation_length')).then(
        lambda x: x, gradio('loader'), gradio('filter_by_loader'))

    # 加载模型按钮事件处理
    shared.gradio['load_model'].click(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        update_model_parameters, gradio('interface_state'), None).then(
        partial(load_model_wrapper, autoload=True), gradio('model_menu', 'loader'), gradio('model_status'), show_progress=False).success(
        update_truncation_length, gradio('truncation_length', 'interface_state'), gradio('truncation_length')).then(
        lambda x: x, gradio('loader'), gradio('filter_by_loader'))

    # 重新加载模型按钮事件处理
    shared.gradio['reload_model'].click(
        unload_model, None, None).then(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        update_model_parameters, gradio('interface_state'), None).then(
        partial(load_model_wrapper, autoload=True), gradio('model_menu', 'loader'), gradio('model_status'), show_progress=False).success(
        update_truncation_length, gradio('truncation_length', 'interface_state'), gradio('truncation_length')).then(
        lambda x: x, gradio('loader'), gradio('filter_by_loader'))

    # 卸载模型按钮事件处理
    shared.gradio['unload_model'].click(
        unload_model, None, None).then(
        lambda: "模型已卸载", None, gradio('model_status'))

    # 保存模型设置按钮事件处理
    shared.gradio['save_model_settings'].click(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        save_model_settings, gradio('model_menu', 'interface_state'), gradio('model_status'), show_progress=False)

    # 应用 LoRA 按钮事件处理
    shared.gradio['lora_menu_apply'].click(load_lora_wrapper, gradio('lora_menu'), gradio('model_status'), show_progress=False)

    # 下载模型按钮事件处理
    shared.gradio['download_model_button'].click(download_model_wrapper, gradio('custom_model_menu', 'download_specific_file'), gradio('model_status'), show_progress=True)

    # 获取文件列表按钮事件处理
    shared.gradio['get_file_list'].click(partial(download_model_wrapper, return_links=True), gradio('custom_model_menu', 'download_specific_file'), gradio('model_status'), show_progress=True)

    # 自动加载模型复选框事件处理
    shared.gradio['autoload_model'].change(lambda x: gr.update(visible=not x), gradio('autoload_model'), gradio('load_model'))

    # 创建 llamacpp_HF 按钮事件处理
    shared.gradio['create_llamacpp_hf_button'].click(create_llamacpp_hf, gradio('gguf_menu', 'unquantized_url'), gradio('model_status'), show_progress=True)

    # 自定义模板提交按钮事件处理
    shared.gradio['customized_template_submit'].click(save_instruction_template, gradio('model_menu', 'customized_template'), gradio('model_status'), show_progress=True)


def load_model_wrapper(selected_model, loader, autoload=False):
    """
    加载模型的包装函数
    """
    if not autoload:
        yield f"已更新 `{selected_model}` 的设置。\n\n点击 \"加载\" 以加载它。"
        return

    if selected_model == 'None':
        yield "没有选择模型"
    else:
        try:
            yield f"正在加载 `{selected_model}`..."
            unload_model()
            if selected_model != '':
                shared.model, shared.tokenizer = load_model(selected_model, loader)

            if shared.model is not None:
                output = f"成功加载 `{selected_model}`。"

                settings = get_model_metadata(selected_model)
                if 'instruction_template' in settings:
                    output += '\n\n它似乎是一个具有模板 "{}". 的指令跟随模型。在聊天选项卡中，应使用指令或聊天指令模式。'.format(settings['instruction_template'])

                yield output
            else:
                yield f"加载 `{selected_model}` 失败。"
        except:
            exc = traceback.format_exc()
            logger.error('加载模型失败。')
            print(exc)
            yield exc.replace('\n', '\n\n')


def load_lora_wrapper(selected_loras):
    """
    加载 LoRA 的包装函数
    """
    yield ("正在将以下 LoRAs 应用于 {}:\n\n{}".format(shared.model_name, '\n'.join(selected_loras)))
    add_lora_to_model(selected_loras)
    yield ("成功应用了 LoRAs")


def download_model_wrapper(repo_id, specific_file, progress=gr.Progress(), return_links=False, check=False):
    """
    下载模型的包装函数
    """
    try:
        downloader = importlib.import_module("download-model").ModelDownloader()

        progress(0.0)
        model, branch = downloader.sanitize_model_and_branch_names(repo_id, None)

        yield ("正在从 Hugging Face 获取下载链接")
        links, sha256, is_lora, is_llamacpp = downloader.get_download_links_from_huggingface(model, branch, text_only=False, specific_file=specific_file)
        if return_links:
            output = "```\n"
            for link in links:
                output += f"{Path(link).name}" + "\n"

            output += "```"
            yield output
            return

        yield ("正在获取输出文件夹")
        output_folder = downloader.get_output_folder(model, branch, is_lora, is_llamacpp=is_llamacpp)

        if output_folder == Path("models"):
            output_folder = Path(shared.args.model_dir)
        elif output_folder == Path("loras"):
            output_folder = Path(shared.args.lora_dir)

        if check:
            progress(0.5)

            yield ("正在检查之前下载的文件")
            downloader.check_model_files(model, branch, links, sha256, output_folder)
            progress(1.0)
        else:
            yield (f"正在将文件{'s' if len(links) > 1 else ''} 下载到 `{output_folder}`")
            downloader.download_model_files(model, branch, links, sha256, output_folder, progress_bar=progress, threads=4, is_llamacpp=is_llamacpp)

            yield (f"模型已成功保存到 `{output_folder}/`.")
    except:
        progress(1.0)
        yield traceback.format_exc().replace('\n', '\n\n')


def create_llamacpp_hf(gguf_name, unquantized_url, progress=gr.Progress()):
    """
    创建 llamacpp_HF 模型的包装函数
    """
    try:
        downloader = importlib.import_module("download-model").ModelDownloader()

        progress(0.0)
        model, branch = downloader.sanitize_model_and_branch_names(unquantized_url, None)

        yield ("正在从 Hugging Face 获取词典文件链接")
        links, sha256, is_lora, is_llamacpp = downloader.get_download_links_from_huggingface(model, branch, text_only=True)
        output_folder = Path(shared.args.model_dir) / (re.sub(r'(?i)\.gguf$', '', gguf_name) + "-HF")

        yield (f"下载 tokenizer to `{output_folder}`")
        downloader.download_model_files(model, branch, links, sha256, output_folder, progress_bar=progress, threads=4, is_llamacpp=False)

        # Move the GGUF
        (Path(shared.args.model_dir) / gguf_name).rename(output_folder / gguf_name)

        yield (f"Model saved to `{output_folder}/`.\n\n你现在可以加载他用 llamacpp_HF.")
    except:
        progress(1.0)
        yield traceback.format_exc().replace('\n', '\n\n')


def update_truncation_length(current_length, state):
    if 'loader' in state:
        if state['loader'].lower().startswith('exllama'):
            return state['max_seq_len']
        elif state['loader'] in ['llama.cpp', 'llamacpp_HF']:
            return state['n_ctx']

    return current_length
