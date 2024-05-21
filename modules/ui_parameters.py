from pathlib import Path

import gradio as gr

from modules import loaders, presets, shared, ui, ui_chat, utils
from modules.utils import gradio


def create_ui(default_preset):
    mu = shared.args.multi_user
    generate_params = presets.load_preset(default_preset)
    with gr.Tab("Parameters", elem_id="parameters"):
        with gr.Tab("Generation"):
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        shared.gradio['preset_menu'] = gr.Dropdown(choices=utils.get_available_presets(), value=default_preset, label='Preset', elem_classes='slim-dropdown')
                        ui.create_refresh_button(shared.gradio['preset_menu'], lambda: None, lambda: {'choices': utils.get_available_presets()}, 'refresh-button', interactive=not mu)
                        shared.gradio['save_preset'] = gr.Button('💾', elem_classes='refresh-button', interactive=not mu)
                        shared.gradio['delete_preset'] = gr.Button('🗑️', elem_classes='refresh-button', interactive=not mu)
                        shared.gradio['random_preset'] = gr.Button('🎲', elem_classes='refresh-button')

                with gr.Column():
                    shared.gradio['filter_by_loader'] = gr.Dropdown(label="Filter by loader", choices=["All"] + list(loaders.loaders_and_params.keys()), value="All", elem_classes='slim-dropdown')

            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        with gr.Column():
                            shared.gradio['max_new_tokens'] = gr.Slider(minimum=shared.settings['max_new_tokens_min'], maximum=shared.settings['max_new_tokens_max'], step=1, label='max_new_tokens', value=shared.settings['max_new_tokens'])
                            shared.gradio['temperature'] = gr.Slider(0.01, 5, value=generate_params['temperature'], step=0.01, label='temperature')
                            shared.gradio['top_p'] = gr.Slider(0.0, 1.0, value=generate_params['top_p'], step=0.01, label='top_p')
                            shared.gradio['top_k'] = gr.Slider(0, 200, value=generate_params['top_k'], step=1, label='top_k')
                            shared.gradio['typical_p'] = gr.Slider(0.0, 1.0, value=generate_params['typical_p'], step=0.01, label='typical_p')
                            shared.gradio['min_p'] = gr.Slider(0.0, 1.0, value=generate_params['min_p'], step=0.01, label='min_p')
                            shared.gradio['repetition_penalty'] = gr.Slider(1.0, 1.5, value=generate_params['repetition_penalty'], step=0.01, label='repetition_penalty')
                            shared.gradio['frequency_penalty'] = gr.Slider(0, 2, value=generate_params['frequency_penalty'], step=0.05, label='frequency_penalty')
                            shared.gradio['presence_penalty'] = gr.Slider(0, 2, value=generate_params['presence_penalty'], step=0.05, label='presence_penalty')
                            shared.gradio['repetition_penalty_range'] = gr.Slider(0, 4096, step=64, value=generate_params['repetition_penalty_range'], label='repetition_penalty_range')
                            shared.gradio['do_sample'] = gr.Checkbox(value=generate_params['do_sample'], label='do_sample')
                            gr.Markdown("[Learn more](https://github.com/A-X-MY/webui-gpt/wiki/03-%E2%80%90-Parameters-Tab)")

                        with gr.Column():
                            with gr.Group():
                                shared.gradio['auto_max_new_tokens'] = gr.Checkbox(value=shared.settings['auto_max_new_tokens'], label='auto_max_new_tokens', info='将 max_new_tokens 扩展到可用的上下文长度。')
                                shared.gradio['ban_eos_token'] = gr.Checkbox(value=shared.settings['ban_eos_token'], label='禁止 eos_token', info='强制模型永远不会过早结束生成。')
                                shared.gradio['add_bos_token'] = gr.Checkbox(value=shared.settings['add_bos_token'], label='在提示词开头添加 bos_token', info='禁用此选项可以使回复更具创造力。')
                                shared.gradio['custom_stopping_strings'] = gr.Textbox(lines=2, value=shared.settings["自定义停止字符串"] or None, label='用 "" 包裹并用逗号分隔。', placeholder='"\\n", "\\nYou:"')
                                shared.gradio['custom_token_bans'] = gr.Textbox(value=shared.settings['custom_token_bans'] or None, label='token禁止', info='要禁止的token ID，用逗号分隔。可以在“默认”或“笔记本”选项卡中找到 ID。')

                            shared.gradio['penalty_alpha'] = gr.Slider(0, 5, value=generate_params['penalty_alpha'], label='penalty_alpha', info='用于对比搜索。do_sample 必须取消选中。')
                            shared.gradio['guidance_scale'] = gr.Slider(-0.5, 2.5, step=0.05, value=generate_params['guidance_scale'], label='guidance_scale', info='用于 CFG。1.5 是一个不错的值。')
                            shared.gradio['negative_prompt'] = gr.Textbox(value=shared.settings['negative_prompt'], label='负面提示词', lines=3, elem_classes=['add_scrollbar'])
                            shared.gradio['mirostat_mode'] = gr.Slider(0, 2, step=1, value=generate_params['mirostat_mode'], label='mirostat_mode', info='mode=1 仅适用于 llama.cpp。')
                            shared.gradio['mirostat_tau'] = gr.Slider(0, 10, step=0.01, value=generate_params['mirostat_tau'], label='mirostat_tau')
                            shared.gradio['mirostat_eta'] = gr.Slider(0, 1, step=0.01, value=generate_params['mirostat_eta'], label='mirostat_eta')
                            shared.gradio['epsilon_cutoff'] = gr.Slider(0, 9, value=generate_params['epsilon_cutoff'], step=0.01, label='epsilon_cutoff')
                            shared.gradio['eta_cutoff'] = gr.Slider(0, 20, value=generate_params['eta_cutoff'], step=0.01, label='eta_cutoff')
                            shared.gradio['encoder_repetition_penalty'] = gr.Slider(0.8, 1.5, value=generate_params['encoder_repetition_penalty'], step=0.01, label='encoder_repetition_penalty')
                            shared.gradio['no_repeat_ngram_size'] = gr.Slider(0, 20, step=1, value=generate_params['no_repeat_ngram_size'], label='no_repeat_ngram_size')

                with gr.Column():
                    with gr.Row() as shared.gradio['grammar_file_row']:
                        shared.gradio['grammar_file'] = gr.Dropdown(value='None', choices=utils.get_available_grammars(), label='从文件加载语法 (.gbnf)', elem_classes='slim-dropdown')
                        ui.create_refresh_button(shared.gradio['grammar_file'], lambda: None, lambda: {'choices': utils.get_available_grammars()}, 'refresh-button', interactive=not mu)
                        shared.gradio['save_grammar'] = gr.Button('💾', elem_classes='refresh-button', interactive=not mu)
                        shared.gradio['delete_grammar'] = gr.Button('🗑️ ', elem_classes='refresh-button', interactive=not mu)

                    shared.gradio['grammar_string'] = gr.Textbox(value='', label='Grammar', lines=16, elem_classes=['add_scrollbar', 'monospace'])

                    with gr.Row():
                        with gr.Column():
                            shared.gradio['tfs'] = gr.Slider(0.0, 1.0, value=generate_params['tfs'], step=0.01, label='tfs')
                            shared.gradio['top_a'] = gr.Slider(0.0, 1.0, value=generate_params['top_a'], step=0.01, label='top_a')
                            shared.gradio['smoothing_factor'] = gr.Slider(0.0, 10.0, value=generate_params['smoothing_factor'], step=0.01, label='smoothing_factor', info='激活二次采样。')
                            shared.gradio['smoothing_curve'] = gr.Slider(1.0, 10.0, value=generate_params['smoothing_curve'], step=0.01, label='smoothing_curve', info='调整二次采样的衰减曲线。')
                            shared.gradio['dynamic_temperature'] = gr.Checkbox(value=generate_params['dynamic_temperature'], label='dynamic_temperature')
                            shared.gradio['dynatemp_low'] = gr.Slider(0.01, 5, value=generate_params['dynatemp_low'], step=0.01, label='dynatemp_low', visible=generate_params['dynamic_temperature'])
                            shared.gradio['dynatemp_high'] = gr.Slider(0.01, 5, value=generate_params['dynatemp_high'], step=0.01, label='dynatemp_high', visible=generate_params['dynamic_temperature'])
                            shared.gradio['dynatemp_exponent'] = gr.Slider(0.01, 5, value=generate_params['dynatemp_exponent'], step=0.01, label='dynatemp_exponent', visible=generate_params['dynamic_temperature'])
                            shared.gradio['temperature_last'] = gr.Checkbox(value=generate_params['temperature_last'], label='temperature_last', info='将 temperature/dynamic temperature/quadratic sampling 移动到采样器堆栈的末尾，忽略它们在“采样器优先级”中的位置。')
                            shared.gradio['sampler_priority'] = gr.Textbox(value=generate_params['sampler_priority'], lines=12, label='采样器优先级', info='用换行符或逗号分隔的参数名称。')

                        with gr.Column():
                            shared.gradio['truncation_length'] = gr.Slider(value=get_truncation_length(), minimum=shared.settings['truncation_length_min'], maximum=shared.settings['truncation_length_max'], step=256, label='将提示词截断到此长度', info='如果提示词超过此长度，则会移除最左边的令牌。大多数模型要求此长度最多为 2048。')
                            shared.gradio['prompt_lookup_num_tokens'] = gr.Slider(value=shared.settings['prompt_lookup_num_tokens'], minimum=0, maximum=10, step=1, label='prompt_lookup_num_tokens', info='激活提示词查找解码。')
                            shared.gradio['max_tokens_second'] = gr.Slider(value=shared.settings['max_tokens_second'], minimum=0, maximum=20, step=1, label='每秒最大令牌数', info='使文本实时可读。')
                            shared.gradio['max_updates_second'] = gr.Slider(value=shared.settings['max_updates_second'], minimum=0, maximum=24, step=1, label='每秒最大 UI 更新次数', info='如果在流式传输期间 UI 出现滞后，请设置此选项。')
                            shared.gradio['seed'] = gr.Number(value=shared.settings['seed'], label='种子 (-1 表示随机)')
                            shared.gradio['skip_special_tokens'] = gr.Checkbox(value=shared.settings['skip_special_tokens'], label='跳过特殊令牌tokens', info='某些特定模型需要取消设置此选项。')
                            shared.gradio['stream'] = gr.Checkbox(value=shared.settings['stream'], label='激活文本流式传输')

        ui_chat.create_chat_settings_ui()


def create_event_handlers():
    shared.gradio['filter_by_loader'].change(loaders.blacklist_samplers, gradio('filter_by_loader', 'dynamic_temperature'), gradio(loaders.list_all_samplers()), show_progress=False)
    shared.gradio['preset_menu'].change(presets.load_preset_for_ui, gradio('preset_menu', 'interface_state'), gradio('interface_state') + gradio(presets.presets_params()))
    shared.gradio['random_preset'].click(presets.random_preset, gradio('interface_state'), gradio('interface_state') + gradio(presets.presets_params()))
    shared.gradio['grammar_file'].change(load_grammar, gradio('grammar_file'), gradio('grammar_string'))
    shared.gradio['dynamic_temperature'].change(lambda x: [gr.update(visible=x)] * 3, gradio('dynamic_temperature'), gradio('dynatemp_low', 'dynatemp_high', 'dynatemp_exponent'))


def get_truncation_length():
    if 'max_seq_len' in shared.provided_arguments or shared.args.max_seq_len != shared.args_defaults.max_seq_len:
        return shared.args.max_seq_len
    elif 'n_ctx' in shared.provided_arguments or shared.args.n_ctx != shared.args_defaults.n_ctx:
        return shared.args.n_ctx
    else:
        return shared.settings['truncation_length']


def load_grammar(name):
    p = Path(f'grammars/{name}')
    if p.exists():
        return open(p, 'r', encoding='utf-8').read()
    else:
        return ''
