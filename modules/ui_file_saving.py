import gradio as gr

from modules import chat, presets, shared, ui, utils
from modules.utils import gradio


def create_ui():
    mu = shared.args.multi_user

    # 文本文件保存器
    with gr.Group(visible=False, elem_classes='file-saver') as shared.gradio['file_saver']:
        shared.gradio['save_filename'] = gr.Textbox(lines=1, label='文件名')
        shared.gradio['save_root'] = gr.Textbox(lines=1, label='文件文件夹', info='仅供参考，不可修改', interactive=False)
        shared.gradio['save_contents'] = gr.Textbox(lines=10, label='文件内容')
        with gr.Row():
            shared.gradio['save_cancel'] = gr.Button('取消', elem_classes="small-button")
            shared.gradio['save_confirm'] = gr.Button('保存', elem_classes="small-button", variant='primary', interactive=not mu)

    # 文本文件删除器
    with gr.Group(visible=False, elem_classes='file-saver') as shared.gradio['file_deleter']:
        shared.gradio['delete_filename'] = gr.Textbox(lines=1, label='文件名')
        shared.gradio['delete_root'] = gr.Textbox(lines=1, label='文件文件夹', info='仅供参考，不可修改', interactive=False)
        with gr.Row():
            shared.gradio['delete_cancel'] = gr.Button('取消', elem_classes="small-button")
            shared.gradio['delete_confirm'] = gr.Button('删除', elem_classes="small-button", variant='stop', interactive=not mu)

    # 角色保存器/删除器
    with gr.Group(visible=False, elem_classes='file-saver') as shared.gradio['character_saver']:
        shared.gradio['save_character_filename'] = gr.Textbox(lines=1, label='文件名', info='角色将被保存到你的 characters/ 文件夹，并以该文件名作为基础')
        with gr.Row():
            shared.gradio['save_character_cancel'] = gr.Button('取消', elem_classes="small-button")
            shared.gradio['save_character_confirm'] = gr.Button('保存', elem_classes="small-button", variant='primary', interactive=not mu)

    with gr.Group(visible=False, elem_classes='file-saver') as shared.gradio['character_deleter']:
        gr.Markdown('确认删除角色？')
        with gr.Row():
            shared.gradio['delete_character_cancel'] = gr.Button('取消', elem_classes="small-button")
            shared.gradio['delete_character_confirm'] = gr.Button('删除', elem_classes="small-button", variant='stop', interactive=not mu)

    # 预设保存器
    with gr.Group(visible=False, elem_classes='file-saver') as shared.gradio['preset_saver']:
        shared.gradio['save_preset_filename'] = gr.Textbox(lines=1, label='文件名', info='预设将被保存到你的 presets/ 文件夹，并以该文件名作为基础')
        shared.gradio['save_preset_contents'] = gr.Textbox(lines=10, label='文件内容')
        with gr.Row():
            shared.gradio['save_preset_cancel'] = gr.Button('取消', elem_classes="small-button")
            shared.gradio['save_preset_confirm'] = gr.Button('保存', elem_classes="small-button", variant='primary', interactive=not mu)


def create_event_handlers():
    shared.gradio['save_confirm'].click(
        lambda x, y, z: utils.save_file(x + y, z), gradio('save_root', 'save_filename', 'save_contents'), None).then(
        lambda: gr.update(visible=False), None, gradio('file_saver'))

    shared.gradio['delete_confirm'].click(
        lambda x, y: utils.delete_file(x + y), gradio('delete_root', 'delete_filename'), None).then(
        lambda: gr.update(visible=False), None, gradio('file_deleter'))

    shared.gradio['delete_cancel'].click(lambda: gr.update(visible=False), None, gradio('file_deleter'))
    shared.gradio['save_cancel'].click(lambda: gr.update(visible=False), None, gradio('file_saver'))

    shared.gradio['save_character_confirm'].click(
        chat.save_character, gradio('name2', 'greeting', 'context', 'character_picture', 'save_character_filename'), None).then(
        lambda: gr.update(visible=False), None, gradio('character_saver')).then(
        lambda x: gr.update(choices=utils.get_available_characters(), value=x), gradio('save_character_filename'), gradio('character_menu'))

    shared.gradio['delete_character_confirm'].click(
        lambda x: str(utils.get_available_characters().index(x)), gradio('character_menu'), gradio('temporary_text')).then(
        chat.delete_character, gradio('character_menu'), None).then(
        chat.update_character_menu_after_deletion, gradio('temporary_text'), gradio('character_menu')).then(
        lambda: gr.update(visible=False), None, gradio('character_deleter'))

    shared.gradio['save_character_cancel'].click(lambda: gr.update(visible=False), None, gradio('character_saver'))
    shared.gradio['delete_character_cancel'].click(lambda: gr.update(visible=False), None, gradio('character_deleter'))

    shared.gradio['save_preset'].click(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        presets.generate_preset_yaml, gradio('interface_state'), gradio('save_preset_contents')).then(
        lambda: 'My Preset', None, gradio('save_preset_filename')).then(
        lambda: gr.update(visible=True), None, gradio('preset_saver'))

    shared.gradio['save_preset_confirm'].click(
        lambda x, y: utils.save_file(f'presets/{x}.yaml', y), gradio('save_preset_filename', 'save_preset_contents'), None).then(
        lambda: gr.update(visible=False), None, gradio('preset_saver')).then(
        lambda x: gr.update(choices=utils.get_available_presets(), value=x), gradio('save_preset_filename'), gradio('preset_menu'))

    shared.gradio['save_preset_cancel'].click(lambda: gr.update(visible=False), None, gradio('preset_saver'))

    shared.gradio['delete_preset'].click(
        lambda x: f'{x}.yaml', gradio('preset_menu'), gradio('delete_filename')).then(
        lambda: 'presets/', None, gradio('delete_root')).then(
        lambda: gr.update(visible=True), None, gradio('file_deleter'))

    shared.gradio['save_grammar'].click(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        lambda x: x, gradio('grammar_string'), gradio('save_contents')).then(
        lambda: 'grammars/', None, gradio('save_root')).then(
        lambda: 'My Fancy Grammar.gbnf', None, gradio('save_filename')).then(
        lambda: gr.update(visible=True), None, gradio('file_saver'))

    shared.gradio['delete_grammar'].click(
        lambda x: x, gradio('grammar_file'), gradio('delete_filename')).then(
        lambda: 'grammars/', None, gradio('delete_root')).then(
        lambda: gr.update(visible=True), None, gradio('file_deleter'))
