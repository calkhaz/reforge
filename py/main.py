# TODO: https://github.com/hamdanal/rich-argparse
import argparse
from types import ModuleType
#import ui
import asyncio
import importlib
import importlib.util
import os
import time

from ffmpeg_interface import Encoder, Decoder

import sys

class Args:
    input_file: str
    output_file: str | None
    config_path: str | None
    shader_file_path: str | None
    shader_dir: str

def parse_args() -> Args:
    parser = argparse.ArgumentParser(description="Reforge")
    parser.add_argument('shader_file_path',   help="Direct shader file", metavar="shader-file", default='', nargs='?')
    parser.add_argument('-i', '--input',      help="Input path")
    parser.add_argument('-o', '--output',     help="Output path")
    parser.add_argument('-c', '--config',     help="Config path")
    parser.add_argument('-l', '--lib-path',   help="Reforge library path (where the .so is)")
    parser.add_argument('-d', '--shader-dir', help="Directory to find shaders in", default="shaders")

    pargs = parser.parse_args()

    a = Args()
    a.input_file = pargs.input
    a.output_file = pargs.output
    a.config_path = pargs.config
    a.shader_file_path = pargs.shader_file_path
    a.shader_dir = pargs.shader_dir

    if pargs.lib_path:
        sys.path.insert(1, pargs.lib_path)

    return a

args = parse_args()
import reforge as reforge

def ms_s(val: float) -> str:
    return f'{val*1000:.2f}ms'

# TODO: types for module https://stackoverflow.com/questions/69090545/typehint-importing-module-dynamically-using-importlib
def load_python_config(path: str) -> ModuleType | None:
    spec = importlib.util.spec_from_file_location("module_name", path)
    if spec and spec.loader:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    return None

def write_config_to_buffer(renderer: reforge.Renderer, module: ModuleType):
    # Set all buffer values from config
    if module.nodes and isinstance(module.nodes,dict):
        for key, value in module.nodes.items():
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    renderer.set_buffer(key, subkey, subvalue)
                    # print(key, subkey, subvalue)

def file_timestamp(path: str | None) -> float:
    if path is None:
        return 0.0

    try:
        return os.path.getmtime(path)

    # Sometimes if we catch the file during save, we end up here
    # In such a case, we'll simply sleep .5ms and check again
    except FileNotFoundError:
        time.sleep(.5/1e3)
        return os.path.getmtime(path)

async def run_reforge():
    decoder = Decoder(args.input_file) if args.input_file else None

    width, height = (decoder.width, decoder.height) if decoder else (800, 600)
    encoder = Encoder(args.output_file, width, height) if args.output_file else None
    shader_dir = args.shader_dir

    bytes_per_frame = width * height * 4
    use_swapchain = args.output_file is None

    config_path = args.config_path

    graph = ""
    last_graph = ""
    last_config_modification_time = 0

    module: None | ModuleType = None
    if config_path:
        module = load_python_config(config_path)

        if not module:
            print(f"Failed to load python config from {config_path}")
            sys.exit(-1)

        last_config_modification_time = file_timestamp(config_path)
        graph = last_graph = module.graph
    elif args.shader_file_path:
        shader_dir = os.path.dirname(args.shader_file_path)
        file_name = os.path.basename(args.shader_file_path)
        if args.input_file: graph = f"input -> {file_name} -> output"
        else:               graph = f"{file_name} -> output"

    rf = reforge.Reforge(shader_path = shader_dir)
    renderer = rf.new_renderer(graph, width, height, use_swapchain = use_swapchain)
    output_frame = bytearray(bytes_per_frame) if args.output_file else None

    if module:
        write_config_to_buffer(renderer, module)

    in_frame = None
    num_frames = 0

    while True:
        # Decode the next frame
        if decoder and not decoder.out_of_frames:
            next_frame = decoder.read_frame(bytes_per_frame)

            # No frames left to decode
            if next_frame is None:
                # Restart decoder so we can infinitely loop videos in the swapchain
                if num_frames > 1 and use_swapchain:
                    decoder.close()
                    decoder = Decoder(args.input_file)
                    continue
            else:
                num_frames += 1
                in_frame = next_frame


        # Don't run forever if we aren't running in a window
        if decoder and decoder.out_of_frames and not use_swapchain:
            break

        config_modification_time = file_timestamp(config_path)
        if config_modification_time != last_config_modification_time:
            module = load_python_config(config_path)
            if module.graph != last_graph:
                renderer.reload_graph(module.graph)
                last_graph = module.graph
            last_config_modification_time = config_modification_time
            write_config_to_buffer(renderer, module)

        renderer.execute(in_frame, output_frame)
        if renderer.requested_exit(): break

        if encoder and output_frame:
            encoder.write_frame(output_frame)

def main():
    try:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(run_reforge())
    except KeyboardInterrupt:
        pass
    #ui.run_ui(run_reforge)
