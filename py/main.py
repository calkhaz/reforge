# TODO: https://github.com/hamdanal/rich-argparse
import argparse
from types import ModuleType
import ffmpeg
import subprocess
#import ui
from subprocess import Popen
import asyncio
import importlib
import importlib.util
import os
import time

import sys

class Args:
    input_file: str
    output_file: str | None
    config_path: str | None
    shader_file_path: str | None

def parse_args() -> Args:
    parser = argparse.ArgumentParser(description="Reforge")
    parser.add_argument('shader_file_path', help="Direct shader file", metavar="shader-file", default='', nargs='?')
    parser.add_argument('-i', '--input',    help="Input path")
    parser.add_argument('-o', '--output',   help="Output path")
    parser.add_argument('-c', '--config',   help="Config path")
    parser.add_argument('-l', '--lib-path', help="Reforge library path (where the .so is)")

    pargs = parser.parse_args()

    a = Args()
    a.input_file = pargs.input
    a.output_file = pargs.output
    a.config_path = pargs.config
    a.shader_file_path = pargs.shader_file_path

    if pargs.lib_path:
        sys.path.insert(1, pargs.lib_path)

    return a

args = parse_args()
import reforge as reforge

def get_video_size(filename):
    probe = ffmpeg.probe(filename)
    video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
    width =  int(video_info['width'])
    height = int(video_info['height'])
    return width, height

def ffmpeg_decode_process(in_filename):
    args = (
        ffmpeg
        .input(in_filename)
        .output('pipe:', format='rawvideo', pix_fmt='rgba')
        .compile()
    )
    return subprocess.Popen(args, stdin=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdout=subprocess.PIPE)

def ffmpeg_encode_process(out_filename, width, height):
    args = (
        ffmpeg
        .input('pipe:', format='rawvideo', pix_fmt='rgba', s='{}x{}'.format(width, height))
        .output(out_filename, pix_fmt='yuv420p')
        .overwrite_output()
        .compile()
    )
    return subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.PIPE)

def read_frame(process: Popen[bytes], bytes_amount: int) -> bytes | None:

    if process.stdout is None:
        return None

    in_bytes = process.stdout.read(bytes_amount)

    if len(in_bytes) == 0: return None
    else: return in_bytes

def write_frame(encoder, frame: bytearray):
    encoder.stdin.write(frame)

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

def file_timestamp(path: str) -> float:
    try:
        return os.path.getmtime(path)

    # Sometimes if we catch the file during save, we end up here
    # In such a case, we'll simply sleep .5ms and check again
    except FileNotFoundError:
        time.sleep(.5/1e3)
        return os.path.getmtime(path)



async def run_reforge():
    width, height = get_video_size(args.input_file)
    decoder = ffmpeg_decode_process(args.input_file)
    encoder = ffmpeg_encode_process(args.output_file, width, height) if args.output_file else None

    bytes_per_frame = width * height * 4
    use_swapchain = args.output_file is None

    config_path = args.config_path

    if not config_path:
        print("Need python config filepath")
        sys.exit(-1)

    module = load_python_config(config_path)
    last_config_modification_time = file_timestamp(config_path)
    last_graph = module.graph

    rf = reforge.Reforge(shader_path = "shaders")
    renderer = rf.new_renderer(module.graph, width, height, use_swapchain = use_swapchain)
    output_frame = bytearray(bytes_per_frame) if args.output_file else None

    write_config_to_buffer(renderer, module)

    out_of_frames = False

    in_frame = None
    
    num_frames = 0

    while True:
        # Decode the next frame
        if not out_of_frames:
            next_frame = read_frame(decoder, bytes_per_frame)

            # No frames left to decode
            if next_frame is None:
                # Restart decoder so we can infinitely loop videos in the swapchain
                if num_frames > 1 and use_swapchain:
                    if decoder.stdout: decoder.stdout.close()
                    decoder.wait()
                    decoder = ffmpeg_decode_process(args.input_file)
                    continue
                else:
                    out_of_frames = True
            else:
                num_frames += 1
                in_frame = next_frame


        assert in_frame is not None
        if out_of_frames and not use_swapchain: break

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

        if output_frame and not out_of_frames:
            write_frame(encoder, output_frame)

    if decoder.stdout: decoder.stdout.close()

    if out_of_frames: decoder.wait()
    else:             decoder.terminate()

    if encoder:
        if encoder.stdin is not None:
            encoder.stdin.close()
        encoder.wait()

def main():
    try:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(run_reforge())
    except KeyboardInterrupt:
        pass
    #ui.run_ui(run_reforge)
