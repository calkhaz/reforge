# TODO: https://github.com/hamdanal/rich-argparse
import argparse
import ffmpeg
import subprocess
from subprocess import Popen

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
    print(probe)
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
    return subprocess.Popen(args, stdout=subprocess.PIPE)

def ffmpeg_encode_process(out_filename, width, height):
    args = (
        ffmpeg
        .input('pipe:', format='rawvideo', pix_fmt='rgba', s='{}x{}'.format(width, height))
        .output(out_filename, pix_fmt='yuv420p')
        .overwrite_output()
        .compile()
    )
    return subprocess.Popen(args, stdin=subprocess.PIPE)

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

def main():
    width, height = get_video_size(args.input_file)
    decoder = ffmpeg_decode_process(args.input_file)
    encoder = ffmpeg_encode_process(args.output_file, width, height) if args.output_file else None

    bytes_per_frame = width * height * 4
    use_swapchain = args.output_file is None

    rf = reforge.Reforge(shader_path = "shaders")
    renderer = rf.new_renderer(width, height, config_path=args.config_path, use_swapchain = use_swapchain)
    output_frame = bytearray(bytes_per_frame) if args.output_file else None

    out_of_frames = False

    while True:
        in_frame = read_frame(decoder, bytes_per_frame)
        out_of_frames = True if in_frame is None else False

        if out_of_frames: break

        renderer.execute(in_frame, output_frame)

        if renderer.requested_exit(): break

        if output_frame:
            write_frame(encoder, output_frame)

    if decoder.stdout: decoder.stdout.close()

    if out_of_frames: decoder.wait()
    else:             decoder.terminate()

    if encoder:
        if encoder.stdin is not None:
            encoder.stdin.close()
        encoder.wait()
