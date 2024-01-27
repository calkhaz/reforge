from subprocess import Popen
from typing import Tuple
import subprocess

import ffmpeg

class Decoder:
    process: Popen[bytes]
    width:  int
    height: int
    out_of_frames: bool = False

    def __init__(self, path: str):
        self.width, self.height = self._get_video_size(path)

        args = (
            ffmpeg
            .input(path)
            #.filter('scale', 400, -1)
            .output('pipe:', format='rawvideo', pix_fmt='rgba')
            .compile()
        )
        self.process = subprocess.Popen(args, stdin=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdout=subprocess.PIPE)

    def __del__(self):
        self.close()

    def close(self):
        if self.process.stdout: self.process.stdout.close()
        self.process.wait()

    def read_frame(self, bytes_amount: int) -> bytes | None:
    
        if self.process.stdout is None:
            return None
    
        in_bytes = self.process.stdout.read(bytes_amount)
    
        if len(in_bytes) == 0:
            self.out_of_frames = True
            return None
        else:
            return in_bytes

    def _get_video_size(self, path: str) -> Tuple[int, int]:
        probe = ffmpeg.probe(path)
        video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
        width =  int(video_info['width'])
        height = int(video_info['height'])
        return width, height


class Encoder:
    process: Popen[bytes]

    def __init__(self, path: str, width: int, height: int):
        args = (
            ffmpeg
            .input('pipe:', format='rawvideo', pix_fmt='rgba', s='{}x{}'.format(width, height))
            .output(path, pix_fmt='yuv420p')
            .overwrite_output()
            .compile()
        )
        self.process = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.PIPE)

    def __del__(self):
        self.close()

    def write_frame(self, frame: bytearray):
        if self.process.stdin is None:
            raise RuntimeError("Error: Cannot write to ffmpeg process stdin - it is None")

        self.process.stdin.write(frame)

    def close(self):
        if self.process.stdin:
            self.process.stdin.close()
        self.process.wait()

