use std::{process::Command, io::{Read, Write}};
use anyhow::{anyhow, Context, Result};
use tracing::trace;
use crate::utils;

pub struct Decoder {
    stdout_reader: std::io::BufReader<std::process::ChildStdout>,
    pub width: u32,
    pub height: u32,
    pub num_frames: u32
}

pub struct Encoder {
    stdin_writer: std::io::BufWriter<std::process::ChildStdin>
}

fn ffprobe_info(input_path: &str) -> Result<(u32, u32, u32)> {
    let cmd = Command::new("ffprobe")
        .args([
            "-loglevel", "error", // Only show critical messages
            "-select_streams", "v:0",
            "-count_packets",
            "-show_entries", "stream=width,height,nb_read_packets",
            "-of", "csv=p=0",
            input_path
         ])
        .stdout(std::process::Stdio::piped())
        .spawn()?;

    let output = cmd.wait_with_output()?;

    match output.status.code() {
        Some(code) => {
            let stdout_str : &str = std::str::from_utf8(&output.stdout).unwrap();
            if code != 0 {
                let stderr_str : &str = std::str::from_utf8(&output.stderr).unwrap();

                return Err(anyhow!("ffprobe exited with status code: {} - {} - {}", code, stdout_str, stderr_str));
            }

            let parts: Vec<&str> = stdout_str.split(',').take(3).collect();

            if parts.len() != 3 {
                return Err(anyhow!("ffprobe returned nonsense: {}", stdout_str));
            }

            let parse_err = |e: &str| format!("Failed to parse ffprobe - Tried to parse ({}) as u32", e);

            let width     : u32 = parts[0].trim().parse().context(parse_err(parts[0]))?;
            let height    : u32 = parts[1].trim().parse().context(parse_err(parts[1]))?;
            let num_frames: u32 = parts[2].trim().parse().context(parse_err(parts[2]))?;

            Ok((width, height, num_frames))
        },
        None => Err(anyhow!("ffprobe terminated by signal"))
    }
}

impl Decoder {
    pub fn new(input_path: &str, desired_width: Option<u32>, desired_height: Option<u32>) -> Result<Decoder> {
        let span = tracing::trace_span!("decode");
        let _guard = span.enter();

        let (width, height, num_frames) = ffprobe_info(input_path)?;

        trace!("Original size: {} X {}", width, height);
        trace!("Num frames: {}", num_frames);

        let (width, height) = utils::get_dim(width, height as u32, desired_width, desired_height);

        let scale_filter = format!("scale={}:{}", width, height);

        let mut args = vec![
            "-loglevel", "error", // Only show critical messages
            "-i", input_path,
            "-f", "rawvideo",
            "-pix_fmt", "rgba",
        ];

        // Only add the scale filter if resize is requested
        if desired_width.is_some() || desired_height.is_some() {
            trace!("Scaling to size: {} X {}", width, height);
            args.push("-vf");
            args.push(&scale_filter);
        }

        args.push("-"); // Output to stdin

        let mut cmd = Command::new("ffmpeg")
            .args(args)
            .stdout(std::process::Stdio::piped())
            .spawn()?;

        let stdout = cmd.stdout.take().expect("Failed to obtain decoder stdout");
        let stdout_reader = std::io::BufReader::new(stdout);

        Ok(Decoder{stdout_reader, width, height, num_frames})
    }

    pub fn read_frame(&mut self) -> Result<(Vec<u8>, bool)> {
        let frame_size = self.width as usize * self.height as usize * 4;
        // Set capacity + set len to avoid callocing
        let mut buffer: Vec<u8> = Vec::with_capacity(frame_size);
        unsafe { buffer.set_len(frame_size); }

        match self.stdout_reader.read_exact(&mut buffer) {
            Ok(_) => {
                Ok((buffer, false))
            },
            Err(err) => {
                if err.kind() == std::io::ErrorKind::UnexpectedEof {
                    Ok((Vec::new(), true))
                }
                else {
                    Err(err.into())
                }
            }
        }
    }
}

impl Encoder {
    pub fn new(output_file: &str, width: u32, height: u32) -> Result<Encoder, std::io::Error> {
        let mut cmd = Command::new("ffmpeg")
            .args([
                "-loglevel", "error", // Only show critical messages
                "-y",                 // Overwrite output without asking
                "-f", "rawvideo",
                "-pix_fmt", "rgba",
                "-video_size", format!("{}x{}", width, height).as_str(),
                "-i", "-", // Read from STDIN
                output_file
             ])
            .stdin(std::process::Stdio::piped())
            .spawn()?;

        let stdin = cmd.stdin.take().expect("Failed to obtain decoder stdout");
        let stdin_writer = std::io::BufWriter::new(stdin);

        Ok(Encoder{stdin_writer})
    }

    pub fn write_frame(&mut self, buffer: &[u8]) -> Result<()> {
        self.stdin_writer.write_all(buffer)?;
        Ok(())
    }
}
