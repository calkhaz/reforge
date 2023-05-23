extern crate ffmpeg_next as ffmpeg;

pub struct ImageFileDecoder {
    decoder: ffmpeg::decoder::Video,
    input_context: ffmpeg::format::context::Input,
    pub width: u32,
    pub height: u32
}

pub fn init() {
    ffmpeg::init().unwrap();
    ffmpeg::log::set_level(ffmpeg::log::Level::Verbose);

}

impl ImageFileDecoder {
    pub fn new(file_path: &str) -> Self {
        let input_context = ffmpeg::format::input(&file_path).unwrap();

        let stream = input_context
            .streams()
            .best(ffmpeg::media::Type::Video)
            .ok_or(ffmpeg::Error::StreamNotFound).unwrap();

        let context_decoder = ffmpeg::codec::context::Context::from_parameters(stream.parameters()).unwrap();
        let decoder = context_decoder.decoder().video().unwrap();

        let width = decoder.width();
        let height = decoder.height();

        ImageFileDecoder {
            decoder: decoder,
            input_context: input_context,
            width: width,
            height: height
        }
    }

    pub fn decode(&mut self, buffer: *mut u8, scale_width: u32, scale_height: u32) {
        let mut scaler = ffmpeg::software::scaling::context::Context::get(
            self.decoder.format(),
            self.decoder.width(),
            self.decoder.height(),
            ffmpeg::format::Pixel::RGBA, // 8 bits per channel
            scale_width,
            scale_height,
            ffmpeg::software::scaling::flag::Flags::LANCZOS,
        ).unwrap();

        let mut frame_index = 0;

        let best_video_stream_index = self.input_context
            .streams()
            .best(ffmpeg::media::Type::Video)
            .map(|stream| stream.index());

        let mut receive_and_process_decoded_frames =
            |decoder: &mut ffmpeg::decoder::Video| -> Result<(), ffmpeg::Error> {
                let mut decoded = ffmpeg::util::frame::Video::empty();
                while decoder.receive_frame(&mut decoded).is_ok() {
                    let mut rgb_frame = ffmpeg::util::frame::Video::empty();
                    scaler.run(&decoded, &mut rgb_frame)?;
                    Self::write_frame_to_buffer(&rgb_frame, buffer);
                    frame_index += 1;

                    // TODO: Check how to the number of frames in advance
                    break;
                }
                Ok(())
            };

        for (stream, packet) in self.input_context.packets() {
            if stream.index() == best_video_stream_index.unwrap() {
                self.decoder.send_packet(&packet).unwrap();
                receive_and_process_decoded_frames(&mut self.decoder).unwrap();
            }
        }
        self.decoder.send_eof().unwrap();
        receive_and_process_decoded_frames(&mut self.decoder).unwrap();

        println!("format: {:?}", self.decoder.format());
        println!("codec format: {:?}", self.decoder.codec().unwrap().name());
    }

    fn write_frame_to_buffer(frame: &ffmpeg::util::frame::video::Video, buffer: *mut u8) {
        // https://github.com/zmwangx/rust-ffmpeg/issues/64
        let data = frame.data(0);
        let stride = frame.stride(0);
        let byte_width: usize = 4 * frame.width() as usize;
        let width: usize = frame.width() as usize;
        let height: usize = frame.height() as usize;

        let mut buffer_index: usize = 0;

        unsafe {
        // *4 for rgba
        let buffer_slice = std::slice::from_raw_parts_mut(buffer, width*height*4);

        for line in 0..height {
            let begin = line * stride;
            let end = begin + byte_width;
            let len = end-begin;
            buffer_slice[buffer_index..buffer_index+len].copy_from_slice(&data[begin..end]);
            buffer_index = buffer_index+len;
        }
        }
    }
}
