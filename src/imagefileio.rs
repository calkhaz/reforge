use std::ffi::{CStr, CString};
use std::os::raw::c_int;
use std::os::raw::c_char;
use std::ptr;

pub struct ImageFileDecoder {
    format_ctx: *mut ffmpeg::AVFormatContext,
    codec: *mut ffmpeg::AVCodec,
    codec_ctx: *mut ffmpeg::AVCodecContext,
    pub width: u32,
    pub height: u32
}

trait AvOk {
    fn is_av_ok(&self, msg: &str) -> Result<c_int, String>;
}

impl AvOk for c_int {
    fn is_av_ok(&self, msg: &str) -> Result<c_int, String> {
        if *self < 0 {
            return Err(format!("[FFmpeg]: {msg} - {}", av_err(*self)))
        }

        Ok(*self)
    }
}

pub fn init() {
    unsafe {
    ffmpeg::av_log_set_level(ffmpeg::AV_LOG_VERBOSE);
    }
}

fn av_err(err_num: c_int) -> String {
    let empty_str = " ".repeat(ffmpeg::AV_ERROR_MAX_STRING_SIZE);
    let err_buf: *mut c_char = CString::new(empty_str).unwrap().into_raw();

    let err_c_str = unsafe {
        ffmpeg::av_make_error_string(err_buf, ffmpeg::AV_ERROR_MAX_STRING_SIZE, err_num);
        CStr::from_ptr(err_buf)
    };

    err_c_str.to_string_lossy().into_owned()
}

impl ImageFileDecoder {
    pub fn new(file_path: &str) -> Result<Self, String> {
        let mut decoder = ImageFileDecoder {
            format_ctx: std::ptr::null_mut(),
            codec:      std::ptr::null_mut(),
            codec_ctx:  std::ptr::null_mut(),
            width: 0,
            height: 0
        };

        unsafe {
        let path_cstr = CString::new(file_path).unwrap();
        let path = path_cstr.as_ptr() as *const c_char;

        // Open the file for reading
        ffmpeg::avformat_open_input(&mut decoder.format_ctx, path, ptr::null_mut(), ptr::null_mut())
            .is_av_ok(&format!("Failed to open input file: {}", file_path))?;

        // Retrieve the stream information
        ffmpeg::avformat_find_stream_info(decoder.format_ctx, std::ptr::null_mut())
            .is_av_ok("Failed to retrieve stream information")?;

        // Find the video stream
        let stream_index = ffmpeg::av_find_best_stream(decoder.format_ctx, ffmpeg::AVMediaType::AVMEDIA_TYPE_VIDEO, -1, -1, &mut decoder.codec, 0)
            .is_av_ok("Failed to find video stream")?;

        // Open the codec
        decoder.codec_ctx = ffmpeg::avcodec_alloc_context3(decoder.codec);
        if decoder.codec_ctx == std::ptr::null_mut() {
            return Err("Failed to allocate codec context".to_string());
        }

        let codec_params: *mut ffmpeg::AVCodecParameters = (**(*decoder.format_ctx).streams.offset(stream_index as isize)).codecpar;
        ffmpeg::avcodec_parameters_to_context(decoder.codec_ctx, codec_params).is_av_ok("Failed to copy codec parameters to context")?;

        ffmpeg::avcodec_open2(decoder.codec_ctx, decoder.codec, std::ptr::null_mut()).is_av_ok("Failed to open codec")?;

        (*decoder.codec_ctx).width .is_av_ok("Width was negative")?;
        (*decoder.codec_ctx).height.is_av_ok("Height was negative")?;
        decoder.width  = (*decoder.codec_ctx).width  as u32;
        decoder.height = (*decoder.codec_ctx).height as u32;
        }

        Ok(decoder)
    }

    pub fn decode(&mut self, buffer: *mut u8, scale_width: u32, scale_height: u32) -> Result<(), String> {
        unsafe {

        // Allocate a frame buffer
        let mut frame = ffmpeg::av_frame_alloc();
        if frame == std::ptr::null_mut() {
            return Err("Failed to allocate frame".to_string());
        }

        // Allocate a packet
        let mut packet = ffmpeg::av_packet_alloc();
        (*packet).data = std::ptr::null_mut();
        (*packet).size = 0;

        // Read the packet
        ffmpeg::av_read_frame(self.format_ctx, packet).is_av_ok("Failed to read packet")?;

        let mut got_frame: i32 = 0;
        ffmpeg::avcodec_decode_video2(self.codec_ctx, frame, &mut got_frame, packet).is_av_ok("Error decoding into frame")?;

        if got_frame == 0 {
            return Err("Could not decompress into frame".to_string());
        }

        let w = (*frame).width;
        let h = (*frame).height;

         // Create scaling context
         let sws_context = ffmpeg::sws_getContext(
             w, h, (*self.codec_ctx).pix_fmt,
             scale_width as i32, scale_height as i32, ffmpeg::AVPixelFormat::AV_PIX_FMT_RGBA,
             ffmpeg::SWS_LANCZOS, ptr::null_mut(), ptr::null_mut(), ptr::null_mut());

         if sws_context.is_null() {
             panic!("Failed to create scaling context");
         }

        let mut linesizes: [i32; 4] = [0; 4];

        ffmpeg::av_image_fill_linesizes(linesizes.as_mut_ptr(), ffmpeg::AVPixelFormat::AV_PIX_FMT_RGBA, scale_width as i32)
            .is_av_ok("Could not estimate line sizes")?;

         // Convert to RGBA
         ffmpeg::sws_scale(sws_context, (*frame).data.as_ptr() as *const *const u8, (*frame).linesize.as_mut_ptr(),
                       0, h, &buffer, linesizes.as_mut_ptr())
            .is_av_ok("Failed to perform sws scale")?;

        ffmpeg::sws_freeContext(sws_context);
        ffmpeg::av_frame_free(&mut frame);
        ffmpeg::av_packet_free(&mut packet);

        }

        Ok(())
    }
}

impl Drop for ImageFileDecoder {
    fn drop(&mut self) {
        unsafe {
        ffmpeg::avformat_close_input(&mut self.format_ctx);
        ffmpeg::avformat_free_context(self.format_ctx);
        ffmpeg::avcodec_free_context(&mut self.codec_ctx);
        }
    }
}
