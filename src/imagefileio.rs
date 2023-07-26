use std::ffi::{CStr, CString, c_void};
use std::os::raw::c_int;
use std::os::raw::c_char;
use std::{ptr, mem};

pub struct ImageFileDecoder {
    format_ctx: *mut ffmpeg::AVFormatContext,
    codec: *mut ffmpeg::AVCodec,
    codec_ctx: *mut ffmpeg::AVCodecContext,
    pub width: u32,
    pub height: u32
}

trait AvOk<T> {
    fn is_av_ok(&self, msg: &str) -> Result<T, String>;
}

impl AvOk<c_int> for c_int {
    fn is_av_ok(&self, msg: &str) -> Result<c_int, String> {
        if *self < 0 {
            return Err(format!("[FFmpeg]: {msg} - {}", av_err(*self)))
        }

        Ok(*self)
    }
}

impl<T> AvOk<*mut T> for *mut T {
    fn is_av_ok(&self, msg: &str) -> Result<*mut T, String> {
        if (*self).is_null() {
            return Err(format!("[FFmpeg]: {msg}"))
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

pub fn encode(file_path: &str, buffer: *const u8, width: i32, height: i32) -> Result<(), String> {
    let codec_type = ffmpeg::AVCodecID::AV_CODEC_ID_MJPEG;
    let output_pix_fmt = ffmpeg::AVPixelFormat::AV_PIX_FMT_YUVJ420P;
    let input_format_name = CString::new("mjpeg").unwrap();

    let input_pix_fmt  = ffmpeg::AVPixelFormat::AV_PIX_FMT_RGBA;

    // PNG
    //let codec_type = ffmpeg::AVCodecID::AV_CODEC_ID_PNG;
    //let output_pix_fmt = ffmpeg::AVPixelFormat::AV_PIX_FMT_RGBA;

    // BMP
    //let codec_type = ffmpeg::AVCodecID::AV_CODEC_ID_BMP;
    //let output_pix_fmt = ffmpeg::AVPixelFormat::AV_PIX_FMT_BGRA;

    // We're allocating a lot of C data and potentially exiting at any point 
    // if any error occurs. So, we use this struct to conveniently drop/free data on exit
    struct Data {
        codec_context:  *mut ffmpeg::AVCodecContext,
        format_context: *mut ffmpeg::AVFormatContext,
        rgba_frame:     *mut ffmpeg::AVFrame,
        yuv_frame:      *mut ffmpeg::AVFrame,
        yuv_data:       *mut c_void,
        sws_context:    *mut ffmpeg::SwsContext,
        packet:         *mut ffmpeg::AVPacket,
        open:           bool
    }

    impl Drop for Data {
        fn drop(&mut self) {
            unsafe {
                ffmpeg::avcodec_free_context(&mut self.codec_context);
                ffmpeg::av_frame_free(&mut self.rgba_frame);
                ffmpeg::av_frame_free(&mut self.yuv_frame);
                ffmpeg::av_free(self.yuv_data);
                ffmpeg::sws_freeContext(self.sws_context);
                ffmpeg::av_packet_free(&mut self.packet);
                if self.open {
                    ffmpeg::avio_close((*self.format_context).pb);
                }
                ffmpeg::avformat_free_context(self.format_context);
            }
        }
    }

    // Initialize and zero out all the pointers
    let mut d: Data = unsafe { mem::zeroed() };

    unsafe {

    // Find the JPEG encoder
    let codec_name_cstr = CStr::from_ptr(ffmpeg::avcodec_get_name(codec_type));
    let codec_name = codec_name_cstr.to_string_lossy().into_owned();

    let codec = ffmpeg::avcodec_find_encoder(codec_type)
        .is_av_ok(&format!("Failed to find suitable encoder for type: {}", codec_name))?;

    // Create a codec context
    d.codec_context = ffmpeg::avcodec_alloc_context3(codec)
        .is_av_ok("Failed to allocate codec context")?;

    // Set codec parameters
    (*d.codec_context).codec_type = ffmpeg::AVMediaType::AVMEDIA_TYPE_VIDEO;
    (*d.codec_context).width = width;
    (*d.codec_context).height = height;
    (*d.codec_context).pix_fmt = output_pix_fmt;
    (*d.codec_context).time_base = ffmpeg::AVRational { num: 1, den: 30 };

    // Open the codec
    ffmpeg::avcodec_open2(d.codec_context, codec, ptr::null_mut())
        .is_av_ok("Failed to open codec")?;

    d.format_context = ffmpeg::avformat_alloc_context().is_av_ok("Failed to allocate format context")?;

    (*d.format_context).oformat = ffmpeg::av_guess_format(input_format_name.as_ptr(), std::ptr::null(), std::ptr::null()).is_av_ok("Could not find format name")?;
    (*d.format_context).video_codec_id = (*codec).id;

    // Open the output file
    let c_file_path  = CString::new(file_path).unwrap();
    ffmpeg::avio_open(&mut (*d.format_context).pb, c_file_path.as_ptr(), ffmpeg::AVIO_FLAG_WRITE).is_av_ok("Failed to open file")?;
    d.open = true;

    let stream = ffmpeg::avformat_new_stream(d.format_context, codec).is_av_ok("Failed to open stream")?;

    (*(*stream).codecpar).codec_type = ffmpeg::AVMediaType::AVMEDIA_TYPE_VIDEO;
    (*(*stream).codecpar).codec_id = codec_type;
    (*(*stream).codecpar).width = width;
    (*(*stream).codecpar).height = height;

    // Write the header of the file out
    ffmpeg::avformat_write_header(d.format_context, std::ptr::null_mut()).is_av_ok("Failed to write header")?;

    // Allocate RGBA frame and buffer
    d.rgba_frame = ffmpeg::av_frame_alloc();

    ffmpeg::av_image_fill_arrays((*d.rgba_frame).data.as_mut_ptr(), (*d.rgba_frame).linesize.as_mut_ptr(), buffer as *const u8,
                                 input_pix_fmt, width, height, 1);

    // Set RGBA frame properties
    (*d.rgba_frame).width = width;
    (*d.rgba_frame).height = height;
    (*d.rgba_frame).format = input_pix_fmt as i32;

    // Allocate YUV frame and buffer
    d.yuv_frame = ffmpeg::av_frame_alloc().is_av_ok("Failed to allocate frame")?;
    let yuv_data_size = ffmpeg::av_image_get_buffer_size(output_pix_fmt, width, height, 1)
        .is_av_ok("Failed to get buffer size")?;

    d.yuv_data = ffmpeg::av_malloc(yuv_data_size as usize).is_av_ok("Failed to allocate YUV data")?;

    ffmpeg::av_image_fill_arrays(
        (*d.yuv_frame).data.as_mut_ptr(), (*d.yuv_frame).linesize.as_mut_ptr(), d.yuv_data as *const u8,
        output_pix_fmt, width, height, 1).is_av_ok("Failed to fill yuv data frame array")?;

    // Set YUV frame properties
    (*d.yuv_frame).width = width;
    (*d.yuv_frame).height = height;
    (*d.yuv_frame).format = output_pix_fmt as i32;

    // Create scaling context
    d.sws_context = ffmpeg::sws_getContext(
        width, height, input_pix_fmt,
        width, height, output_pix_fmt,
        ffmpeg::SWS_FAST_BILINEAR, ptr::null_mut(), ptr::null_mut(), ptr::null_mut())
        .is_av_ok("Failed to create scaling context")?;

    // Convert RGBA to YUVJ444P
    ffmpeg::sws_scale(
        d.sws_context, (*d.rgba_frame).data.as_ptr() as *const *const u8, (*d.rgba_frame).linesize.as_mut_ptr(),
        0, height, (*d.yuv_frame).data.as_mut_ptr(), (*d.yuv_frame).linesize.as_mut_ptr());

    // Encode and write frames
    d.packet = ffmpeg::av_packet_alloc().is_av_ok("Failed to allocate packet")?;

    (*d.packet).stream_index = (*stream).index;

    let write_to_encoder = || -> Result<(), String> {
        let mut ret = ffmpeg::avcodec_send_frame(d.codec_context, d.yuv_frame)
            .is_av_ok("Failed to send frame")?;

        while ret >= 0 {
            ret = ffmpeg::avcodec_receive_packet(d.codec_context, d.packet);

            if ret == ffmpeg::AVERROR(ffmpeg::EAGAIN) || ret == ffmpeg::AVERROR_EOF {
                break;
            }
            ret.is_av_ok("Failed to receive packet")?;

//                output_file.write_all(std::slice::from_raw_parts((*d.packet).data, (*d.packet).size as usize)).unwrap();
            ffmpeg::av_write_frame(d.format_context, d.packet).is_av_ok("Failed to write frame")?;
            ffmpeg::av_packet_unref(d.packet);
        }

        Ok(())
    };

    // Write data out to the encoder
    write_to_encoder()?;

    // A second time flushes the encoder
    write_to_encoder()?;

    ffmpeg::av_write_trailer(d.format_context).is_av_ok("Failed to write trailer")?;

    }
    Ok(())
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
