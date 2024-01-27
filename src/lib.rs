extern crate ash;
extern crate clap;
extern crate gpu_allocator;
extern crate shaderc;

use pyo3::prelude::*;

mod config;
mod render;
mod utils;
mod vulkan;

use ash::vk;
use render::Render;
use render::RenderInfo;
use utils::TERM_CLEAR;
use render::ParamData;

use std::collections::HashMap;

use winit::{
    event::{ElementState, Event, KeyboardInput, VirtualKeyCode, WindowEvent},
    event_loop::{ControlFlow, EventLoop, EventLoopBuilder},
    platform::run_return::EventLoopExtRunReturn
};

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, clap::ValueEnum)]
#[pyclass]
pub enum ShaderFormat {
    Rgba8,
    Rgba32f
}

impl ShaderFormat {
    fn to_vk_format(self) -> vk::Format {
        match self {
            ShaderFormat::Rgba8 => vk::Format::R8G8B8A8_UNORM,
            ShaderFormat::Rgba32f => vk::Format::R32G32B32A32_SFLOAT
        }
    }
}

#[pyclass(unsendable)]
struct Reforge {
    shader_path: String,
}

#[pyclass(unsendable)]
struct Renderer {
    render: Render,
    time_since_start: std::time::Instant,
    event_loop: Option<EventLoop<()>>,
    requested_exit: bool
}

#[pymethods]
impl Renderer {
    pub fn gpu_times(&self) -> HashMap<String, f32> {
        self.render.last_frame_gpu_times()
    }

    pub fn idle(&self) -> bool {
        self.render.frame_fence_signaled()
    }

    pub fn requested_exit(&self) -> bool {
        self.requested_exit
    }

    #[pyo3(name = "execute")]
    pub fn execute_py(&mut self, input_bytes: Option<&[u8]>, py_output_bytes: Option<&pyo3::types::PyByteArray>) {
        let output_bytes = unsafe {
            if let Some(bytes) = py_output_bytes {
                Some(bytes.as_bytes_mut())
            }
            else { None }
        };

        self.execute(input_bytes, output_bytes);
    }

    pub fn set_buffer(&mut self, pipeline: String, param: String, data: &pyo3::types::PyAny) -> PyResult<()> {

        let param_map = self.render.pipeline_buffer_data.entry(pipeline).or_default();

        if data.is_instance_of::<pyo3::types::PyList>() {
            let list = data.extract::<&pyo3::types::PyList>()?;
            if let Ok(vec) = list.extract::<Vec<i32>>() {
                param_map.insert(param, ParamData::IntegerArray(vec));
            }
            else if let Ok(vec) = list.extract::<Vec<f32>>() {
                param_map.insert(param, ParamData::FloatArray(vec));
            }
            else {
                println!("Invalid vector in set_buffer");
            }
        }
        else if let Ok(val) = data.extract::<bool>() {
            param_map.insert(param, ParamData::Boolean(val));
        }
        else if let Ok(val) = data.extract::<i32>() {
            param_map.insert(param, ParamData::Integer(val));
        }
        else if let Ok(val) = data.extract::<f32>() {
            param_map.insert(param, ParamData::Float(val));
        }

        self.render.outdate_frames();
        Ok(())
    }

    pub fn reload_graph(&mut self, graph: String) {
        self.render.update_graph(graph);
    }

}
impl Renderer {
    pub fn execute(&mut self, input_bytes: Option<&[u8]>, output_bytes: Option<&mut [u8]>) {
        let mut first_run = vec![true; self.render.num_frames()];
        self.render.info.has_input_image = input_bytes.is_some();

        //let mut avg_ms = 0.0;
        let mapped_input_image_data: *mut u8 = self.render.staging_buffer_ptr();

        // Write bytes into staging image
        if let Some(input_bytes) = input_bytes {
            unsafe { std::ptr::copy_nonoverlapping(input_bytes.as_ptr(), mapped_input_image_data, input_bytes.len()); }
        }

        let timer = std::time::Instant::now();
        //let elapsed_ms = utils::get_elapsed_ms(&timer);
//        println!("reforge copy time: {:.2}ms", elapsed_ms);

        //let elapsed_ms = utils::get_elapsed_ms(&timer);
        //println!("File Decode and resize: {:.2}ms", elapsed_ms);

        let mut render_fn = |render: &mut Render| {
            // Wait for the previous iteration of this frame before
            // changing or executing on its resources
            render.wait_for_frame_fence();

            if render.trigger_reloads() {
                // Clear current line of timers
                eprint!("{TERM_CLEAR}");
                first_run.iter_mut().for_each(|b| *b = true);
            }

            render.update_ubos(self.time_since_start.elapsed().as_secs_f32());

            // Pull in the next image from the swapchain
            if render.has_swapchain() {
                render.acquire_swapchain();
            }

            //let elapsed_ms = utils::get_elapsed_ms(&timer);
            //avg_ms = utils::moving_avg(avg_ms, elapsed_ms);
            //let timer = std::time::Instant::now();

            let gpu_times = render.set_last_frame_gpu_times();
            //eprint!("\rFrame: {:5.2}ms, Frame-Avg: {:5.2}ms, GPU: {{{}}}", elapsed_ms, avg_ms, gpu_times);
            //eprint!("\rGPU: {:?}", gpu_times);

            render.begin_record();

            // On the first run, we:
            // 1. Transitioned images as needed
            // 2. Load the staging input buffer into an image and convert it to linear
            if first_run[render.frame_index] {
                if input_bytes.as_ref().is_some() {
                    render.record_initial_image_load();
                }
                render.record_pipeline_image_transitions();
                //first_run[render.frame_index] = false;
            }

            render.record();

            if output_bytes.is_some() {
                render.write_output_to_buffer();
            }

            render.end_record();

            // Send the work to the gpu
            render.submit();
        };

        let mut first_resize = true;

        if let Some(event_loop) = &mut self.event_loop {
            event_loop.run_return(|event, _, control_flow| {
                *control_flow = ControlFlow::Poll;
                match event {
                    Event::WindowEvent {
                        event: WindowEvent::Resized(_size),
                        ..
                    } => {
                        // This event gets triggered on initial window creation
                        // and we don't want to recreate the swapchain at that point
                        if !first_resize {
                            self.render.swapchain_rebuilt_required = true;
                        }

                        first_resize = false;
                    },
                    Event::WindowEvent {
                        event: WindowEvent::CloseRequested | WindowEvent::KeyboardInput {
                            input: KeyboardInput {
                                state: ElementState::Pressed,
                                virtual_keycode: Some(VirtualKeyCode::Escape),
                                ..
                            },
                            ..
                        },
                    .. } => {
                        *control_flow = ControlFlow::Exit;
                        self.requested_exit = true;
                    }
                    Event::MainEventsCleared => {
                        render_fn(&mut self.render);
                        *control_flow = ControlFlow::Exit;
                    }
                    _ => (),
                }
            });
        }
        else {
            //let mut timer: std::time::Instant = std::time::Instant::now();

            render_fn(&mut self.render);
            self.render.wait_for_frame_fence();

//            let elapsed_ms = utils::get_elapsed_ms(&timer);
//            println!("Actual reforge time: {:.2}ms", elapsed_ms);

//            let mut timer: std::time::Instant = std::time::Instant::now();

            if let Some(output_bytes) = output_bytes {
                unsafe { std::ptr::copy_nonoverlapping(mapped_input_image_data, output_bytes.as_mut_ptr(), output_bytes.len()); }
            }

//            let elapsed_ms = utils::get_elapsed_ms(&timer);
//            println!("copy time 2: {:.2}ms", elapsed_ms);
//            println!("total copy time lost: {:.2}ms", copy_total_ms);

        }
    }
}

#[pymethods]
impl Reforge {
    #[new]
    pub fn new(shader_path: String) -> Self{
        Reforge { shader_path }
    }

    pub fn new_renderer(&self, graph: String, width: u32, height: u32, num_workers: Option<u32>,
                        use_swapchain: Option<bool>, shader_file_path: Option<String>) -> PyResult<Renderer> {
        let render_info = RenderInfo {
            graph,
            width,
            height,
            num_frames: num_workers.unwrap_or(1) as usize,
            shader_path: self.shader_path.clone(),
            format: (ShaderFormat::Rgba32f).to_vk_format(),
            swapchain: use_swapchain.unwrap_or(false),
            has_input_image: true,
            shader_file_path
        };


        let event_loop = || {
            if cfg!(unix) || cfg!(windows) {
                #[cfg(unix)]
                use winit::platform::unix::EventLoopBuilderExtUnix;
                #[cfg(windows)]
                use winit::platform::windows::EventLoopBuilderExtWindows;

                EventLoopBuilder::new().with_any_thread(true).build()
            }
            else {
                EventLoopBuilder::new().build()
            }
        };

        let event_loop = if render_info.swapchain { Some(event_loop()) } else { None };
        let render = Render::new(render_info, &event_loop);

        Ok(Renderer { render,
                      event_loop,
                      time_since_start: std::time::Instant::now(),
                      requested_exit: false })
    }
}

#[pymodule]
fn reforge(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<Reforge>()?;
    m.add_class::<Renderer>()?;
    Ok(())
}
