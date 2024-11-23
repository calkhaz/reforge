use anyhow::{anyhow, Context, Result};
use crate::ffmpeg::{Decoder, Encoder};
use crate::render::Render;
use crate::render::RenderInfo;
use tracing::{debug, info, warn};
use crate::utils::TERM_CLEAR;
use crate::utils;
use crate::py;
use crate::Args;
use crate::config;
use winit::window::Window;

use std::collections::HashMap;

pub struct Reforge {
    args: Args,
    width: u32,
    height: u32,
    decoder: Option<Decoder>,
    encoder: Option<Encoder>,
    pub render: Render,
    pub cfg: py::PyConfig,
    py_config_timestamp: u64,
    time_since_start: std::time::Instant,
    first_run: Vec<bool>
}

impl Reforge {
    fn write_params(&mut self) {
        self.cfg.node_params.iter().for_each(|(node_name, params)| {
            params.iter().for_each(|(param_name, value)| {
                let param_map = self.render.pipeline_buffer_data.entry(node_name.clone()).or_default();
                param_map.insert(param_name.clone(), value.clone());

                self.render.outdate_frames();
            })
        });
    }

    fn py_needs_reload(&mut self) -> bool {
        if let Some(python_config) = self.args.python_config.as_ref() {
            let current_py_config_timestamp = utils::get_modified_time(&python_config);

            if current_py_config_timestamp == 0 {
                warn!("Unable to access python config: {}", python_config);
            }

            if current_py_config_timestamp > self.py_config_timestamp {
                self.py_config_timestamp = current_py_config_timestamp;
                return true;
            }
        }

        false
    }

    fn reload_py_config(&mut self) -> Result<()> {
        let Some(py_config_path) = self.args.python_config.as_ref() else { return Ok(()) };

        let shader_path = self.args.shader_path.as_ref().unwrap().clone();

        let config = py::py_config(&py_config_path).context(format!("Python config err in {py_config_path}"))?;

        if config.graph != self.cfg.graph {
            let graph_config = config::parse(config.graph.clone(), &shader_path).context("Failed to create config")?;
            self.render.update_config(graph_config);
            self.cfg.graph = config.graph;

            if let Some(ui) = self.render.ui.as_mut() {
                ui.params = config.ui_params;
            }

            debug!("Reload graph: {}", self.cfg.graph);
        }

        debug!("Reload Params: {:?}", self.cfg.node_params);

        self.write_params();

        Ok(())
    }

    pub fn resize_swapchain(&mut self, width: u32, height: u32) -> Result<()> {
        self.render.resize_swapchain(width, height)
    }

    pub fn new(args: Args, window: Option<&Window>) -> Result<Reforge> {
    
        let (mut width, mut height) = utils::get_dim(800, 600, args.width, args.height);
    
        let decoder = if let Some(input_file) = &args.input_file {
            let decoder = Decoder::new(input_file, args.width, args.height).context("Failed to create decoder")?;
            info!("Decoded size: {} X {}", decoder.width, decoder.height);
            width = decoder.width;
            height = decoder.height;
            Some(decoder)
        } else { None };
    
    
        let encoder = if let Some(output_file) = &args.output_file {
            Some(Encoder::new(&output_file, width, height).context("Failed to create encoder")?)
        } else { None };

        let (cfg, py_config_timestamp) = if let Some(python_config) = args.python_config.as_ref() {
            (py::py_config(&python_config)?, utils::get_modified_time(&python_config))
        }
        else {
            // Verified by this point in an earlier check
            let shader_file = args.shader_file.as_ref().unwrap();

            let graph = match decoder {
                Some(_) => format!("input -> {} -> output", shader_file),
                None    => format!(         "{} -> output", shader_file)
            };

            (py::PyConfig{graph, node_params: HashMap::new(), ui_params: HashMap::new()}, 0)
        };
    
        let shader_path = args.shader_path.clone().unwrap_or("".to_string());
        let graph_config = config::parse(cfg.graph.clone(), &shader_path).context("Failed to create config")?;
        debug!("Graph: {}", cfg.graph);
        debug!("Graph Config: {:?}", graph_config);

        // We write each frame to the encoder and that cannot
        // currently work in multi-frame mode
        let num_frames = if encoder.is_some() {
            1
        } else { args.num_frames.unwrap() };

        let render_info = RenderInfo {
            config: graph_config,
            width,
            height,
            num_frames,
            format: args.shader_format.unwrap().to_vk_format(),
            has_input_image: decoder.is_some(),
        };

        let mut render = Render::new(render_info, window)?;
        if let Some(ui) = &mut render.ui {
            ui.params = cfg.ui_params.clone();
        }

        let time_since_start: std::time::Instant = std::time::Instant::now();

        let first_run = vec![true; args.num_frames.unwrap()];

        Ok(Reforge { args, width, height, decoder, encoder, render, cfg, py_config_timestamp, time_since_start, first_run })
    }

    pub fn execute(&mut self, input_bytes: Option<&[u8]>, output_bytes: Option<&mut [u8]>) -> Result<()> {
        //let mut avg_ms = 0.0;
        let mapped_input_image_data: *mut u8 = self.render.staging_buffer_ptr();

        // TODO: We really don't want to always write into the staging buffer
        //       if the contents haven't changed - 'first_run[]' needs to work together with run() to
        //       optimize this away
        // Write bytes into staging image
        if let Some(input_bytes) = input_bytes {
            unsafe { std::ptr::copy_nonoverlapping(input_bytes.as_ptr(), mapped_input_image_data, input_bytes.len()); }
        }

        // Wait for the previous iteration of this frame before
        // changing or executing on its resources
        self.render.wait_for_frame_fence();

        if self.render.trigger_reloads()? {
            eprint!("{TERM_CLEAR}");
            self.first_run.iter_mut().for_each(|b| *b = true);
        }

        if let Err(err) = self.render.update_ubos(self.time_since_start.elapsed().as_secs_f32()) {
            warn!("{}", err);
        }
            // Clear current line of timers

        // Pull in the next image from the swapchain
        if self.render.has_swapchain() {
            self.render.acquire_swapchain()?;
        }

        self.render.begin_record();

        // On the first run, we:
        // 1. Transitioned images as needed
        // 2. Load the staging input buffer into an image and convert it to linear
        if self.first_run[self.render.frame_index] {
            self.write_params();
            if input_bytes.as_ref().is_some() {
                self.render.record_initial_image_load();
            }
            self.render.record_pipeline_image_transitions();
            self.first_run[self.render.frame_index] = false;
        }

        self.render.record();

        if let Some(ui) = self.render.ui.as_ref() {
            if !ui.hidden {
                self.render.record_ui();
            }
        }

        if output_bytes.is_some() {
            self.render.write_output_to_buffer();
        }

        self.render.end_record();

        // Send the work to the gpu
        self.render.submit();

        if self.encoder.is_some() {
            self.render.wait_for_frame_fence();
    
            if let Some(output_bytes) = output_bytes {
                unsafe { std::ptr::copy_nonoverlapping(mapped_input_image_data, output_bytes.as_mut_ptr(), output_bytes.len()); }
            }
        }

        Ok(())
    }

    pub fn run(&mut self) -> Result<bool> {
        let frame_size = self.width as usize * self.height as usize * 4;

        let mut rf_output: Vec<u8> = Vec::with_capacity(frame_size);
        unsafe { rf_output.set_len(frame_size) }

        let (frame, is_last_frame) = if let Some(decoder) = self.decoder.as_mut() {
            let (frame, is_last_frame) = decoder.read_frame()?;

            // For videos, we will always be on our "first run"
            // where we need to load images into a buffer and transition them
            if decoder.num_frames > 1 {
                self.first_run.iter_mut().for_each(|b| *b = true);
            }
            (Some(frame), is_last_frame)
        }
        else { (None, false) };

        if self.py_needs_reload() {
            if let Err(err) = self.reload_py_config() {
                warn!("{}", err);
            }
        }

        if is_last_frame && self.encoder.is_some() {
            return Ok(true);
        }
        else if is_last_frame && self.decoder.as_ref().unwrap().num_frames > 1 {
            self.decoder = Some(Decoder::new(&self.args.input_file.as_ref().unwrap(), self.args.width, self.args.height).context("Failed to create decoder")?);
        }

        // Render to swapchain or file
        if self.encoder.is_some() {
            self.execute(frame.as_deref(), Some(&mut rf_output))?;

            // Encode to file
            self.encoder.as_mut().unwrap().write_frame(&rf_output)?;

            // Only need to write one frame and exit
            if !self.encoder.as_ref().unwrap().is_multi_frame {
                return Ok(true);
            }
        }
        else {
            self.execute(frame.as_deref(), None)?;
        }
    
        Ok(false)
    }
}
