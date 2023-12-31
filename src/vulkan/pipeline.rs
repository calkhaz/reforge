extern crate ash;
extern crate shaderc;
extern crate gpu_allocator;

use ash::vk;
use spirv_reflect::types::ReflectDescriptorBinding;
use std::ffi::CStr;
use std::default::Default;
use std::collections::HashMap;
use std::rc::Rc;
use std::cell::RefCell;
use std::ops::Drop;

use crate::vulkan::vkutils;
use crate::vulkan::shader::Shader;

#[derive(Clone)]
pub struct PipelineInfo {
    pub name: String,
    pub shader: Rc<RefCell<Shader>>,
    pub input_images:  Vec<(String, ReflectDescriptorBinding)>,
    pub output_images: Vec<(String, ReflectDescriptorBinding)>,
    pub input_ssbos:   Vec<(String, ReflectDescriptorBinding)>,
    pub output_ssbos:  Vec<(String, ReflectDescriptorBinding)>,
}

pub struct Pipeline {
    pub device: Rc<ash::Device>,
    pub name: String,
    pub info: PipelineInfo,
    pub layout: PipelineLayout,
    pub vk_pipeline: ash::vk::Pipeline,

    // In the case of a graphics pipeline
    pub vertex_shader: Option<Rc<Shader>>,
    pub render_pass: Option<vk::RenderPass>
}

pub struct PipelineLayout {
    pub vk: vk::PipelineLayout,
    pub descriptor_layout: vk::DescriptorSetLayout
}

impl Pipeline {
    pub unsafe fn build_compute_vk(device: &Rc<ash::Device>,
                                   shader: &Shader,
                                   pipeline_layout: &PipelineLayout) -> Result<vk::Pipeline, String> {

        let shader_entry_name = CStr::from_bytes_with_nul_unchecked(b"main\0");
        let shader_stage_create_infos = vk::PipelineShaderStageCreateInfo {
            module: shader.module,
            p_name: shader_entry_name.as_ptr(),
            stage: vk::ShaderStageFlags::COMPUTE,
            ..Default::default()
        };

        let vk_info = vk::ComputePipelineCreateInfo {
            layout: pipeline_layout.vk,
            stage: shader_stage_create_infos,
            ..Default::default()
        };

        match device.create_compute_pipelines(vk::PipelineCache::null(), &[vk_info], None) {
            Ok(pipelines) => {
                Ok(pipelines[0])
            },
            Err(err) => {
                Err(format!("Failed to create graphics pipeline: {:?}", err))
            }
        }
    }

    pub unsafe fn new_compute(device: Rc<ash::Device>,
                              info: PipelineInfo,
                              pipeline_layout: PipelineLayout) -> Result<Pipeline, String> {

        let vk_pipeline = Self::build_compute_vk(&device, &info.shader.borrow(), &pipeline_layout)?;

        Ok(Pipeline {
            device,
            name: info.name.clone(),
            info,
            layout: pipeline_layout,
            vk_pipeline,
            render_pass: None,
            vertex_shader: None
        })
    }

    pub unsafe fn build_gfx_vk(device: &Rc<ash::Device>,
                               width: u32,
                               height: u32,
                               vertex_shader: &Rc<Shader>,
                               fragment_shader: &Shader,
                               pipeline_layout: &PipelineLayout,
                               render_pass: vk::RenderPass) -> Result<vk::Pipeline, String> {
        let shader_entry_name = CStr::from_bytes_with_nul_unchecked(b"main\0");

        let shader_stages = vec![
            vk::PipelineShaderStageCreateInfo {
                stage: vk::ShaderStageFlags::VERTEX,
                module: vertex_shader.module,
                p_name: shader_entry_name.as_ptr(),
                ..Default::default()
            },
            vk::PipelineShaderStageCreateInfo {
                stage: vk::ShaderStageFlags::FRAGMENT,
                module: fragment_shader.module,
                p_name: shader_entry_name.as_ptr(),
                ..Default::default()
            }
        ];

        let rasterization_state = vk::PipelineRasterizationStateCreateInfo {
            depth_clamp_enable: vk::FALSE,
            rasterizer_discard_enable: vk::FALSE,
            polygon_mode: vk::PolygonMode::FILL,
            cull_mode: vk::CullModeFlags::NONE,
            front_face: vk::FrontFace::COUNTER_CLOCKWISE,
            depth_bias_enable: vk::FALSE,
            line_width: 1.0,
            ..Default::default()
        };

        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo {
            topology: vk::PrimitiveTopology::TRIANGLE_LIST,
            ..Default::default()
        };

        let vertex_input_state = vk::PipelineVertexInputStateCreateInfo {
            ..Default::default()
        };

        let scissors = vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: vk::Extent2D { width, height}
        };

        let viewport = vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: width as f32,
            height: height as f32,
            max_depth: 1.0,
            ..Default::default()
        };

        let viewport_state = vk::PipelineViewportStateCreateInfo {
            viewport_count: 1,
            scissor_count: 1,
            p_viewports: &viewport,
            p_scissors: &scissors,
            ..Default::default()
        };

        let blend_attachment = vk::PipelineColorBlendAttachmentState {
            blend_enable: vk::FALSE,
            color_write_mask: vk::ColorComponentFlags::RGBA,
            ..Default::default()
        };

        let blend_state = vk::PipelineColorBlendStateCreateInfo {
            logic_op_enable: vk::FALSE,
            attachment_count: 1,
            p_attachments: &blend_attachment,
            ..Default::default()
        };

        let vk_info = vk::GraphicsPipelineCreateInfo {
            render_pass,
            layout: pipeline_layout.vk,
            stage_count: 2,
            p_stages: shader_stages.as_ptr(),
            subpass: 0,
            p_input_assembly_state: &input_assembly,
            p_rasterization_state: &rasterization_state,
            p_viewport_state: &viewport_state,
            p_color_blend_state: &blend_state,
            p_vertex_input_state: &vertex_input_state,
            ..Default::default()
        };

        match device.create_graphics_pipelines(vk::PipelineCache::null(), &[vk_info], None) {
            Ok(pipelines) => {
                Ok(pipelines[0])
            },
            Err(err) => {
                Err(format!("Failed to create graphics pipeline: {:?}", err))
            }
        }
    }

    pub unsafe fn new_gfx(device: Rc<ash::Device>,
                          width: u32,
                          height: u32,
                          vertex_shader: Rc<Shader>,
                          info: PipelineInfo,
                          pipeline_layout: PipelineLayout,
                          render_pass: vk::RenderPass) -> Result<Pipeline, String> {

        let vk_pipeline = Self::build_gfx_vk(
            &device, width, height, &vertex_shader, &info.shader.borrow(), &pipeline_layout, render_pass
        )?;

        Ok(Pipeline {
            device,
            name: info.name.clone(),
            info,
            layout: pipeline_layout,
            vk_pipeline,
            render_pass: Some(render_pass),
            vertex_shader: Some(vertex_shader)
        })
    }

    pub unsafe fn new_layout(device: Rc<ash::Device>,
                             info: &PipelineInfo,
                             pool_sizes: &mut HashMap<vk::DescriptorType, u32>,
                             num_frames: usize) -> PipelineLayout {
        // create descriptor layouts, add descriptor pool sizes, and add pipelines to hashmap
        let layout_bindings = vkutils::create_descriptor_layout_bindings(&info.shader.borrow().bindings, num_frames, pool_sizes);

        let descriptor_info = vk::DescriptorSetLayoutCreateInfo::builder().bindings(&layout_bindings);
        let descriptor_layout = [device
            .create_descriptor_set_layout(&descriptor_info, None)
            .unwrap()];

        let pipeline_layout = device.
            create_pipeline_layout(&vk::PipelineLayoutCreateInfo::builder()
                .set_layouts(&descriptor_layout), None).unwrap();

        PipelineLayout {
            vk: pipeline_layout,
            descriptor_layout: descriptor_layout[0]
        }
    }

    pub unsafe fn rebuild(&mut self, width: u32, height:u32, shader: Shader) -> Result<(), String> {
        let vk_pipeline = if self.vertex_shader.is_some() {
            Self::build_gfx_vk(
                &self.device, width, height, &self.vertex_shader.as_ref().unwrap(),
                &shader, &self.layout, self.render_pass.unwrap()
            )
        }
        else {
            Self::build_compute_vk(&self.device, &shader, &self.layout)
        }?;

        self.destroy(false);
        self.info.shader.replace(shader);
        self.vk_pipeline = vk_pipeline;
        Ok(())
    }

    pub unsafe fn destroy(&mut self, destroy_non_resizables: bool) {
        let device = &self.device;
    
        device.device_wait_idle().unwrap();
        device.destroy_pipeline(self.vk_pipeline, None);
        if destroy_non_resizables {
            if let Some(render_pass) = self.render_pass {
                 device.destroy_render_pass(render_pass, None)
            }
            device.destroy_pipeline_layout(self.layout.vk, None);
            device.destroy_descriptor_set_layout(self.layout.descriptor_layout, None);
        }
    }
}


impl Drop for Pipeline {
    fn drop(&mut self) {
        unsafe { self.destroy(true); }
    }
}

