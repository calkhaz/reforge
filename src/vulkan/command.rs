extern crate ash;
use ash::vk;

use crate::vulkan::vkutils::Image;
use crate::vulkan::frame::Frame;
use crate::vulkan::pipeline_graph::GraphAction;
use crate::vulkan::pipeline_graph::PipelineGraph;
use crate::vulkan::pipeline_graph::PipelineGraphFrame;

pub fn transition_image_layout(device: &ash::Device, cmd: vk::CommandBuffer, image: vk::Image, src_layout: vk::ImageLayout, dst_layout: vk::ImageLayout) {

    let mut src_access = vk::AccessFlags::empty();
    let mut dst_access = vk::AccessFlags::empty();
    let mut src_pipeline = vk::PipelineStageFlags::TOP_OF_PIPE;
    let mut dst_pipeline = vk::PipelineStageFlags::TOP_OF_PIPE;

    match src_layout {
        vk::ImageLayout::UNDEFINED => {}
        vk::ImageLayout::GENERAL => {
            dst_access = vk::AccessFlags::SHADER_READ;
            dst_pipeline = vk::PipelineStageFlags::COMPUTE_SHADER
        }
        vk::ImageLayout::TRANSFER_SRC_OPTIMAL => {
            src_access = vk::AccessFlags::TRANSFER_READ;
            src_pipeline = vk::PipelineStageFlags::TRANSFER
        }
        vk::ImageLayout::TRANSFER_DST_OPTIMAL => {
            src_access = vk::AccessFlags::TRANSFER_WRITE;
            src_pipeline = vk::PipelineStageFlags::TRANSFER
        }
        _ => panic!("No matching result for src_layout: {:?}", src_layout)
    }

    match dst_layout {
        vk::ImageLayout::UNDEFINED => {}
        vk::ImageLayout::GENERAL => {
            dst_access = vk::AccessFlags::SHADER_WRITE;
            dst_pipeline = vk::PipelineStageFlags::COMPUTE_SHADER
        }
        vk::ImageLayout::TRANSFER_DST_OPTIMAL => {
            dst_access = vk::AccessFlags::TRANSFER_WRITE;
            dst_pipeline = vk::PipelineStageFlags::TRANSFER
        }
        vk::ImageLayout::TRANSFER_SRC_OPTIMAL => {
            dst_access = vk::AccessFlags::TRANSFER_READ;
            dst_pipeline = vk::PipelineStageFlags::TRANSFER
        }
        vk::ImageLayout::PRESENT_SRC_KHR => {
            dst_pipeline = vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT;
            dst_access = vk::AccessFlags::COLOR_ATTACHMENT_WRITE
        }
        _ => panic!("No matching result for dst_layout: {:?}", dst_layout)
    }

    let image_barrier = vk::ImageMemoryBarrier {
        src_access_mask: src_access,
        dst_access_mask: dst_access,
        old_layout: src_layout,
        new_layout: dst_layout,
        //image: res.swapchain.images[present_index as usize],
        image,
        subresource_range: vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            level_count: 1,
            layer_count: 1,
            ..Default::default()
                },
        ..Default::default()
    };

    unsafe {
    device.cmd_pipeline_barrier(cmd, src_pipeline, dst_pipeline, vk::DependencyFlags::empty(), &[], &[], &[image_barrier]);
    }

}

#[derive(Default)]
pub struct BlitCopy {
    pub src_width: u32,
    pub src_height: u32,
    pub dst_width: u32,
    pub dst_height: u32,
    pub src_image: vk::Image,
    pub dst_image: vk::Image,
    pub src_layout: vk::ImageLayout,
    pub dst_layout: vk::ImageLayout,
    pub center: bool
}

pub struct ImageToBuffer<'a> {
    pub width: u32,
    pub height: u32,
    pub src_image: &'a Image,
    pub dst_buffer: vk::Buffer,
}

pub fn blit_copy(device: &ash::Device, cmd: vk::CommandBuffer, info: &BlitCopy) {
        let copy_subresource = vk::ImageSubresourceLayers {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            mip_level: 0,
            base_array_layer: 0,
            layer_count: 1
        };

        let src_begin_offset = vk::Offset3D { x: 0, y: 0, z: 0 };
        let src_end_offset = vk::Offset3D { x: info.src_width as i32, y: info.src_height as i32, z: 1 };
        let mut dst_begin_offset = vk::Offset3D { x: 0, y: 0, z: 0 };
        let mut dst_end_offset = vk::Offset3D { x: info.dst_width as i32, y: info.dst_height as i32, z: 1 };

        // Center the src image into the dst image
        if info.center {
            let calc_dst_offset = |src_dim: i32, dst_dim: i32| -> (i32, i32) {
                let diff  = std::cmp::max(dst_dim - src_dim, 0);
                let padding = diff/2;
                let begin_offset = padding;
                let end_offset = dst_dim - padding;

                (begin_offset, end_offset)
            };

            (dst_begin_offset.x, dst_end_offset.x) = calc_dst_offset(info.src_width as i32, info.dst_width as i32);
            (dst_begin_offset.y, dst_end_offset.y) = calc_dst_offset(info.src_height as i32, info.dst_height as i32);
        }

        let blit = vk::ImageBlit {
            src_subresource: copy_subresource,
            src_offsets: [src_begin_offset, src_end_offset],
            dst_subresource: copy_subresource,
            dst_offsets: [dst_begin_offset, dst_end_offset]
        };

        unsafe {
        device.cmd_blit_image(cmd,
                              info.src_image,
                              info.src_layout,
                              info.dst_image,
                              info.dst_layout,
                              &[blit],
                              vk::Filter::LINEAR);
        }
}

pub fn copy_image_to_buffer(device: &ash::Device, cmd: vk::CommandBuffer, info: &ImageToBuffer) {
    let copy_subresource = vk::ImageSubresourceLayers {
        aspect_mask: vk::ImageAspectFlags::COLOR,
        mip_level: 0,
        base_array_layer: 0,
        layer_count: 1
    };

    let region = vk::BufferImageCopy {
        buffer_offset: 0,
        buffer_row_length: info.width,
        buffer_image_height: info.height,
        image_subresource: copy_subresource,
        image_offset: vk::Offset3D{x: 0, y: 0, z: 0},
        image_extent: vk::Extent3D{width: info.width, height: info.height, depth: 1}
    };

    unsafe {
    device.cmd_copy_image_to_buffer(cmd, info.src_image.vk, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, info.dst_buffer, &[region]);
    }
}


pub fn execute_pipeline_graph(device: &ash::Device, frame: &mut Frame, graph_frame: &PipelineGraphFrame, graph: &PipelineGraph) {
    let dispatch_x = (graph.width as f32/16.0).ceil() as u32;
    let dispatch_y = (graph.height as f32/16.0).ceil() as u32;

    for action in &graph.flattened {
        match action {
            GraphAction::Barrier => {
                let mem_barrier = vk::MemoryBarrier {
                    src_access_mask: vk::AccessFlags::SHADER_READ,
                    dst_access_mask: vk::AccessFlags::SHADER_WRITE,
                    ..Default::default()
                };

                unsafe {
                device.cmd_pipeline_barrier(frame.cmd_buffer,
                                            vk::PipelineStageFlags::COMPUTE_SHADER,
                                            vk::PipelineStageFlags::COMPUTE_SHADER,
                                            vk::DependencyFlags::empty(), &[mem_barrier], &[], &[]);
                }
            },
            GraphAction::Pipeline(pipeline) => {
                unsafe {
                device.cmd_bind_descriptor_sets(
                    frame.cmd_buffer,
                    vk::PipelineBindPoint::COMPUTE,
                    pipeline.borrow().layout.vk,
                    0,
                    &[*graph_frame.descriptor_sets.get(&pipeline.borrow().name).unwrap()],
                    &[],
                );
                device.cmd_bind_pipeline(
                    frame.cmd_buffer,
                    vk::PipelineBindPoint::COMPUTE,
                    pipeline.borrow().vk_pipeline
                );

                // Start timer
                device.cmd_write_timestamp(frame.cmd_buffer,
                                           vk::PipelineStageFlags::TOP_OF_PIPE,
                                           frame.timer.query_pool,
                                           frame.timer.start(&pipeline.borrow().name));

                device.cmd_dispatch(frame.cmd_buffer, dispatch_x, dispatch_y, 1);

                // Stop timer
                device.cmd_write_timestamp(frame.cmd_buffer,
                                           vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                                           frame.timer.query_pool,
                                           frame.timer.stop(&pipeline.borrow().name));
                    }
            }
        }
    }
}
