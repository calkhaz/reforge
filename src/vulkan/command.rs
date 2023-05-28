extern crate ash;

use ash::vk;

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

pub struct BlitCopy {
    pub width: u32,
    pub height: u32,
    pub src_image: vk::Image,
    pub dst_image: vk::Image,
    pub src_layout: vk::ImageLayout,
    pub dst_layout: vk::ImageLayout
}

pub fn blit_copy(device: &ash::Device, cmd: vk::CommandBuffer, info: &BlitCopy) {
        let copy_subresource = vk::ImageSubresourceLayers {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            mip_level: 0,
            base_array_layer: 0,
            layer_count: 1
        };

        let begin_offset = vk::Offset3D {
            x: 0, y: 0, z: 0
        };
        let end_offset = vk::Offset3D {
            x: info.width as i32, y: info.height as i32, z: 1
        };

        let blit = vk::ImageBlit {
            src_subresource: copy_subresource,
            src_offsets: [begin_offset, end_offset],
            dst_subresource: copy_subresource,
            dst_offsets: [begin_offset, end_offset]
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
