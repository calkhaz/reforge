extern crate ash;
extern crate shaderc;
extern crate gpu_allocator;

use ash::vk;
use anyhow::Result;
use std::default::Default;
use crate::vulkan::core::VkCore;

pub unsafe fn build_framebuffer(core: &VkCore, image_view: vk::ImageView, render_pass: vk::RenderPass, width: u32, height: u32) -> Result<vk::Framebuffer> {
    let info = vk::FramebufferCreateInfo {
        render_pass,
        attachment_count: 1,
        p_attachments: &image_view,
        width,
        height,
        layers: 1,
        ..Default::default()
    };

    Ok(core.device.create_framebuffer(&info, None)?)
}

pub unsafe fn build_render_pass(device: &ash::Device, format: vk::Format, load_op: vk::AttachmentLoadOp, initial_layout: vk::ImageLayout, final_layout: vk::ImageLayout) -> Result<vk::RenderPass> {
    let color_attachment = vk::AttachmentReference {
        attachment: 0,
        layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL
    };

    let attachment_desc = vk::AttachmentDescription {
        format,
        samples: vk::SampleCountFlags::TYPE_1,
        load_op,
        store_op: vk::AttachmentStoreOp::STORE,
        initial_layout,
        final_layout,
        ..Default::default()
    };

    let subpass_desc = vk::SubpassDescription {
        pipeline_bind_point: vk::PipelineBindPoint::GRAPHICS,
        color_attachment_count: 1,
        p_color_attachments: &color_attachment,
        ..Default::default()
    };

    let info = vk::RenderPassCreateInfo {
        attachment_count: 1,
        p_attachments: &attachment_desc,
        subpass_count: 1,
        p_subpasses: &subpass_desc,
        ..Default::default()
    };

    Ok(device.create_render_pass(&info, None)?)
}

