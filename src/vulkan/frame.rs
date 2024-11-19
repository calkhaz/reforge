
extern crate ash;

use anyhow::Result;
use ash::vk;
use crate::vulkan::core::VkCore;
use crate::vulkan::vkutils::GpuTimer;

use std::rc::Rc;

pub struct Frame {
    device: Rc<ash::Device>,
    pub fence: vk::Fence,
    pub present_complete_semaphore: vk::Semaphore,
    pub render_complete_semaphore: vk::Semaphore,
    pub cmd_pool: vk::CommandPool,
    pub cmd_buffer: vk::CommandBuffer,
    pub timer: GpuTimer,
}

impl Frame {
    unsafe fn create_commands(device: &ash::Device, queue_family_index: u32) -> Result<(vk::CommandPool, vk::CommandBuffer)> {
        let pool_create_info = vk::CommandPoolCreateInfo::default()
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
            .queue_family_index(queue_family_index);

        let pool = device.create_command_pool(&pool_create_info, None)?;

        let command_buffer_allocate_info = vk::CommandBufferAllocateInfo::default()
            .command_buffer_count(1)
            .command_pool(pool)
            .level(vk::CommandBufferLevel::PRIMARY);

        let command_buffer = device
            .allocate_command_buffers(&command_buffer_allocate_info)?;

        Ok((pool, command_buffer[0]))
    }

    pub fn rebuild_timer(&mut self, query_buffer_size: u32) {
        self.timer = GpuTimer::new(Rc::clone(&self.device), query_buffer_size);
    }

    pub fn new(core: &VkCore, query_buffer_size: u32) -> Result<Frame> {
        let semaphore_create_info = vk::SemaphoreCreateInfo::default();
        let fence_create_info =
            vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);

        unsafe {
            let (cmd_pool, cmd_buff) = Self::create_commands(&core.device, core.queue_family_index)?;

            Ok(Frame {
                device: Rc::clone(&core.device),
                fence: core.device.create_fence(&fence_create_info, None).expect("Create fence failed."),
                present_complete_semaphore: core.device.create_semaphore(&semaphore_create_info, None)?,
                render_complete_semaphore: core.device.create_semaphore(&semaphore_create_info, None)?,
                cmd_pool,
                cmd_buffer: cmd_buff,
                timer: GpuTimer::new(Rc::clone(&core.device), query_buffer_size),
            })
        }
    }
}

impl Drop for Frame {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_fence(self.fence, None);
            self.device.destroy_semaphore(self.render_complete_semaphore, None);
            self.device.destroy_semaphore(self.present_complete_semaphore, None);
            self.device.destroy_command_pool(self.cmd_pool, None);
        }
    }
}
