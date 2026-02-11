use std::any::Any;
use std::sync::mpsc;

use wgpu::util::DeviceExt;

use crate::prelude::Workgroup;
use crate::vbuffer::VBuffer;
use crate::workgroup::VBufferHandle;

#[derive(Copy, Clone)]
pub enum PartitionMode {
    // Buffer is unmanaged across a workgroup. The entire buffer is given to each device.
    Unmanaged,
}

pub struct Task<'t> {
    pub(crate) workgroup: &'t mut Workgroup,

    pub(crate) output_buffers: Vec<(u32, VBufferHandle, PartitionMode)>,

    pub(crate) staging_buffers: Vec<Vec<wgpu::Buffer>>,
    pub(crate) command_buffers: Vec<wgpu::CommandBuffer>,
}

impl<'t> Task<'t> {
    pub(crate) fn from_builder(builder: TaskBuilder<'t>) -> Option<Self> {
        let TaskBuilder {
            workgroup,
            shader,
            kernel,
            size,
            input_buffers,
            output_buffers,
        } = builder;

        let kernel = kernel?;
        let size = size?;

        let num_devices = workgroup.vdevices.len();

        let mut buffers: Vec<Vec<wgpu::Buffer>> = vec![vec![]; num_devices];
        let mut layouts: Vec<Vec<wgpu::BindGroupLayoutEntry>> = vec![vec![]; num_devices];
        let mut staging_buffers: Vec<Vec<wgpu::Buffer>> = vec![vec![]; num_devices];
        let mut output_wgpu_buffers: Vec<Vec<wgpu::Buffer>> = vec![vec![]; num_devices];

        for (id, key, mode) in &input_buffers {
            let vbuffer = workgroup.vbuffers.get(*key)?;

            for (vdi, vd) in workgroup.vdevices.iter().enumerate() {
                let byte_slice: &[u8] = match mode {
                    PartitionMode::Unmanaged => vbuffer_bytes(vbuffer),
                };

                let label = format!("WISC Input Buffer {} (VDevice {})", id, vd.label);

                let wgpu_buffer = vd
                    .device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some(&label),
                        contents: byte_slice,
                        usage: wgpu::BufferUsages::STORAGE,
                    });

                let layout_entry = wgpu::BindGroupLayoutEntry {
                    binding: *id,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                };

                buffers[vdi].push(wgpu_buffer);
                layouts[vdi].push(layout_entry);
            }
        }

        for (id, key, mode) in &output_buffers {
            let vbuffer = workgroup.vbuffers.get(*key)?;

            for (vdi, vd) in workgroup.vdevices.iter().enumerate() {
                let mappable_primary = vd
                    .features
                    .contains(wgpu::Features::MAPPABLE_PRIMARY_BUFFERS);

                let byte_slice: &[u8] = match mode {
                    PartitionMode::Unmanaged => vbuffer_bytes(vbuffer),
                };

                let label = format!("WISC Output Buffer {} (VDevice {})", id, vd.label);

                let wgpu_buffer = vd
                    .device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some(&label),
                        contents: byte_slice,
                        usage: wgpu::BufferUsages::STORAGE
                            | wgpu::BufferUsages::COPY_SRC
                            | if mappable_primary {
                                wgpu::BufferUsages::MAP_READ
                            } else {
                                wgpu::BufferUsages::empty()
                            },
                    });

                let layout_entry = wgpu::BindGroupLayoutEntry {
                    binding: *id,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                };

                let staging_buffer = if mappable_primary {
                    wgpu_buffer.clone()
                } else {
                    let byte_len = vbuffer.length * vbuffer.stride;
                    vd.device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some(&format!(
                            "WISC Staging Buffer {} (VDevice {})",
                            id, vd.label
                        )),
                        size: byte_len as wgpu::BufferAddress,
                        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    })
                };

                buffers[vdi].push(wgpu_buffer.clone());
                layouts[vdi].push(layout_entry);
                output_wgpu_buffers[vdi].push(wgpu_buffer);
                staging_buffers[vdi].push(staging_buffer);
            }
        }

        let mut command_buffers: Vec<wgpu::CommandBuffer> = Vec::with_capacity(num_devices);

        for (vdi, vd) in workgroup.vdevices.iter().enumerate() {
            let bind_group_layout =
                vd.device
                    .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                        label: None,
                        entries: &layouts[vdi],
                    });

            let bind_group_entries: Vec<wgpu::BindGroupEntry> = layouts[vdi]
                .iter()
                .zip(buffers[vdi].iter())
                .map(|(entry, buffer)| wgpu::BindGroupEntry {
                    binding: entry.binding,
                    resource: buffer.as_entire_binding(),
                })
                .collect();

            let bind_group = vd.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &bind_group_layout,
                entries: &bind_group_entries,
            });

            let pipeline_layout =
                vd.device
                    .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: None,
                        bind_group_layouts: &[&bind_group_layout],
                        immediate_size: 0,
                    });

            let pipeline = vd
                .device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: None,
                    layout: Some(&pipeline_layout),
                    module: &vd.device.create_shader_module(shader.clone()),
                    entry_point: Some(kernel.as_str()),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    cache: None,
                });

            let mut encoder = vd
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

            {
                let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: None,
                    timestamp_writes: None,
                });

                compute_pass.set_pipeline(&pipeline);
                compute_pass.set_bind_group(0, &bind_group, &[]);

                let (x, y, z) = size;
                compute_pass.dispatch_workgroups(x, y, z);
            }

            let mappable_primary = vd
                .features
                .contains(wgpu::Features::MAPPABLE_PRIMARY_BUFFERS);

            if !mappable_primary {
                for (output_buffer, staging_buffer) in output_wgpu_buffers[vdi]
                    .iter()
                    .zip(staging_buffers[vdi].iter())
                {
                    encoder.copy_buffer_to_buffer(
                        output_buffer,
                        0,
                        staging_buffer,
                        0,
                        output_buffer.size(),
                    );
                }
            }

            command_buffers.push(encoder.finish());
        }

        Some(Task {
            workgroup,

            output_buffers,

            staging_buffers,
            command_buffers,
        })
    }

    pub fn run(self) {
        for (device, command_buffer) in self
            .workgroup
            .vdevices
            .iter()
            .zip(self.command_buffers.into_iter())
        {
            device.queue.submit([command_buffer]);
        }

        let mut receivers = Vec::new();

        for (device_id, _device) in self.workgroup.vdevices.iter().enumerate() {
            for staging_buffer in self.staging_buffers[device_id].iter() {
                let buffer_slice = staging_buffer.slice(..);
                let (tx, rx) = mpsc::channel();
                buffer_slice.map_async(wgpu::MapMode::Read, move |_| {
                    let _ = tx.send(());
                });
                receivers.push(rx);
            }
        }

        for device in self.workgroup.vdevices.iter() {
            device
                .device
                .poll(wgpu::PollType::wait_indefinitely())
                .unwrap();
        }

        for rx in receivers {
            let _ = rx.recv();
        }

        for (device_id, _device) in self.workgroup.vdevices.iter().enumerate() {
            for (output_index, staging_buffer) in self.staging_buffers[device_id].iter().enumerate()
            {
                let Some((_, handle, mode)) = self.output_buffers.get(output_index) else {
                    continue;
                };
                let buffer_slice = staging_buffer.slice(..);
                let data = buffer_slice.get_mapped_range();
                let bytes: &[u8] = &data;

                match mode {
                    PartitionMode::Unmanaged => {
                        if let Some(vbuffer) = self.workgroup.vbuffers.get_mut(*handle) {
                            let dst = vbuffer_bytes_mut(vbuffer);
                            let copy_len = dst.len().min(bytes.len());
                            dst[..copy_len].copy_from_slice(&bytes[..copy_len]);
                        }
                    }
                }

                drop(data);
                staging_buffer.unmap();
            }
        }
    }
}

pub struct TaskBuilder<'b> {
    pub(crate) workgroup: &'b mut Workgroup,
    pub(crate) shader: wgpu::ShaderModuleDescriptor<'b>,
    pub(crate) kernel: Option<String>,
    pub(crate) size: Option<(u32, u32, u32)>,

    pub(crate) input_buffers: Vec<(u32, VBufferHandle, PartitionMode)>,
    pub(crate) output_buffers: Vec<(u32, VBufferHandle, PartitionMode)>,
}

impl<'b> TaskBuilder<'b> {
    pub fn new(workgroup: &'b mut Workgroup, shader: wgpu::ShaderModuleDescriptor<'b>) -> Self {
        Self {
            workgroup,
            shader,
            kernel: None,
            size: None,

            input_buffers: vec![],
            output_buffers: vec![],
        }
    }

    pub fn build(self) -> Option<Task<'b>> {
        Task::from_builder(self)
    }

    pub fn with_kernel<S: Into<String>>(mut self, id: S) -> Self {
        self.kernel.replace(id.into());

        self
    }

    pub fn with_size(mut self, size: (u32, u32, u32)) -> Self {
        assert!(size.0 > 0, "Workgroup size must be greater than zero.");
        assert!(size.1 > 0, "Workgroup size must be greater than zero.");
        assert!(size.2 > 0, "Workgroup size must be greater than zero.");

        self.size.replace(size);

        self
    }

    pub fn with_input_buffer(
        mut self,
        id: u32,
        handle: VBufferHandle,
        partition_mode: PartitionMode,
    ) -> Self {
        self.input_buffers.push((id, handle, partition_mode));

        self
    }

    pub fn with_output_buffer(
        mut self,
        id: u32,
        handle: VBufferHandle,
        partition_mode: PartitionMode,
    ) -> Self {
        self.output_buffers.push((id, handle, partition_mode));

        self
    }
}

fn vbuffer_bytes(vbuffer: &VBuffer) -> &[u8] {
    let byte_length = vbuffer.length * vbuffer.stride;

    unsafe {
        let vec = &*(vbuffer.inner.as_ref() as *const dyn Any as *const Vec<u8>);
        let data_ptr = vec.as_ptr();
        std::slice::from_raw_parts(data_ptr, byte_length)
    }
}

fn vbuffer_bytes_mut(vbuffer: &mut VBuffer) -> &mut [u8] {
    let byte_length = vbuffer.length * vbuffer.stride;

    unsafe {
        let vec = &mut *(vbuffer.inner.as_mut() as *mut dyn Any as *mut Vec<u8>);
        let data_ptr = vec.as_mut_ptr();
        std::slice::from_raw_parts_mut(data_ptr, byte_length)
    }
}
