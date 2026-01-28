mod util;

use util::SlicePointerWriter;

use futures_lite::future;
use wgpu::{self, util::DeviceExt};

pub use wgpu::Features;
pub use wgpu::include_wgsl;

pub struct Device {
    info: wgpu::AdapterInfo,
    _limits: wgpu::Limits,
    features: wgpu::Features,
    device: wgpu::Device,
    queue: wgpu::Queue,
}

impl Device {
    pub fn best() -> Option<Self> {
        Self::best_with_features(wgpu::Features::empty())
    }

    pub fn all() -> Vec<Self> {
        Self::all_with_features(wgpu::Features::empty())
    }

    pub fn best_with_features(features: wgpu::Features) -> Option<Self> {
        future::block_on(async {
            let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());

            let adapter = instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    ..Default::default()
                })
                .await
                .ok()?;

            let downlevel_capabilities = adapter.get_downlevel_capabilities();
            if !downlevel_capabilities
                .flags
                .contains(wgpu::DownlevelFlags::COMPUTE_SHADERS)
            {
                return None;
            }

            let (device, queue) = adapter
                .request_device(&wgpu::DeviceDescriptor {
                    label: None,
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::downlevel_defaults(),
                    memory_hints: wgpu::MemoryHints::Performance,
                    trace: wgpu::Trace::Off,
                    experimental_features: wgpu::ExperimentalFeatures::disabled(),
                })
                .await
                .ok()?;

            Some(Self {
                info: adapter.get_info(),
                _limits: adapter.limits(),
                features: device.features().intersection(features),
                device,
                queue,
            })
        })
    }

    pub fn all_with_features(features: wgpu::Features) -> Vec<Self> {
        future::block_on(async {
            let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());

            let adapters = instance.enumerate_adapters(wgpu::Backends::PRIMARY).await;

            let mut result: Vec<Self> = Vec::with_capacity(adapters.len());

            for adapter in adapters {
                let downlevel_capabilities = adapter.get_downlevel_capabilities();
                if !downlevel_capabilities
                    .flags
                    .contains(wgpu::DownlevelFlags::COMPUTE_SHADERS)
                {
                    continue;
                };

                if let Ok((device, queue)) = adapter
                    .request_device(&wgpu::DeviceDescriptor {
                        label: None,
                        required_features: wgpu::Features::empty(),
                        required_limits: wgpu::Limits::downlevel_defaults(),
                        memory_hints: wgpu::MemoryHints::Performance,
                        trace: wgpu::Trace::Off,
                        experimental_features: wgpu::ExperimentalFeatures::disabled(),
                    })
                    .await
                {
                    result.push(Self {
                        info: adapter.get_info(),
                        _limits: adapter.limits(),
                        features: device.features().intersection(features),
                        device,
                        queue,
                    });
                }
            }

            result
        })
    }

    pub fn info(&self) -> wgpu::AdapterInfo {
        self.info.clone()
    }
}

pub struct Workgroup<'w> {
    devices: Vec<Device>,
    shader_descriptor: wgpu::ShaderModuleDescriptor<'w>,
}

impl<'w> Workgroup<'w> {
    pub fn new(devices: Vec<Device>, shader: wgpu::ShaderModuleDescriptor<'w>) -> Self {
        Self {
            devices,
            shader_descriptor: shader,
        }
    }
}

pub struct Task<'t> {
    workgroup: Workgroup<'t>,
    staging_buffers: Vec<Vec<wgpu::Buffer>>,
    output_slice_pointers: Vec<SlicePointerWriter>,
    command_buffers: Vec<wgpu::CommandBuffer>,
}

impl<'t> Task<'t> {
    pub fn run(mut self) {
        for (device, command_buffer) in self
            .workgroup
            .devices
            .iter()
            .zip(self.command_buffers.into_iter())
        {
            device.queue.submit([command_buffer]);
        }

        // Collect all map_async completions into a Vec of std::sync::mpsc::Receiver
        let mut receivers = Vec::new();

        for (device_id, _device) in self.workgroup.devices.iter().enumerate() {
            for staging_buffer in self.staging_buffers[device_id].iter() {
                let buffer_slice = staging_buffer.slice(..);
                let (tx, rx) = std::sync::mpsc::channel();
                buffer_slice.map_async(wgpu::MapMode::Read, move |_| {
                    let _ = tx.send(());
                });
                receivers.push(rx);
            }
        }

        // Wait for all devices to finish.
        for device in self.workgroup.devices.iter() {
            device
                .device
                .poll(wgpu::PollType::wait_indefinitely())
                .unwrap();
        }

        // Wait for all devices to send data back to CPU owned memory.
        for rx in receivers {
            let _ = rx.recv();
        }

        for (device_id, _device) in self.workgroup.devices.iter().enumerate() {
            for (output_ptr_index, staging_buffer) in
                self.staging_buffers[device_id].iter().enumerate()
            {
                let buffer_slice = staging_buffer.slice(..);
                let data = buffer_slice.get_mapped_range();
                let result: &[u8] = bytemuck::cast_slice(&data);

                let slice_ptr_writer = &mut self.output_slice_pointers[output_ptr_index];
                slice_ptr_writer
                    .write(result)
                    .expect("Tried to write device result out of bounds of result buffer.")
            }
        }
    }
}

pub struct TaskBuilder<'t> {
    workgroup: Workgroup<'t>,
    kernel: Option<&'t str>,
    workgroups: Option<(u32, u32, u32)>,
    input_buffers: Vec<Vec<(wgpu::Buffer, wgpu::BindGroupLayoutEntry)>>,
    output_buffers: Vec<Vec<(wgpu::Buffer, wgpu::BindGroupLayoutEntry)>>,
    staging_buffers: Vec<Vec<wgpu::Buffer>>,
    output_slice_pointers: Vec<SlicePointerWriter>,
}

impl<'t> TaskBuilder<'t> {
    pub fn new(workgroup: Workgroup<'t>) -> Self {
        let num_devices = workgroup.devices.len();

        Self {
            workgroup,
            kernel: None,
            workgroups: None,
            input_buffers: vec![vec![]; num_devices],
            output_buffers: vec![vec![]; num_devices],
            staging_buffers: vec![vec![]; num_devices],
            output_slice_pointers: vec![],
        }
    }

    pub fn build(self) -> Task<'t> {
        let TaskBuilder {
            workgroup,
            kernel,
            workgroups,
            input_buffers,
            output_buffers,
            output_slice_pointers,
            staging_buffers,
        } = self;

        let mut command_buffers: Vec<wgpu::CommandBuffer> =
            Vec::with_capacity(workgroup.devices.len());

        for (device_index, device) in workgroup.devices.iter().enumerate() {
            let device_ref = &device.device;

            let bind_group_layout_entries: Vec<wgpu::BindGroupLayoutEntry> = input_buffers
                [device_index]
                .iter()
                .map(|ib| ib.1)
                .chain(output_buffers[device_index].iter().map(|ob| ob.1))
                .collect();

            let bind_group_layout =
                device_ref.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: None,
                    entries: &bind_group_layout_entries,
                });

            let bind_group_entries: Vec<wgpu::BindGroupEntry> = input_buffers[device_index]
                .iter()
                .chain(output_buffers[device_index].iter())
                .map(|e| wgpu::BindGroupEntry {
                    binding: e.1.binding,
                    resource: e.0.as_entire_binding(),
                })
                .collect();

            let bind_group = device_ref.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &bind_group_layout,
                entries: &bind_group_entries,
            });

            let pipeline_layout =
                device_ref.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: None,
                    bind_group_layouts: &[&bind_group_layout],
                    immediate_size: 0,
                });

            let pipeline = device_ref.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: None,
                layout: Some(&pipeline_layout),
                module: &device_ref.create_shader_module(workgroup.shader_descriptor.clone()),
                entry_point: kernel,
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

            let mut encoder =
                device_ref.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

            {
                let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: None,
                    timestamp_writes: None,
                });

                compute_pass.set_pipeline(&pipeline);
                compute_pass.set_bind_group(0, &bind_group, &[]);

                let (x, y, z) = workgroups.unwrap();
                compute_pass.dispatch_workgroups(x, y, z);

                // Drop compute pass
            } //// <------

            for (output_buffer, staging_buffer) in output_buffers[device_index]
                .iter()
                .map(|ob| &ob.0)
                .zip(staging_buffers[device_index].iter())
            {
                encoder.copy_buffer_to_buffer(
                    output_buffer,
                    0,
                    staging_buffer,
                    0,
                    output_buffer.size(),
                );
            }

            command_buffers.push(encoder.finish());
        }

        Task {
            workgroup,
            output_slice_pointers,
            staging_buffers,
            command_buffers,
        }
    }

    pub fn with_kernel(mut self, kernel: &'t str) -> Self {
        self.kernel = Some(kernel);
        self
    }

    pub fn with_workgroups(mut self, x: u32, y: u32, z: u32) -> Self {
        assert!(x != 0, "Workgroup size must be greater than 0.");
        assert!(y != 0, "Workgroup size must be greater than 0.");
        assert!(z != 0, "Workgroup size must be greater than 0.");
        self.workgroups = Some((x, y, z));
        self
    }

    pub fn with_input_buffer<T>(mut self, index: u32, buf: &[T]) -> Self
    where
        T: bytemuck::Pod,
    {
        for (device_index, device) in self.workgroup.devices.iter_mut().enumerate() {
            let label = format!("WSC_I{}", index);

            let buffer = device
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(&label),
                    contents: bytemuck::cast_slice(buf),
                    usage: wgpu::BufferUsages::STORAGE,
                });

            let layout_entry = wgpu::BindGroupLayoutEntry {
                binding: index,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            };

            self.input_buffers[device_index].push((buffer, layout_entry));
        }
        self
    }

    pub fn with_output_buffer<T>(mut self, index: u32, buf: &'t mut [T]) -> Self
    where
        T: bytemuck::Pod,
    {
        let buf_len = buf.len();

        for (device_index, device) in self.workgroup.devices.iter_mut().enumerate() {
            let mappable_primary_buffers =
                device.features.contains(Features::MAPPABLE_PRIMARY_BUFFERS);

            let output_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("WSC_O{}", index)),
                size: (buf_len * std::mem::size_of::<T>()) as wgpu::BufferAddress,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | if mappable_primary_buffers {
                        wgpu::BufferUsages::MAP_READ
                    } else {
                        wgpu::BufferUsages::empty()
                    },
                mapped_at_creation: false,
            });

            let output_layout_entry = wgpu::BindGroupLayoutEntry {
                binding: index,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            };

            self.output_buffers[device_index].push((output_buffer, output_layout_entry));

            // Staging buffer

            if mappable_primary_buffers {
                // No need for staging buffer because we are using the output buffer
                // as a staging buffer.
                continue;
            }

            let staging_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("WSC_S{}", index)),
                size: (buf_len * std::mem::size_of::<T>()) as wgpu::BufferAddress,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            self.staging_buffers[device_index].push(staging_buffer);
        }

        let buf_u8 = bytemuck::cast_slice_mut(buf);
        self.output_slice_pointers
            .push(SlicePointerWriter::from_slice(buf_u8));

        self
    }
}
