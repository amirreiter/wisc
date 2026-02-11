use std::collections::HashMap;

use futures_lite::future;
use wgpu;

const REQUESTED_FEATURES: wgpu::Features = wgpu::Features::MAPPABLE_PRIMARY_BUFFERS;

#[derive(Debug)]
pub struct VDevice {
    pub(crate) label: String,
    pub(crate) info: wgpu::AdapterInfo,
    pub(crate) limits: wgpu::Limits,
    pub(crate) features: wgpu::Features,
    pub(crate) device: wgpu::Device,
    pub(crate) queue: wgpu::Queue,
}

impl VDevice {
    pub fn best() -> Option<Self> {
        Self::best_with_features(REQUESTED_FEATURES, wgpu::Features::empty())
    }

    pub fn best_with_features(requested: wgpu::Features, required: wgpu::Features) -> Option<Self> {
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

            let label = format!("WISC VDevice {}", adapter.get_info().device);

            let (device, queue) = adapter
                .request_device(&wgpu::DeviceDescriptor {
                    label: Some(&label),
                    required_features: adapter.features().intersection(requested).union(required),
                    required_limits: wgpu::Limits::downlevel_defaults(),
                    memory_hints: wgpu::MemoryHints::Performance,
                    trace: wgpu::Trace::Off,
                    experimental_features: wgpu::ExperimentalFeatures::disabled(),
                })
                .await
                .ok()?;

            Some(Self {
                label,
                info: adapter.get_info(),
                limits: adapter.limits(),
                features: device.features(),
                device,
                queue,
            })
        })
    }

    pub fn all() -> Vec<Self> {
        Self::all_with_features(REQUESTED_FEATURES, wgpu::Features::empty())
    }

    pub fn all_with_features(requested: wgpu::Features, required: wgpu::Features) -> Vec<Self> {
        future::block_on(async {
            let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
                backends: wgpu::Backends::all(),
                ..Default::default()
            });
            let adapters = instance.enumerate_adapters(wgpu::Backends::all()).await;

            let mut physical_groups: HashMap<(u32, u32), Vec<wgpu::Adapter>> = HashMap::new();
            for adapter in adapters {
                let info = adapter.get_info();
                physical_groups
                    .entry((info.vendor, info.device))
                    .or_default()
                    .push(adapter);
            }

            let mut results = Vec::new();

            for (_, mut adapters) in physical_groups {
                adapters.sort_by_key(|a| match a.get_info().backend {
                    wgpu::Backend::Vulkan => 0,
                    wgpu::Backend::Dx12 => 1,
                    wgpu::Backend::Metal => 2,
                    wgpu::Backend::Gl => 4,
                    _ => 5,
                });

                let adapter = &adapters[0];

                if !adapter
                    .get_downlevel_capabilities()
                    .flags
                    .contains(wgpu::DownlevelFlags::COMPUTE_SHADERS)
                {
                    continue;
                }

                let label = format!("WISC VDevice {}", adapter.get_info().device);

                let device_result = adapter
                    .request_device(&wgpu::DeviceDescriptor {
                        label: Some(&label),
                        required_features: adapter
                            .features()
                            .intersection(requested)
                            .union(required),
                        required_limits: adapter.limits(),
                        memory_hints: wgpu::MemoryHints::Performance,
                        ..Default::default()
                    })
                    .await;

                if let Ok((device, queue)) = device_result {
                    results.push(Self {
                        label,
                        info: adapter.get_info(),
                        limits: adapter.limits(),
                        features: device.features(),
                        device,
                        queue,
                    });
                }
            }

            results
        })
    }
}
