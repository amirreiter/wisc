use std::any::TypeId;

use bytemuck::Pod;
use slotmap::SlotMap;

use crate::{vbuffer::VBuffer, vdevice::VDevice};

slotmap::new_key_type! { pub struct VBufferHandle; }

pub struct Workgroup {
    pub(crate) vdevices: Vec<VDevice>,
    pub(crate) vdevice_weightings: Vec<f32>,

    // The owned I/O buffers that implement pod, as enforced by constructor.
    pub(crate) vbuffers: SlotMap<VBufferHandle, VBuffer>,
}

impl Workgroup {
    pub fn vdevice_weightings(&self) -> Vec<(String, f32)> {
        self.vdevices
            .iter()
            .map(|vd| vd.label.clone())
            .zip(self.vdevice_weightings.iter().cloned())
            .collect()
    }

    pub fn from_devices(devices: Vec<VDevice>) -> Self {
        // If we have multiple devices, we weight them based on estimates of their
        // compute power.
        //
        // TODO: This estimation is very crude, so in the future this might
        // be configurable by the user.

        let device_weights: Vec<f32> = devices
            .iter()
            .map(|vd| {
                let base = vd.limits.max_compute_invocations_per_workgroup as f32;

                let memory_proxy = if vd.info.device_type == wgpu::DeviceType::Cpu {
                    1.0
                } else {
                    (vd.limits.max_buffer_size as f32 / 1_048_576.0)
                        .log2()
                        .max(1.0)
                };

                let type_multiplier = match vd.info.device_type {
                    wgpu::DeviceType::DiscreteGpu => 10.0,
                    wgpu::DeviceType::IntegratedGpu => 3.0,
                    wgpu::DeviceType::VirtualGpu => 2.0,
                    wgpu::DeviceType::Cpu => 1.0,
                    wgpu::DeviceType::Other => 1.0,
                };

                base * memory_proxy * type_multiplier
            })
            .collect();

        let total_weight: f32 = device_weights.iter().sum();
        let device_weights_normalized: Vec<f32> =
            device_weights.iter().map(|w| w / total_weight).collect();

        // Sort devices from strongest to weakest
        let mut device_weight_pairs: Vec<(VDevice, f32)> = devices
            .into_iter()
            .zip(device_weights_normalized.into_iter())
            .collect();

        device_weight_pairs
            .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let (devices, device_weights_normalized): (Vec<_>, Vec<_>) =
            device_weight_pairs.into_iter().unzip();

        Self {
            vdevices: devices,
            vdevice_weightings: device_weights_normalized,
            vbuffers: SlotMap::default(),
        }
    }

    pub fn create_vbuffer<T: Pod>(&mut self, data: Vec<T>) -> VBufferHandle {
        let length = data.len();
        let stride = std::mem::size_of::<T>();
        let tbox = Box::new(data);
        self.vbuffers.insert(VBuffer {
            inner: tbox,
            typeid: TypeId::of::<T>(),
            stride,
            length,
        })
    }

    pub fn take_vbuffer<T: Pod>(&mut self, buffer_handle: VBufferHandle) -> Option<Vec<T>> {
        let typeid = self.vbuffers.get(buffer_handle)?.typeid;

        if TypeId::of::<T>() == typeid {
            let vbuffer = self.vbuffers.remove(buffer_handle)?;
            let anybox = vbuffer.inner;
            anybox.downcast::<Vec<T>>().ok().map(|anybox| *anybox)
        } else {
            None
        }
    }
}
