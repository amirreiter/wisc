use wisc::{prelude::*, workgroup::VBufferHandle};

#[test]
fn array_addition() {
    // Get all the hardware devices available to our system.
    let devices = VDevice::all();

    // Create a Workgroup out of our device(s).
    let mut workgroup = Workgroup::from_devices(devices);

    // Register our buffers with the runtime.
    // The runtime returns a handle to the virtual buffer.
    let ibuf1: VBufferHandle = workgroup.create_vbuffer(vec![2u32; 1024]);
    let ibuf2: VBufferHandle = workgroup.create_vbuffer(vec![3u32; 1024]);
    let obuf1: VBufferHandle = workgroup.create_vbuffer(vec![0u32; 1024]);

    // Define our task and input our buffers.
    let task = TaskBuilder::new(&mut workgroup, include_wgsl!("./array_addition.wgsl"))
        .with_kernel("main")
        // The shader is defined with workgroup size (256, 1, 1), so 4 * 256 invocations is
        // enough to cover the length of our data (1024)
        .with_size((4, 1, 1))
        .with_input_buffer(0, ibuf1, PartitionMode::Unmanaged)
        .with_input_buffer(1, ibuf2, PartitionMode::Unmanaged)
        .with_output_buffer(2, obuf1, PartitionMode::Unmanaged)
        .build()
        .expect("Failed to build task");

    // Block the current thread while the task runs.
    task.run();

    // Take ownership of the buffer from the runtime.
    let obuf1: Vec<u32> = workgroup.take_vbuffer(obuf1).unwrap();

    assert_eq!(obuf1, vec![5u32; 1024]);

    // Any buffers still managed by the runtime are freed automatically.
}
