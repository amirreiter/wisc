use wisc::prelude::*;

#[test]
fn overrides() {
    // Get all the hardware devices available to our system.
    let devices = VDevice::all();

    // Create a Workgroup out of our device(s).
    let mut workgroup = Workgroup::from_devices(devices);

    // Set compile-time override constants for the shader.
    let a = 2.0;
    let b = -1.0;

    // Register our buffers with the runtime.
    let ibuf = workgroup.create_vbuffer(vec![13.0f32; 256]);
    let obuf = workgroup.create_vbuffer(vec![-1.0f32; 256]);

    // Define our task and input our buffers.
    let task = TaskBuilder::new(&mut workgroup, include_wgsl!("./overrides.wgsl"))
        .with_kernel("main")
        .with_size((1, 1, 1))
        .with_override(0, a)
        .with_override(1, b)
        .with_input_buffer(0, ibuf)
        .with_output_buffer(1, obuf)
        .build()
        .expect("Failed to build task");

    // Block the current thread while the task runs.
    task.run();

    // Take ownership of the buffer from the runtime.
    let obuf1: Vec<f32> = workgroup.take_vbuffer(obuf).unwrap();

    assert_eq!(obuf1, vec![a * 13.0 + b; 256]);
}
