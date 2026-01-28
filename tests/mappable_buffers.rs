use std::time::Instant;

use wisc::*;

#[test]
fn mappable_buffers() {
    const ITERATIONS: usize = 8;

    let a: Vec<u32> = vec![2; 1024];
    let b: Vec<u32> = vec![3; 1024];

    let mut r: Vec<u32> = vec![0; 1024];

    let start_non_mappable = Instant::now();

    for _ in 0..ITERATIONS {
        // Get all the hardware devices available to our system.
        let devices = Device::all_with_features(Features::empty());

        // Create a work group out of our shader and device cluster.
        let workgroup = Workgroup::new(devices, include_wgsl!("array_addition.wgsl"));

        // Create a task to be executed.
        let task = TaskBuilder::new(workgroup)
            .with_kernel("main")
            .with_workgroups(16, 1, 1)
            .with_input_buffer(0, a.as_slice())
            .with_input_buffer(1, b.as_slice())
            .with_output_buffer(2, r.as_mut_slice())
            .build();

        // Execute the task.
        task.run();
    }

    let duration_non_mappable = Instant::now().duration_since(start_non_mappable);

    let start_mappable = Instant::now();

    for _ in 0..ITERATIONS {
        // Get all the hardware devices available to our system.
        let devices = Device::all_with_features(Features::MAPPABLE_PRIMARY_BUFFERS);

        // Create a work group out of our shader and device cluster.
        let workgroup = Workgroup::new(devices, include_wgsl!("array_addition.wgsl"));

        // Create a task to be executed.
        let task = TaskBuilder::new(workgroup)
            .with_kernel("main")
            .with_workgroups(16, 1, 1)
            .with_input_buffer(0, a.as_slice())
            .with_input_buffer(1, b.as_slice())
            .with_output_buffer(2, r.as_mut_slice())
            .build();

        // Execute the task.
        task.run();
    }

    let duration_mappable = Instant::now().duration_since(start_mappable);

    // Confirm the result.
    assert!(duration_mappable < duration_non_mappable);
}
