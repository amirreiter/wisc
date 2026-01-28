use wisc::*;

#[test]
fn array_addition() {
    // Get all the hardware devices available to our system.
    let devices = Device::all();

    // Create a work group out of our shader and device cluster.
    let workgroup = Workgroup::new(devices, include_wgsl!("array_addition.wgsl"));

    let a: Vec<u32> = vec![2; 1024];
    let b: Vec<u32> = vec![3; 1024];

    let mut r: Vec<u32> = vec![0; 1024];

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

    // Confirm the result.
    assert_eq!(r, vec![5; 1024]);
}
