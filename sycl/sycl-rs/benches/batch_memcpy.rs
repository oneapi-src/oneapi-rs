use std::{fmt::Display, time::Duration};

use bytemuck::Pod;
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use sycl_rs::{prelude::*, usm::UsmAlloc};

fn criterion_benchmark(c: &mut Criterion) {
    memcpy_benchmark(c, 10).unwrap();
}

fn memcpy_benchmark(c: &mut Criterion, test_count: u32) -> sycl_rs::Result<()> {
    static KB: usize = 1024;

    let used_memory = 4_usize.pow(test_count) * KB;

    let mut group = c.benchmark_group("memcpy_test");
    group.measurement_time(Duration::from_secs(30));

    for i in 1..test_count {
        let buf_count = 4_usize.pow(i);
        let buf_size = used_memory / buf_count;
        let copy_count = buf_count / 2;

        let mut queue = Queue::new();
        let mut host_buffers = Vec::<HostUsmBox<u32>>::new();
        let mut device_buffers = Vec::<DeviceUsmBox<u32>>::new();

        for _ in 1..copy_count {
            let host_buffer = unsafe { queue.alloc_uninit_host::<u32>(buf_size) };
            let device_buffer = unsafe { queue.alloc_uninit_device::<u32>(buf_size) };

            host_buffers.push(host_buffer);
            device_buffers.push(device_buffer);
        }

        queue.wait()?;

        group.bench_function(
            BenchmarkId::new(
                "h2d copy",
                BenchmarkParams {
                    copy_count,
                    buf_size,
                },
            ),
            |b| {
                b.iter(|| batch_mempcy(&mut queue, &host_buffers, &mut device_buffers));
            },
        );
    }

    group.finish();

    Ok(())
}

fn batch_mempcy<T, A1, A2>(
    queue: &mut Queue,
    sources: &[UsmBox<T, A1>],
    destinations: &mut [UsmBox<T, A2>],
) -> sycl_rs::Result<()>
where
    T: Pod,
    A1: UsmAlloc,
    A2: UsmAlloc,
{
    for (src, dst) in sources.iter().zip(destinations) {
        queue.copy(src, dst)?;
    }

    queue.wait()
}

struct BenchmarkParams {
    copy_count: usize,
    buf_size: usize,
}

impl Display for BenchmarkParams {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} copies of size {}", self.copy_count, self.buf_size)
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
