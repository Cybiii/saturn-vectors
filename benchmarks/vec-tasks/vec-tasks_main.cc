// See LICENSE for license details.

#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#include "utasks.h"
#include "vec-tasks.h"
#include "util.h"
#include "dataset1.h"

#define NUM_CORES (1)

uint32_t output_data[DATA_SIZE];

extern "C" void __main(void) {
  size_t mhartid = read_csr(mhartid);
  if (mhartid >= NUM_CORES) while (1);

  runner_t runner(0, new heap_allocator_t);

  // construct the tasks
  stream_task_t<uint32_t>* stream_task = new stream_task_t<uint32_t>(512);
  uint32_scale_task_t* scale_task = new uint32_scale_task_t(2, 2048, 512);
  uint32_add_task_t* add_task = new uint32_add_task_t(3, 2048, 512);

  // construct the task graph
  stream_task->chain(scale_task);
  scale_task->chain(add_task);
  add_task->terminate(output_data);

  // assign all tasks to the single runner — add in reverse pipeline order so
  // schedule_task drains downstream stages first, preventing upstream deadlock
  runner.add_task(add_task);
  runner.add_task(scale_task);
  runner.add_task(stream_task);

  // warm up the system, drive the pipeline cooperatively
  stream_task->push_work(input_data, DATA_SIZE);
  while (stream_task->has_work() || scale_task->has_work() || add_task->has_work())
    runner.step();

  for (size_t i = 0; i < DATA_SIZE; i++) {
    uint32_t ref = verify_data[i];
    uint32_t out = output_data[i];
    if (out != ref) {
      PRINTF("early Mismatch %p %x != %x\n", &output_data[i], out, ref);
      exit(1);
    }
  }

  add_task->terminate(output_data);

  // measure the system, drive the pipeline cooperatively
  PRINTF("Warmed up, starting measurement\n");
  size_t start = read_csr(mcycle);
  stream_task->push_work(input_data, DATA_SIZE);
  stream_task->set_may_finish();
  while (!add_task->is_finished())
    runner.step();
  size_t end = read_csr(mcycle);

  PRINTF("task took %ld cycles\n", end - start);

  for (size_t i = 0; i < DATA_SIZE; i++) {
    uint32_t ref = verify_data[i];
    uint32_t out = output_data[i];
    if (out != ref) {
      PRINTF("Mismatch %ld %p %x != %x\n", i, &output_data[i], out, ref);
      exit(1);
    }
  }
}

int main(void) {
  __main();
  return 0;
}
