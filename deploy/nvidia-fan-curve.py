#!/usr/bin/python3

import ctypes
import signal
import sys
import time


NVML_TEMPERATURE_GPU = 0
CURVE = (
    (0, 45),
    (40, 45),
    (45, 55),
    (50, 65),
    (55, 75),
    (60, 85),
    (64, 95),
    (70, 100),
)


class NvmlError(RuntimeError):
    pass


def check(result, operation):
    if result != 0:
        raise NvmlError(f"{operation} failed with NVML status {result}")


def fan_speed_for_temperature(temperature):
    for index in range(1, len(CURVE)):
        low_temp, low_speed = CURVE[index - 1]
        high_temp, high_speed = CURVE[index]
        if temperature <= high_temp:
            span = high_temp - low_temp
            offset = temperature - low_temp
            return round(low_speed + offset * (high_speed - low_speed) / span)
    return CURVE[-1][1]


def main():
    nvml = ctypes.CDLL("libnvidia-ml.so.1")
    check(nvml.nvmlInit_v2(), "nvmlInit_v2")

    device = ctypes.c_void_p()
    check(nvml.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(device)), "get GPU 0")
    fan_count = ctypes.c_uint()
    check(nvml.nvmlDeviceGetNumFans(device, ctypes.byref(fan_count)), "get fan count")
    if fan_count.value == 0:
        raise NvmlError("GPU 0 reports no controllable fans")

    running = True

    def stop(_signum, _frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    current_speed = None
    try:
        while running:
            temperature = ctypes.c_uint()
            check(
                nvml.nvmlDeviceGetTemperature(
                    device, NVML_TEMPERATURE_GPU, ctypes.byref(temperature)
                ),
                "get GPU temperature",
            )
            desired_speed = fan_speed_for_temperature(temperature.value)
            if current_speed is None or abs(desired_speed - current_speed) >= 2:
                for fan in range(fan_count.value):
                    check(
                        nvml.nvmlDeviceSetFanSpeed_v2(device, fan, desired_speed),
                        f"set fan {fan}",
                    )
                current_speed = desired_speed
                print(
                    f"temperature={temperature.value}C fan={desired_speed}%",
                    flush=True,
                )
            time.sleep(3)
    finally:
        for fan in range(fan_count.value):
            nvml.nvmlDeviceSetDefaultFanSpeed_v2(device, fan)
        nvml.nvmlShutdown()


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"nvidia-fan-curve: {error}", file=sys.stderr, flush=True)
        sys.exit(1)
