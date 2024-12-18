def set_info(self):
    gpu_device_name, gpu_device_count = get_device_info()
    miner_info = {
        "min_stake": self.config.min_stake,
        "device_info": {
            "gpu_device_name": gpu_device_name,
            "gpu_device_count": gpu_device_count,
        },
        "miner_mode": "example"
    }
    return miner_info


def get_device_info():
    try:
        import subprocess

        # Use subprocess to run nvidia-smi command and capture its output
        result = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        if result.returncode == 0:
            gpu_device_name = result.stdout.strip()
            gpu_device_count = len(gpu_device_name.split('\n'))
        else:
            gpu_device_name = "cpu"
            gpu_device_count = 0
    except Exception:
        gpu_device_name = "cpu"
        gpu_device_count = 0
    return gpu_device_name, gpu_device_count
