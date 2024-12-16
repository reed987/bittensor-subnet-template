# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2023 crypto-ai-team

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.


import time
import bittensor as bt
import numpy as np
import copy
import asyncio

from crypto_ai.base.validator import BaseValidatorNeuron
from crypto_ai.validator import forward
from crypto_ai.protocol import MinerDataSynapse


class Validator(BaseValidatorNeuron):
    def __init__(self, config=None):
        super(Validator, self).__init__(config=config)

        bt.logging.info("load_state()")
        self.load_state()

        asyncio.run_coroutine_threadsafe(self.run_miners_update_routine(), self.loop)

    async def run_miners_update_routine(self):
        while True:
            bt.logging.info("Updating miners identity")
            await self.update_miners_identity()
            await asyncio.sleep(self.config.update_miners_routine_interval)
            

    async def forward(self):
        return await forward(self)
    
    async def get_miners_data(self):
        self.all_uids = [int(uid) for uid in self.metagraph.uids]
        uid_to_axon = dict(zip(self.all_uids, self.metagraph.axons))
        query_axons = [uid_to_axon[int(uid)] for uid in self.all_uids]
        bt.logging.info("Requesting miners data...")
        responses = self.dendrite.query(
            axons=query_axons,
            synapse=MinerDataSynapse(),
            deserialize=False,
            timeout=10,
        )
        responses = {
            uid: res.response
            for uid, res in zip(self.all_uids, responses)
        }
        return responses
    
    async def update_miners_identity(self):
        not_available_uids = []
        miners_data = await self.get_miners_data()
        if not miners_data:
            bt.logging.warning("Updating miners identity: No active miner available")
        for uid, data in miners_data.items():
            if data is None:
                not_available_uids.append(uid)
                continue

            miner_state = self.all_uids_info.setdefault(
                uid,
                {
                    "miner_mode": "Unknown",
                    "min_stake": 100,
                    "device_info": {
                        "gpu_device_name": "Unknown",
                        "gpu_device_count": "Unknown",
                    },
                },
            )

            miner_state["min_stake"] = data.get("min_stake", 100)
            miner_state["device_info"] = data.get("device_info", {})
            miner_state["miner_mode"] = data["miner_mode"]
            miner_state["hotkey"] = self.metagraph.hotkeys[uid]

        self.save_state()
        bt.logging.success("Updating miners identity: update successfull")
        bt.logging.info(f"Updating miners identity: No info available for UID {not_available_uids}")

    def save_state(self):
        """Saves the state of the validator to a file."""

        np.savez(
            self.config.neuron.full_path + "/state.npz",
            step=self.step,
            scores=self.scores,
            hotkeys=self.hotkeys,
            all_uids_info=self.all_uids_info,
        )

    def load_state(self):
        """Loads the state of the validator from a file."""
        # Load the state of the validator from file.
        try:
            state = np.load(self.config.neuron.full_path + "/state.npz", allow_pickle=True)
            all_uids_info = state.get("all_uids_info", {})
            if isinstance(all_uids_info, np.ndarray):
                all_uids_info = all_uids_info.item()  # Convert to dictionary if needed
        except Exception as e:
            bt.logging.error(e)
            state = {}
        self.step = state.get("step", 0)
        self.scores = state.get(
            "scores",
            np.zeros(self.metagraph.n, dtype=np.float32),
        )
        self.hotkeys = state.get("hotkeys", copy.deepcopy(self.metagraph.hotkeys))
        self.all_uids_info = all_uids_info if isinstance(all_uids_info, dict) else {
            uid: {"scores": [], "miner_mode": "", "is_tested": False} for uid in self.all_uids
        }


if __name__ == "__main__":
    with Validator() as validator:
        while True:
            bt.logging.info(f"Validator running... {time.time()}")
            time.sleep(5)
