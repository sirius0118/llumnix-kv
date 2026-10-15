"""Keep vLLM's native client and OutputProcessor across execution relocation."""

import asyncio
import struct
from collections import defaultdict

from vllm.v1.engine import EngineCoreOutputs
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder

from ..utils import get_ip
from .transaction import decode, encode


class MigrationClient:
    def __init__(self, native, timeout=300):
        self.native = native
        self.timeout = timeout
        self.engines = dict(zip(native.engine_ranks_managed, native.core_engines))
        self.queue = asyncio.Queue()
        self.cancel_requests = set()
        self.cancellation_changed = asyncio.Event()
        self.children = defaultdict(set)
        self.expected = {}
        self.outboxes = {}
        self.tasks = []
        self.server = None
        self.encoder = MsgpackEncoder(size_threshold=1 << 63)
        self.decoder = MsgpackDecoder(EngineCoreOutputs)

    def __getattr__(self, name):
        return getattr(self.native, name)

    async def start(self):
        self.server = await asyncio.start_server(self._receive, get_ip(), 0)
        host, port = self.server.sockets[0].getsockname()[:2]
        self.address = f"{host}:{port}"
        self.tasks.append(asyncio.create_task(self._pump()))

    async def utility(self, rank, method, *args):
        return await asyncio.wait_for(
            self.native._call_utility_async(method, *args, engine=self.engines[rank]), self.timeout
        )

    async def rpc(self, rank, operation, **payload):
        result = decode(await self.utility(rank, "asyncpb_tail", operation, encode(payload)))
        if operation == "release" and result is True:
            self.released(payload["metadata"]["request_id"], rank)
        return result

    async def pause(self, ranks, paused):
        method, args = ("pause_scheduler", ("keep", False)) if paused else ("resume_scheduler", ())
        await asyncio.gather(*(self.utility(rank, method, *args) for rank in ranks))

    async def activate(self, rank, metadata):
        state = await self.rpc(rank, "state", metadata=metadata, side="target")
        if not state["activated"]:
            request = decode(
                await self.rpc(
                    rank,
                    "activate",
                    metadata=metadata,
                    owner=dict(group_id=metadata["target"], owner_generation=metadata["owner_generation"] + 1),
                )
            )
            request.data_parallel_rank = rank
            await self.native.add_request_async(request)
        async with asyncio.timeout(self.timeout):
            while not (await self.rpc(rank, "state", metadata=metadata, side="target"))["activated"]:  # noqa: ASYNC110
                await asyncio.sleep(0.01)

    async def add_request_async(self, request):
        parent = (request.sampling_params.extra_args or {}).get("llx_parent")
        if parent is None:
            raise ValueError("Migration client requires an admitted parent request")
        request.queue_server_address = self.address
        self.children[parent].add(request.request_id)
        self.expected[request.request_id] = [0, 0, {}]
        await self.native.add_request_async(request)

    async def abort_requests_async(self, request_ids):
        # The coordinator serializes these with commit and retires DMA before
        # reclaiming either side. AsyncLLM already detached the logical outputs.
        self.cancel_requests.update(request_ids)

    def cancelled(self, request_ids):
        self.cancel_requests.difference_update(request_ids)
        self.cancellation_changed.set()

    async def finish(self, parent):
        children = self.children.get(parent, set())
        while self.cancel_requests.intersection(children):
            self.cancellation_changed.clear()
            await self.cancellation_changed.wait()
        for request_id in self.children.pop(parent, ()):
            self.expected.pop(request_id, None)

    def released(self, request_id, rank):
        routes = getattr(self.native, "reqs_in_flight", {})
        if routes.get(request_id) == self.engines[rank]:
            routes.pop(request_id, None)

    async def get_output_async(self):
        output = await self.queue.get()
        if isinstance(output, BaseException):
            raise output
        return output

    def _accept(self, batch):
        ordered = []
        for output in batch.outputs:
            state = self.expected.get(output.request_id)
            if state is None:
                continue  # Logical parent finished/cancelled; acknowledge retries.
            offset, generation = output.migration_offset, output.migration_generation
            if offset < state[0]:
                continue
            key = (offset, generation)
            state[2].setdefault(key, output)
            while state[2]:
                key = min(state[2])
                if key[0] < state[0]:
                    state[2].pop(key)
                    continue
                if key[0] != state[0]:
                    break
                item = state[2].pop(key)
                if key[1] < state[1]:
                    raise RuntimeError("Stale migration owner produced new output")
                state[0] += len(item.new_token_ids)
                state[1] = key[1]
                ordered.append(item)
                if item.finished:
                    self.expected.pop(item.request_id, None)
                    break
        batch.outputs = ordered
        self.queue.put_nowait(batch)

    async def _pump(self):
        try:
            while True:
                batch = await self.native.get_output_async()
                destinations = defaultdict(list)
                for output in batch.outputs:
                    destinations[output.queue_server_address].append(output)
                for address, outputs in destinations.items():
                    if address == self.address:
                        continue
                    if not address:
                        raise RuntimeError("Migrating request lost its result address")
                    if address not in self.outboxes:
                        self.outboxes[address] = asyncio.Queue()
                        self.tasks.append(asyncio.create_task(self._send(address)))
                    # Native finished_requests/DP waves/stats remain on this
                    # native client; only application outputs cross replicas.
                    self.outboxes[address].put_nowait(EngineCoreOutputs(outputs=outputs))
                batch.outputs = destinations.get(self.address, [])
                self._accept(batch)
        except Exception as exc:
            self.queue.put_nowait(exc)

    async def _send(self, address):
        reader = writer = None
        try:
            while True:
                batch = await self.outboxes[address].get()
                payload = b"".join(self.encoder.encode(batch))
                for attempt in range(2):
                    try:
                        if writer is None:
                            host, port = address.rsplit(":", 1)
                            reader, writer = await asyncio.open_connection(host, int(port))
                        writer.write(struct.pack("!I", len(payload)) + payload)
                        await writer.drain()
                        if await asyncio.wait_for(reader.readexactly(1), 30) != b"\x01":
                            raise RuntimeError("Invalid output acknowledgement")
                        break
                    except (OSError, asyncio.IncompleteReadError, TimeoutError):
                        if writer is not None:
                            writer.close()
                        reader = writer = None
                        if attempt:
                            raise
        except Exception as exc:
            self.queue.put_nowait(exc)
        finally:
            if writer is not None:
                writer.close()

    async def _receive(self, reader, writer):
        try:
            while True:
                size = struct.unpack("!I", await reader.readexactly(4))[0]
                if size > 256 * 1024**2:
                    raise ValueError("Oversized migration output frame")
                batch = self.decoder.decode(await reader.readexactly(size))
                self._accept(batch)
                writer.write(b"\x01")
                await writer.drain()
        except (asyncio.IncompleteReadError, ConnectionError):
            pass
        except Exception as exc:
            self.queue.put_nowait(exc)
        finally:
            writer.close()

    def shutdown(self, *args, **kwargs):
        for task in self.tasks:
            task.cancel()
        if self.server is not None:
            self.server.close()
        return self.native.shutdown(*args, **kwargs)
