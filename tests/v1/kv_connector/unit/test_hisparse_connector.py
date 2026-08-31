# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace
from unittest.mock import MagicMock

from vllm.v1.worker.gpu.kv_connector import ActiveKVConnector


def test_no_forward_enqueues_deferred_hisparse_transfers():
    """A zero-token step must still enqueue deferred post-forward transfers."""
    connector = object.__new__(ActiveKVConnector)
    connector._disabled = False
    connector.pre_forward = MagicMock()
    connector.finish_forward = MagicMock()
    connector.post_forward = MagicMock(return_value=None)

    scheduler_output = SimpleNamespace(finished_req_ids=set())
    connector.no_forward(scheduler_output)

    connector.pre_forward.assert_called_once_with(scheduler_output)
    connector.finish_forward.assert_called_once_with()


def test_active_connector_forwards_cudagraph_support():
    connector = object.__new__(ActiveKVConnector)
    connector._disabled = False
    connector.kv_connector = MagicMock()
    connector.kv_connector.supports_cudagraph.return_value = False
    metadata = object()
    scheduler_output = SimpleNamespace(kv_connector_metadata=metadata)

    assert not connector.supports_cudagraph(scheduler_output)
    connector.kv_connector.supports_cudagraph.assert_called_once_with(metadata)
