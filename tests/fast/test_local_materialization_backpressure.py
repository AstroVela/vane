# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""A blocking native sink must not unbound its local transport branch."""

import json
import os
import subprocess
import sys

import pytest


@pytest.mark.parametrize("materializer", ["window", "sort", "aggregate", "join"])
def test_large_stream_through_native_materializer_preserves_backpressure(tmp_path, materializer):
    script = r"""
import hashlib
import json
import sys
import pyarrow as pa
import vane
from vane.execution import local_query_admission as admission
from vane.execution import udf_shm_store as storage

captured=[]
original=admission.LocalQueryAdmission.__init__
def observe(self,*args,**kwargs):
    original(self,*args,**kwargs)
    captured.append(self)
admission.LocalQueryAdmission.__init__=observe
payload_size=2*1024**2
def data(identifier):
    return str(identifier).encode()+b'x'*payload_size
def expand(batch):
    for seed in batch['id'].to_pylist():
        for part in range(3):
            identifier=seed*3+part
            yield pa.table({'id':[identifier], 'payload':[data(identifier)]})
def forward(batch):
    return batch
class Model:
    def __call__(self,batch):
        return pa.table({'id':batch['id'],'payload':batch['payload'],
            'digest':[hashlib.sha256(value).hexdigest() for value in batch['payload'].to_pylist()]})
def consume(batch):
    # Verify again after the materializer, before dropping its large payload.
    assert all(hashlib.sha256(value).hexdigest()==digest
        for value,digest in zip(batch['payload'].to_pylist(),batch['digest'].to_pylist(),strict=True))
    return batch.select(['id','digest'])
options=dict(task_input_max_bytes=16*1024**2,output_target_max_bytes=16*1024**2)
try:
    with vane.connect(config={'threads':4}) as con:
        relation=con.sql('select range::BIGINT as id from range(100)').map_batches(
            expand,schema={'id':'BIGINT','payload':'BLOB'},batch_size=1,min_task_batch_size=1,
            output_batch_size=1,execution_backend='subprocess_task',**options)
        for _ in range(2):
            relation=relation.map_batches(forward,schema={'id':'BIGINT','payload':'BLOB'},
                batch_size=1,min_task_batch_size=1,output_batch_size=8,execution_backend='subprocess_task',**options)
        schema={'id':'BIGINT','payload':'BLOB','digest':'VARCHAR'}
        relation=relation.map_batches(Model,schema=schema,batch_size=8,min_task_batch_size=8,
            actor_number=1,execution_backend='subprocess_actor',**options)
        relation=relation.map_batches(forward,schema=schema,batch_size=1,min_task_batch_size=1,
            execution_backend='subprocess_task',**options)
        case=sys.argv[1]
        if case=='window':
            relation=relation.project('*, row_number() over (order by id) as ordinal')
        elif case=='sort':
            relation=relation.order('id DESC')
        elif case=='aggregate':
            relation=relation.aggregate('id, any_value(payload) AS payload, any_value(digest) AS digest','id')
        else:
            con.execute("SET disabled_optimizers='join_order,build_side_probe_side'")
            probe=con.sql('select range::BIGINT as probe_id from range(300)')
            relation=probe.join(relation,'probe_id=id').project('id,payload,digest')
        relation=relation.map_batches(consume,schema={'id':'BIGINT','digest':'VARCHAR'},
            batch_size=1,min_task_batch_size=1,execution_backend='subprocess_task',**options)
        rows=relation.to_arrow_table().to_pylist()
        assert len(rows)==300 and {row['id'] for row in rows}==set(range(300))
        assert all(row['digest']==hashlib.sha256(data(row['id'])).hexdigest() for row in rows)
    assert captured and any(q.graph.materialization_barriers for q in captured)
    assert all(not b.retains_object_store_inputs for q in captured for b in q.graph.materialization_barriers)
    assert all(q.snapshot()['execution_phase']['object_store_unlimited_unit_ids']==[] for q in captured)
    assert sum(q.snapshot()['liveness']['task_grants_total'] for q in captured)>0
    assert all(not q.cleanup_pending() for q in captured)
finally:
    vane.teardown_runner()
assert storage.local_shm_store_snapshot()['live_bytes']==0
print(json.dumps({'rows':len(rows),'materializer':case}))
"""
    environment = dict(
        os.environ,
        VANE_RUNNER="local-fast",
        VANE_PROGRESS="0",
        VANE_LOCAL_SHM_STORE_BYTES="250000000",
        VANE_LOCAL_SHM_REF_BUDGET_BYTES="125000000",
        OMP_NUM_THREADS="2",
        OPENBLAS_NUM_THREADS="1",
        TEMP=str(tmp_path),
        TMP=str(tmp_path),
        TMPDIR=str(tmp_path),
    )
    result = subprocess.run(
        [sys.executable, "-I", "-c", script, materializer],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=70,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout.splitlines()[-1]) == {"rows": 300, "materializer": materializer}
