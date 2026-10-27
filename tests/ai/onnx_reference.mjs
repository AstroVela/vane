// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

// Run from a temporary project with the official @huggingface/transformers
// dependency. This reference never imports Vane or the Python adapter.
import fs from 'node:fs';
import path from 'node:path';
import { AutoModel, Tensor, env } from '@huggingface/transformers';

const [modelPath, inputPath, outputPath, bundleSHA256] = process.argv.slice(2);
env.allowRemoteModels = false;
env.allowLocalModels = true;
const fixture = JSON.parse(fs.readFileSync(inputPath, 'utf8'));
const model = await AutoModel.from_pretrained(modelPath, { local_files_only: true, device: 'cpu', dtype: 'fp32' });
// Share the declared official Python preprocessing contract, so this reference
// measures model execution rather than different image resampling libraries.
const embedding = async (name) => {
    const values = {};
    for (const [key, definition] of Object.entries(fixture.cases[name])) {
        const bytes = fs.readFileSync(path.join(path.dirname(inputPath), definition.file));
        const buffer = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
        const array = definition.type === 'float32' ? new Float32Array(buffer)
            : definition.type === 'int64' ? new BigInt64Array(buffer) : new Uint8Array(buffer);
        values[key] = new Tensor(definition.type, array, definition.dims);
    }
    return (await model(values)).sentence_embedding.tolist();
};
try {
    const documents = await embedding('documents');
    const queries = await embedding('queries');
    const media = [
        ...(await embedding('image')),
        ...(await embedding('audio')),
        ...(await embedding('video')),
        ...(await embedding('mixed')),
        ...(await embedding('tiny_image')),
        ...(await embedding('tiny_video')),
        ...(await embedding('unequal_audio')),
    ];
    fs.writeFileSync(outputPath, JSON.stringify({ bundle_sha256: bundleSHA256, processor_versions: fixture.processor_versions, documents, queries, media }));
} finally {
    await model.dispose();
}
