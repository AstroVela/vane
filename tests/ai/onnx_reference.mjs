// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

// Run from a temporary project with the official @huggingface/transformers
// dependency. This reference never imports Vane or the Python adapter.
import fs from 'node:fs';
import { AutoModel, AutoProcessor, RawImage, RawVideo, env } from '@huggingface/transformers';

const [modelPath, inputPath, outputPath, bundleSHA256] = process.argv.slice(2);
env.allowRemoteModels = false;
env.allowLocalModels = true;
const fixture = JSON.parse(fs.readFileSync(inputPath, 'utf8'));
const processor = await AutoProcessor.from_pretrained(modelPath, { local_files_only: true });
const model = await AutoModel.from_pretrained(modelPath, { local_files_only: true, device: 'cpu', dtype: 'fp32' });
const image = (pixels) => new RawImage(Uint8ClampedArray.from(pixels.flat(2)), pixels[0].length, pixels.length, 3);
const picture = image(fixture.image);
const audio = Float32Array.from(fixture.audio);
const video = new RawVideo(fixture.frames.map(image), 2);
const embedding = async (...inputs) => (await model(await processor(...inputs))).sentence_embedding.tolist();
try {
    const documents = await embedding(fixture.documents.map((text) => `title: none | text: ${text}`));
    const queries = await embedding(fixture.queries.map((text) => `task: search result | query: ${text}`));
    const media = [
        ...(await embedding(null, picture)),
        ...(await embedding(null, null, audio)),
        ...(await embedding(null, null, null, video)),
        ...(await embedding('title: none | text: red shape<|image|><|audio|><|video|>', picture, audio, video)),
    ];
    fs.writeFileSync(outputPath, JSON.stringify({ bundle_sha256: bundleSHA256, documents, queries, media }));
} finally {
    await model.dispose();
}
