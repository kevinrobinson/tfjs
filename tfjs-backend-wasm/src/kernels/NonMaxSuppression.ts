/**
 * @license
 * Copyright 2019 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import {NamedAttrMap, NamedTensorInfoMap, registerKernel, TensorInfo, util} from '@tensorflow/tfjs-core';

import {BackendWasm} from '../backend_wasm';

interface NonMaxSuppressionInputs extends NamedTensorInfoMap {
  boxes: TensorInfo;
  scores: TensorInfo;
}

interface NonMaxSuppressionAttrs extends NamedAttrMap {
  maxOutputSize: number;
  iouThreshold: number;
  scoreThreshold: number;
}

let wasmNonMaxSuppression: (
    boxesId: number, scoresId: number, maxOutputSize: number,
    iouThreshold: number, scoreThreshold: number) => void;

function setup(backend: BackendWasm) {
  wasmNonMaxSuppression =
      backend.wasm.cwrap('NonMaxSuppression', null /* void */, [
        'number',  // boxes
        'number',  // scores
        'number',  // maxOutputSize
        'number',  // iouThreshold
        'number',  // scoreThreshold
      ]);
}

function nonMaxSuppression(args: {
  inputs: NonMaxSuppressionInputs,
  backend: BackendWasm,
  attrs: NonMaxSuppressionAttrs
}) {
  const {inputs, backend, attrs} = args;
  const {boxes, scores} = inputs;
  const {maxOutputSize, iouThreshold, scoreThreshold} = attrs;
  const boxesId = backend.dataIdMap.get(boxes.dataId).id;
  const scoresId = backend.dataIdMap.get(scores.dataId).id;

  const out = backend.makeOutput([batchDim, leftDim, rightDim], a.dtype);
  const outId = backend.dataIdMap.get(out.dataId).id;

  wasmNonMaxSuppression(
      aId, bId, sharedDim, leftDim, rightDim, batchDim, aBatch, aOuterStep,
      aInnerStep, bBatch, bOuterStep, bInnerStep, outId);
  return out;
}

registerKernel({
  kernelName: 'NonMaxSuppression',
  backendName: 'wasm',
  setupFunc: setup,
  kernelFunc: nonMaxSuppression
});
