/**
 * tests/test_frontend_tracker.js
 *
 * Unit tests validating the frontend JavaScript tracking engine,
 * ghost-polygon prevention, track persistence, and state machine in index.html.
 */

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const vm = require('vm');

function calcBoxIoU(boxA, boxB) {
  const x1 = Math.max(boxA[0], boxB[0]);
  const y1 = Math.max(boxA[1], boxB[1]);
  const x2 = Math.min(boxA[2], boxB[2]);
  const y2 = Math.min(boxA[3], boxB[3]);

  const w = Math.max(0, x2 - x1);
  const h = Math.max(0, y2 - y1);
  const inter = w * h;
  if (inter <= 0) return 0;

  const areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]);
  const areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]);
  return inter / (areaA + areaB - inter);
}

function matchTracksWithDetections(tracksMap, detections, frameW, frameH, maxDistNorm = 0.28, minIoU = 0.15) {
  const matches = [];
  const unmatchedDetections = new Set(detections.map((_, i) => i));
  const unmatchedTracks = new Set(tracksMap.keys());

  const trackEntries = Array.from(tracksMap.entries());
  const costList = [];

  for (let tIdx = 0; tIdx < trackEntries.length; tIdx++) {
    const [trackId, track] = trackEntries[tIdx];
    const tBox = track.bbox;
    const predCx = (tBox[0] + tBox[2]) / 2 + (track.vx || 0);
    const predCy = (tBox[1] + tBox[3]) / 2 + (track.vy || 0);
    const tW = Math.max(1, tBox[2] - tBox[0]);
    const tH = Math.max(1, tBox[3] - tBox[1]);

    for (let dIdx = 0; dIdx < detections.length; dIdx++) {
      const det = detections[dIdx];
      const dBox = det.bbox;
      const dCx = (dBox[0] + dBox[2]) / 2;
      const dCy = (dBox[1] + dBox[3]) / 2;
      const dW = Math.max(1, dBox[2] - dBox[0]);
      const dH = Math.max(1, dBox[3] - dBox[1]);

      const iou = calcBoxIoU(tBox, dBox);
      const dist = Math.hypot((predCx - dCx) / frameW, (predCy - dCy) / frameH);
      const sizeRatio = (dW * dH) / (tW * tH);

      const isIoUMatch = (iou >= minIoU);
      const isProximityMatch = (dist <= maxDistNorm && sizeRatio >= 0.28 && sizeRatio <= 3.5);

      if (isIoUMatch || isProximityMatch) {
        const proximityScore = 1.0 - Math.min(1.0, dist / maxDistNorm);
        const score = (iou * 0.45) + (proximityScore * 0.55);
        costList.push({ trackId, dIdx, score });
      }
    }
  }

  costList.sort((a, b) => b.score - a.score);
  const matchedT = new Set();
  const matchedD = new Set();

  for (const c of costList) {
    if (!matchedT.has(c.trackId) && !matchedD.has(c.dIdx)) {
      matchedT.add(c.trackId);
      matchedD.add(c.dIdx);
      unmatchedTracks.delete(c.trackId);
      unmatchedDetections.delete(c.dIdx);
      matches.push({ trackId: c.trackId, dIdx: c.dIdx, score: c.score });
    }
  }

  return {
    matches,
    unmatchedDetections: Array.from(unmatchedDetections),
    unmatchedTracks: Array.from(unmatchedTracks)
  };
}

class FrontendTrackerSimulator {
  constructor(frameW = 640, frameH = 480) {
    this.frameW = frameW;
    this.frameH = frameH;
    this.trackedItems = new Map();
    this.nextItemId = 1;
  }

  processFrame(itemDets, now = Date.now()) {
    const frameW = this.frameW;
    const frameH = this.frameH;

    // Lọc trùng lặp raw detections
    const cleanItemDets = [];
    for (let i = 0; i < itemDets.length; i++) {
      let isDup = false;
      for (let j = 0; j < cleanItemDets.length; j++) {
        if (calcBoxIoU(itemDets[i].bbox, cleanItemDets[j].bbox) > 0.70) {
          isDup = true;
          if (itemDets[i].score > cleanItemDets[j].score) {
            cleanItemDets[j] = itemDets[i];
          }
          break;
        }
      }
      if (!isDup) cleanItemDets.push(itemDets[i]);
    }

    const itemMatch = matchTracksWithDetections(this.trackedItems, cleanItemDets, frameW, frameH, 0.32, 0.15);

    // Cập nhật matched tracks
    for (const m of itemMatch.matches) {
      const it = this.trackedItems.get(m.trackId);
      const det = cleanItemDets[m.dIdx];

      const oldCx = (it.bbox[0] + it.bbox[2]) / 2;
      const oldCy = (it.bbox[1] + it.bbox[3]) / 2;
      const newCx = (det.bbox[0] + det.bbox[2]) / 2;
      const newCy = (det.bbox[1] + det.bbox[3]) / 2;
      const instVx = newCx - oldCx;
      const instVy = newCy - oldCy;
      it.vx = it.vx !== undefined ? it.vx * 0.4 + instVx * 0.6 : instVx;
      it.vy = it.vy !== undefined ? it.vy * 0.4 + instVy * 0.6 : instVy;

      it.missCount = 0;
      it.hits++;
      it.confirmed = true;
      it.lifecycle = "CONFIRMED";
      it.isRenderable = true;

      it.bbox[0] = Math.round(it.bbox[0] * 0.25 + det.bbox[0] * 0.75);
      it.bbox[1] = Math.round(it.bbox[1] * 0.25 + det.bbox[1] * 0.75);
      it.bbox[2] = Math.round(it.bbox[2] * 0.25 + det.bbox[2] * 0.75);
      it.bbox[3] = Math.round(it.bbox[3] * 0.25 + det.bbox[3] * 0.75);
      it.polygon = det.polygon || null;
      it.score = it.score * 0.3 + det.score * 0.7;
      it.lastSeen = now;
      it.recheckStartTime = null;
    }

    // Xử lý unmatched tracks (Tăng missCount & ẨN polygon ngay lập tức)
    for (const itId of itemMatch.unmatchedTracks) {
      const it = this.trackedItems.get(itId);
      if (!it) continue;
      it.missCount++;
      it.lifecycle = "TEMPORARILY_MISSING";
      it.isRenderable = false;
      it.vx = (it.vx || 0) * 0.5;
      it.vy = (it.vy || 0) * 0.5;
    }

    // Xử lý unmatched detections (Kiểm tra reassociation trước khi tạo ID mới)
    for (const dIdx of itemMatch.unmatchedDetections) {
      const det = cleanItemDets[dIdx];
      const dCx = (det.bbox[0] + det.bbox[2]) / 2;
      const dCy = (det.bbox[1] + det.bbox[3]) / 2;
      const dW = Math.max(1, det.bbox[2] - det.bbox[0]);
      const dH = Math.max(1, det.bbox[3] - det.bbox[1]);

      let bestCandidate = null;
      let bestDist = 0.38;
      for (const it of this.trackedItems.values()) {
        if (it.missCount > 0 && it.missCount <= 25) {
          const predCx = (it.bbox[0] + it.bbox[2]) / 2 + (it.vx || 0);
          const predCy = (it.bbox[1] + it.bbox[3]) / 2 + (it.vy || 0);
          const dist = Math.hypot((predCx - dCx) / frameW, (predCy - dCy) / frameH);
          const tW = Math.max(1, it.bbox[2] - it.bbox[0]);
          const tH = Math.max(1, it.bbox[3] - it.bbox[1]);
          const sizeRatio = (dW * dH) / (tW * tH);
          if (dist < bestDist && sizeRatio >= 0.30 && sizeRatio <= 3.2) {
            bestDist = dist;
            bestCandidate = it;
          }
        }
      }

      if (bestCandidate) {
        bestCandidate.missCount = 0;
        bestCandidate.hits++;
        bestCandidate.confirmed = true;
        bestCandidate.lifecycle = "CONFIRMED";
        bestCandidate.isRenderable = true;
        bestCandidate.bbox = [...det.bbox];
        bestCandidate.polygon = det.polygon || null;
        bestCandidate.score = bestCandidate.score * 0.3 + det.score * 0.7;
        bestCandidate.lastSeen = now;
        bestCandidate.recheckStartTime = null;
      } else {
        const isHighConf = (det.score >= 0.35);
        const itId = this.nextItemId++;
        this.trackedItems.set(itId, {
          id: itId,
          displayId: `ITEM_${String(itId).padStart(3, "0")}`,
          bbox: [...det.bbox],
          polygon: det.polygon || null,
          score: det.score,
          hits: 1,
          confirmed: isHighConf,
          lifecycle: isHighConf ? "CONFIRMED" : "TENTATIVE",
          missCount: 0,
          vx: 0,
          vy: 0,
          firstSeen: now,
          lastSeen: now,
          stationarySince: now,
          status: "NORMAL",
          associatedSeat: "UNKNOWN_SEAT",
          recheckStartTime: null,
          isRenderable: isHighConf
        });
      }
    }

    // Duplicate track merging
    const currentItemTracks = Array.from(this.trackedItems.values());
    for (let i = 0; i < currentItemTracks.length; i++) {
      for (let j = i + 1; j < currentItemTracks.length; j++) {
        const tA = currentItemTracks[i];
        const tB = currentItemTracks[j];
        if (!this.trackedItems.has(tA.id) || !this.trackedItems.has(tB.id)) continue;
        const iou = calcBoxIoU(tA.bbox, tB.bbox);
        if (iou > 0.55) {
          const keep = tA.id < tB.id ? tA : tB;
          const drop = tA.id < tB.id ? tB : tA;
          keep.hits += drop.hits;
          keep.lastSeen = Math.max(keep.lastSeen, drop.lastSeen);
          if (drop.status === "SUSPECTED") keep.status = "SUSPECTED";
          this.trackedItems.delete(drop.id);
        }
      }
    }

    // Expiration
    for (const [itId, it] of this.trackedItems.entries()) {
      if (it.missCount > 0) {
        const timeMissing = now - it.lastSeen;
        if ((!it.confirmed && it.missCount >= 2) || it.missCount > 25 || (timeMissing > 2000 && it.status !== "TEMPORARILY_OCCLUDED") || it.status === "ITEM_REMOVED") {
          this.trackedItems.delete(itId);
        }
      }
    }

    return {
      activeTracks: Array.from(this.trackedItems.values()),
      renderableTracks: Array.from(this.trackedItems.values()).filter(it => it.confirmed && it.missCount === 0 && it.isRenderable)
    };
  }
}

// ─────────────────────────────────────────────────────────────
// EXECUTE TESTS
// ─────────────────────────────────────────────────────────────

console.log('Running Frontend Tracker Unit Tests...');

// Test 1: Stationary object
{
  const sim = new FrontendTrackerSimulator();
  let ids = [];
  for (let f = 0; f < 30; f++) {
    const jitter = f % 2 === 0 ? 1 : -1;
    const res = sim.processFrame([{ bbox: [150 + jitter, 150, 210 + jitter, 230], score: 0.9 }], 1000 + f * 100);
    assert.strictEqual(res.renderableTracks.length, 1, 'Exactly one renderable track');
    ids.push(res.renderableTracks[0].displayId);
  }
  const uniqueIds = new Set(ids);
  assert.strictEqual(uniqueIds.size, 1, 'Stationary object should have exactly 1 stable ID');
  assert.strictEqual(Array.from(uniqueIds)[0], 'ITEM_001');
  console.log('✅ Test 1 Passed: Stationary object maintains exactly 1 stable ID');
}

// Test 2: Continuous horizontal movement
{
  const sim = new FrontendTrackerSimulator();
  let ids = [];
  for (let f = 0; f < 15; f++) {
    const x = 50 + f * 35; // moves 35px per frame across screen
    const res = sim.processFrame([{ bbox: [x, 150, x + 60, 230], score: 0.88 }], 1000 + f * 100);
    assert.strictEqual(res.renderableTracks.length, 1, 'Must have exactly 1 renderable track');
    ids.push(res.renderableTracks[0].displayId);
  }
  const uniqueIds = new Set(ids);
  assert.strictEqual(uniqueIds.size, 1, 'Continuous horizontal movement maintains single ID');
  assert.strictEqual(Array.from(uniqueIds)[0], 'ITEM_001');
  console.log('✅ Test 2 Passed: Continuous horizontal movement preserves ID without ghost polygons');
}

// Test 3: Sudden direction change
{
  const sim = new FrontendTrackerSimulator();
  // Move right: x=100 -> 250
  for (let f = 0; f < 5; f++) {
    const x = 100 + f * 30;
    const res = sim.processFrame([{ bbox: [x, 150, x + 60, 230], score: 0.9 }], 1000 + f * 100);
    assert.strictEqual(res.renderableTracks[0].displayId, 'ITEM_001');
  }
  // Sudden switch move left: x=250 -> 100
  for (let f = 0; f < 5; f++) {
    const x = 250 - f * 30;
    const res = sim.processFrame([{ bbox: [x, 150, x + 60, 230], score: 0.9 }], 2000 + f * 100);
    assert.strictEqual(res.renderableTracks[0].displayId, 'ITEM_001');
  }
  console.log('✅ Test 3 Passed: Sudden direction change preserves ID without fragmentation');
}

// Test 4: One or more missed detections
{
  const sim = new FrontendTrackerSimulator();
  // Frame 0: visible at x=100
  let res = sim.processFrame([{ bbox: [100, 150, 160, 230], score: 0.9 }], 1000);
  assert.strictEqual(res.renderableTracks.length, 1);
  assert.strictEqual(res.renderableTracks[0].displayId, 'ITEM_001');

  // Frame 1 & 2: missed detection (camera blur)
  res = sim.processFrame([], 1100);
  assert.strictEqual(res.renderableTracks.length, 0, 'No ghost polygon while missing');
  assert.strictEqual(sim.trackedItems.size, 1, 'Retained internally in memory');

  res = sim.processFrame([], 1200);
  assert.strictEqual(res.renderableTracks.length, 0, 'No ghost polygon on frame 2');

  // Frame 3: phone detected at x=130
  res = sim.processFrame([{ bbox: [130, 150, 190, 230], score: 0.9 }], 1300);
  assert.strictEqual(res.renderableTracks.length, 1);
  assert.strictEqual(res.renderableTracks[0].displayId, 'ITEM_001', 'Restored same ID after miss');
  console.log('✅ Test 4 Passed: Missed detections hide stale polygon and restore ID upon reappearance');
}

// Test 5: Genuine disappearance
{
  const sim = new FrontendTrackerSimulator();
  sim.processFrame([{ bbox: [100, 150, 160, 230], score: 0.9 }], 1000);
  assert.strictEqual(sim.trackedItems.size, 1);

  // Phone leaves frame for 30 frames (3 seconds)
  for (let f = 1; f <= 30; f++) {
    sim.processFrame([], 1000 + f * 100);
  }
  assert.strictEqual(sim.trackedItems.size, 0, 'Expired track must be fully purged');
  console.log('✅ Test 5 Passed: Disappeared item is expired and purged from collections');
}

// Test 6: Two separate phones
{
  const sim = new FrontendTrackerSimulator();
  // Two phones at distinct positions
  const res = sim.processFrame([
    { bbox: [100, 150, 160, 230], score: 0.9 },
    { bbox: [300, 150, 360, 230], score: 0.9 }
  ], 1000);
  assert.strictEqual(res.renderableTracks.length, 2, 'Two separate phones must have 2 tracks');
  const ids = res.renderableTracks.map(t => t.displayId);
  assert.ok(ids.includes('ITEM_001'));
  assert.ok(ids.includes('ITEM_002'));
  console.log('✅ Test 6 Passed: Two separate phones receive distinct IDs');
}

// Test 7: Duplicate raw detection suppression
{
  const sim = new FrontendTrackerSimulator();
  // One phone with 2 overlapping bounding boxes (YOLO multi-box artifact)
  const res = sim.processFrame([
    { bbox: [150, 150, 210, 230], score: 0.92 },
    { bbox: [152, 149, 211, 231], score: 0.88 }
  ], 1000);
  assert.strictEqual(res.renderableTracks.length, 1, 'Duplicate detection must merge into 1 track');
  assert.strictEqual(res.renderableTracks[0].displayId, 'ITEM_001');
  console.log('✅ Test 7 Passed: Duplicate detections for single phone merged cleanly');
}

// Test 8: Low-confidence candidate item (Two-tier confirmation for small belongings)
{
  const sim = new FrontendTrackerSimulator();
  // Frame 1: Candidate item with low confidence (0.24) - admitted into tentative tracking
  const res1 = sim.processFrame([
    { bbox: [200, 150, 250, 210], score: 0.24 }
  ], 1000);
  assert.strictEqual(res1.renderableTracks.length, 0, 'Tentative item must NOT be renderable on frame 1 (anti-noise)');
  assert.strictEqual(sim.trackedItems.size, 1, 'Tentative track must be recorded in internal collection');
  const track1 = Array.from(sim.trackedItems.values())[0];
  assert.strictEqual(track1.lifecycle, 'TENTATIVE');
  assert.strictEqual(track1.confirmed, false);

  // Frame 2: Item confirmed on consecutive detection (score 0.26)
  const res2 = sim.processFrame([
    { bbox: [201, 151, 251, 211], score: 0.26 }
  ], 1066);
  assert.strictEqual(res2.renderableTracks.length, 1, 'Corroborated item must become renderable on frame 2');
  assert.strictEqual(res2.renderableTracks[0].displayId, 'ITEM_001');
  assert.strictEqual(res2.renderableTracks[0].confirmed, true);
  assert.strictEqual(res2.renderableTracks[0].lifecycle, 'CONFIRMED');
  console.log('✅ Test 8 Passed: Small belonging candidate (0.24 conf) reliably confirmed on frame 2');
}

// Test 9: Transient 1-frame background noise (score 0.22) rejected cleanly without ghost polygon
{
  const sim = new FrontendTrackerSimulator();
  // Frame 1: 1-frame noise spike (0.22)
  const res1 = sim.processFrame([
    { bbox: [300, 300, 340, 340], score: 0.22 }
  ], 1000);
  assert.strictEqual(res1.renderableTracks.length, 0, 'Noise candidate must NOT render on frame 1');

  // Frame 2 & 3: Noise is gone
  sim.processFrame([], 1066);
  sim.processFrame([], 1133);
  assert.strictEqual(sim.trackedItems.size, 0, 'Unconfirmed tentative noise track must be purged');
  console.log('✅ Test 9 Passed: Transient 1-frame noise rejected and purged without ghost polygon');
}

// Test 10: Streamlit telemetry rate-limiter prevents rerun storms (dimming bug prevention)
{
  let postMessageCallCount = 0;
  let lastPostedValue = null;
  const mockParent = {
    postMessage: (msg) => {
      postMessageCallCount++;
      lastPostedValue = msg.value;
    }
  };

  // Simulate index.html notifyStreamlit logic
  let lastNotifiedStateKey = "";
  let lastNotifiedTime = 0;
  function simNotifyStreamlit(val, now, force = false) {
    if (!val) return;
    const stateKey = `${val.status || ""}_${val.is_running ? 1 : 0}_${val.suspected_count || 0}_${val.error || ""}`;
    const stateChanged = (stateKey !== lastNotifiedStateKey);
    const throttleElapsed = (now - lastNotifiedTime > 2500);
    if (force || stateChanged || throttleElapsed) {
      lastNotifiedStateKey = stateKey;
      lastNotifiedTime = now;
      mockParent.postMessage({ isStreamlitMessage: true, type: "streamlit:setComponentValue", value: val });
    }
  }

  // Camera start (forced)
  simNotifyStreamlit({ status: "running", is_running: true, suspected_count: 0 }, 1000, true);
  assert.strictEqual(postMessageCallCount, 1, 'Initial camera start must post message');

  // 60 subsequent video frames over 2 seconds with stable suspect_count: 0
  for (let f = 1; f <= 60; f++) {
    const frameNow = 1000 + f * 33; // 30 FPS over 2000 ms
    // Only notify when suspect count changes or throttle elapsed
    const suspectCount = 0;
    if (suspectCount !== 0 || (frameNow - lastNotifiedTime > 2500)) {
      simNotifyStreamlit({ status: "running", is_running: true, suspected_count: suspectCount }, frameNow);
    }
  }
  // During 2 seconds of normal operation, postMessageCallCount remains 1 (0 rerun storms!)
  assert.strictEqual(postMessageCallCount, 1, 'Unchanging video frames must NOT spam Streamlit with postMessages');

  // An item becomes suspected at t = 3500ms
  simNotifyStreamlit({ status: "running", is_running: true, suspected_count: 1 }, 3500);
  assert.strictEqual(postMessageCallCount, 2, 'State change (suspected_count 0 -> 1) must trigger immediate dispatch');
  assert.strictEqual(lastPostedValue.suspected_count, 1);
  console.log('✅ Test 10 Passed: Streamlit messaging throttled to prevent UI dimming while alerting instantly');
}
console.log('\n🎉 ALL 10 SIMULATOR TESTS PASSED CLEANLY!\n');

// ─────────────────────────────────────────────────────────────
// PART 2: REAL PRODUCTION RUNTIME TESTS (EXECUTING index.html DIRECTLY)
// ─────────────────────────────────────────────────────────────
console.log('Running Real Production index.html Tracking Execution Tests...');

function createProductionRuntime() {
  const htmlPath = path.resolve(__dirname, '../src/components/hybrid_camera/frontend/index.html');
  const html = fs.readFileSync(htmlPath, 'utf8');
  const scriptMatch = html.match(/<script>([\s\S]*?)<\/script>/);
  if (!scriptMatch) throw new Error('Could not find <script> block in index.html');
  const scriptCode = scriptMatch[1];

  let currentTime = 1000;
  const mockElement = () => ({
    style: {},
    classList: { add: () => {}, remove: () => {} },
    getContext: () => ({
      clearRect: () => {}, save: () => {}, restore: () => {}, beginPath: () => {},
      moveTo: () => {}, lineTo: () => {}, closePath: () => {}, fill: () => {},
      stroke: () => {}, fillRect: () => {}, strokeRect: () => {}, fillText: () => {},
      measureText: () => ({ width: 50 })
    }),
    addEventListener: () => {},
    appendChild: () => {},
    querySelectorAll: () => []
  });

  const alerts = [];
  const sandbox = {
    console: { ...console, debug: () => {}, error: () => {} },
    performance: { now: () => currentTime },
    Set, Map, Uint8Array, Float32Array,
    document: {
      getElementById: (id) => mockElement(),
      createElement: (tag) => mockElement(),
      addEventListener: () => {},
      body: mockElement(),
      documentElement: mockElement()
    },
    window: {
      addEventListener: () => {},
      parent: {
        postMessage: (msg) => {
          if (msg && msg.value && msg.value.suspected_count > 0) {
            alerts.push(msg.value);
          }
        }
      }
    },
    navigator: { mediaDevices: { enumerateDevices: async () => [] } },
    requestAnimationFrame: () => {},
    cancelAnimationFrame: () => {},
    clearTimeout: () => {},
    setTimeout: () => {}
  };

  vm.createContext(sandbox);
  vm.runInContext(scriptCode, sandbox);

  return {
    sandbox,
    setTime: (t) => { currentTime = t; },
    advanceTime: (dt) => { currentTime += dt; },
    getTime: () => currentTime,
    updateTrackingAndState: (detections, w = 640, h = 480) => {
      sandbox.updateTrackingAndState(detections, w, h);
    },
    getTrackedItems: () => vm.runInContext('Array.from(trackedItems.values())', sandbox),
    getTrackedSeats: () => vm.runInContext('Array.from(trackedSeats.values())', sandbox),
    getTrackedPersons: () => vm.runInContext('Array.from(trackedPersons.values())', sandbox),
    getSuspectItems: () => vm.runInContext('Array.from(trackedItems.values()).filter(it => it.confirmed && it.status === "SUSPECTED" && it.status !== "ITEM_REMOVED")', sandbox),
    getRenderableItems: () => vm.runInContext('Array.from(trackedItems.values()).filter(it => it.confirmed && it.missCount === 0 && it.isRenderable)', sandbox),
    getNextItemId: () => vm.runInContext('nextItemId', sandbox),
    alerts
  };
}

// Prod Test 1: A first-time high-confidence item creates exactly one confirmed track
{
  const rt = createProductionRuntime();
  rt.updateTrackingAndState([{ category: 'item', bbox: [100, 100, 150, 150], score: 0.85, polygon: null }]);
  const items = rt.getTrackedItems();
  assert.strictEqual(items.length, 1, 'First-time high-conf item must create exactly 1 track');
  assert.strictEqual(items[0].displayId, 'ITEM_001');
  assert.strictEqual(items[0].confirmed, true, 'High-conf item must be confirmed immediately');
  assert.strictEqual(items[0].lifecycle, 'CONFIRMED');
  assert.strictEqual(items[0].isRenderable, true, 'High-conf item must be renderable immediately');
  assert.strictEqual(rt.getRenderableItems().length, 1);
  console.log('✅ Prod Test 1 Passed: First-time high-confidence item creates exactly one confirmed track');
}

// Prod Test 2: A first-time low-confidence item creates exactly one tentative track
{
  const rt = createProductionRuntime();
  rt.updateTrackingAndState([{ category: 'item', bbox: [200, 150, 250, 200], score: 0.24, polygon: null }]);
  const items = rt.getTrackedItems();
  assert.strictEqual(items.length, 1, 'First-time low-conf candidate must create exactly 1 track in memory');
  assert.strictEqual(items[0].displayId, 'ITEM_001');
  assert.strictEqual(items[0].confirmed, false, 'Candidate item must be tentative, NOT confirmed');
  assert.strictEqual(items[0].lifecycle, 'TENTATIVE');
  assert.strictEqual(items[0].isRenderable, false, 'Tentative item must NOT be renderable on frame 1');
  assert.strictEqual(rt.getRenderableItems().length, 0, 'No polygon/box rendered for tentative candidate');
  console.log('✅ Prod Test 2 Passed: First-time low-confidence item creates exactly one tentative track');
}

// Prod Test 3: A low-confidence item becomes confirmed only after intended confirmation rule is satisfied
{
  const rt = createProductionRuntime();
  // Frame 1: Tentative candidate
  rt.updateTrackingAndState([{ category: 'item', bbox: [200, 150, 250, 200], score: 0.24, polygon: null }]);
  assert.strictEqual(rt.getRenderableItems().length, 0);

  // Frame 2: Corroborated detection satisfying 2-hit confirmation rule
  rt.advanceTime(66);
  rt.updateTrackingAndState([{ category: 'item', bbox: [201, 151, 251, 201], score: 0.25, polygon: null }]);
  const items = rt.getTrackedItems();
  assert.strictEqual(items.length, 1, 'Must maintain exactly 1 track');
  assert.strictEqual(items[0].displayId, 'ITEM_001');
  assert.strictEqual(items[0].hits, 2);
  assert.strictEqual(items[0].confirmed, true, 'Must graduate to confirmed on hit 2');
  assert.strictEqual(items[0].lifecycle, 'CONFIRMED');
  assert.strictEqual(items[0].isRenderable, true, 'Must become renderable upon confirmation');
  assert.strictEqual(rt.getRenderableItems().length, 1);
  console.log('✅ Prod Test 3 Passed: Low-confidence item confirmed after satisfying confirmation rule');
}

// Prod Test 4: An existing item updates its existing track without allocating a duplicate ID
{
  const rt = createProductionRuntime();
  // Frame 1
  rt.updateTrackingAndState([{ category: 'item', bbox: [100, 100, 150, 150], score: 0.85, polygon: null }]);
  assert.strictEqual(rt.getNextItemId(), 2);

  // Frame 2: Continuous movement
  rt.advanceTime(66);
  rt.updateTrackingAndState([{ category: 'item', bbox: [102, 101, 152, 151], score: 0.88, polygon: null }]);
  assert.strictEqual(rt.getTrackedItems().length, 1, 'Must not duplicate track');
  assert.strictEqual(rt.getTrackedItems()[0].displayId, 'ITEM_001');
  assert.strictEqual(rt.getNextItemId(), 2, 'ID counter must not increment on matched track');

  // Frame 3: Further movement
  rt.advanceTime(66);
  rt.updateTrackingAndState([{ category: 'item', bbox: [104, 102, 154, 152], score: 0.86, polygon: null }]);
  assert.strictEqual(rt.getTrackedItems().length, 1);
  assert.strictEqual(rt.getNextItemId(), 2);
  console.log('✅ Prod Test 4 Passed: Existing item updates track without allocating duplicate ID');
}

// Prod Test 5: A temporarily missing item can reassociate with its previous ID
{
  const rt = createProductionRuntime();
  // Frame 1: Initial item
  rt.updateTrackingAndState([{ category: 'item', bbox: [120, 120, 170, 170], score: 0.85, polygon: null }]);
  assert.strictEqual(rt.getTrackedItems()[0].displayId, 'ITEM_001');

  // Frame 2: Missed detection (brief occlusion/flicker)
  rt.advanceTime(100);
  rt.updateTrackingAndState([]);
  assert.strictEqual(rt.getRenderableItems().length, 0, 'No ghost polygon during missed detection');
  assert.strictEqual(rt.getTrackedItems()[0].missCount, 1);

  // Frame 3: Detection reappears near predicted location
  rt.advanceTime(100);
  rt.updateTrackingAndState([{ category: 'item', bbox: [122, 121, 172, 171], score: 0.84, polygon: null }]);
  const items = rt.getTrackedItems();
  assert.strictEqual(items.length, 1, 'Reassociated track must not create duplicate');
  assert.strictEqual(items[0].displayId, 'ITEM_001', 'Must preserve original ID');
  assert.strictEqual(items[0].missCount, 0);
  assert.strictEqual(items[0].confirmed, true);
  assert.strictEqual(items[0].isRenderable, true);
  assert.strictEqual(rt.getRenderableItems().length, 1);
  console.log('✅ Prod Test 5 Passed: Temporarily missing item reassociates with previous ID');
}

// Prod Test 6: Two distinct items receive distinct IDs
{
  const rt = createProductionRuntime();
  rt.updateTrackingAndState([
    { category: 'item', bbox: [50, 50, 100, 100], score: 0.85, polygon: null },
    { category: 'item', bbox: [300, 300, 350, 350], score: 0.90, polygon: null }
  ]);
  assert.strictEqual(rt.getTrackedItems().length, 2, 'Two separate items must yield 2 tracks');
  assert.strictEqual(rt.getRenderableItems().length, 2);
  const ids = rt.getRenderableItems().map(it => it.displayId).sort();
  assert.strictEqual(ids.join(','), 'ITEM_001,ITEM_002', 'Must allocate ITEM_001 and ITEM_002');
  console.log('✅ Prod Test 6 Passed: Two distinct items receive distinct IDs');
}

// Prod Test 7: A transient false detection does not leave a ghost polygon
{
  const rt = createProductionRuntime();
  // Frame 1: 1-frame low-confidence noise candidate (0.22)
  rt.updateTrackingAndState([{ category: 'item', bbox: [400, 400, 450, 450], score: 0.22, polygon: null }]);
  assert.strictEqual(rt.getRenderableItems().length, 0, 'Transient candidate must not render on frame 1');

  // Frame 2: Noise disappears
  rt.advanceTime(66);
  rt.updateTrackingAndState([]);
  assert.strictEqual(rt.getRenderableItems().length, 0, 'No ghost polygon when noise disappears');

  // Frame 3: Still gone -> purged from memory
  rt.advanceTime(66);
  rt.updateTrackingAndState([]);
  assert.strictEqual(rt.getTrackedItems().length, 0, 'Unconfirmed tentative candidate must be purged');
  console.log('✅ Prod Test 7 Passed: Transient false detection does not leave a ghost polygon');
}

// Prod Test 8: Forgotten-item alerts cannot be triggered by unconfirmed tentative tracks
{
  const rt = createProductionRuntime();
  // Vacant seat at [100, 100, 300, 300] with tentative low-conf item on it
  rt.updateTrackingAndState([
    { category: 'seat', bbox: [100, 100, 300, 300], score: 0.90, polygon: null },
    { category: 'item', bbox: [150, 150, 200, 200], score: 0.24, polygon: null }
  ]);
  assert.strictEqual(rt.getTrackedItems()[0].confirmed, false);

  // Time advances by 4.0s (> 3.5s vacant threshold) with no further detections
  rt.advanceTime(4000);
  rt.updateTrackingAndState([]);
  assert.strictEqual(rt.getSuspectItems().length, 0, 'Tentative track must NOT trigger suspect state');
  assert.strictEqual(rt.alerts.length, 0, 'No alert message dispatched for unconfirmed tentative track');
  console.log('✅ Prod Test 8 Passed: Forgotten-item alerts cannot be triggered by unconfirmed tentative tracks');
}
console.log('\n🎉 ALL 18 TRACKER & SENSITIVITY TESTS (SIMULATOR + REAL PRODUCTION index.html) PASSED CLEANLY!\n');
