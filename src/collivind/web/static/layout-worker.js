// The force layout off the main thread: at 4k nodes one tick costs about as
// much as a frame, so ticking here keeps drawing and panning at 60fps.
// The page owns alpha and the pinned (dragged) node; this owns nothing but
// the positions between ticks. One reply per request, so it never floods.

import { layoutTick } from "./layout.js";

let gen = 0;
let nodes = [];
let links = [];

self.onmessage = ({ data: m }) => {
  if (m.type === "graph") {
    gen = m.gen;
    nodes = Array.from({ length: m.degrees.length }, (_, i) => ({
      x: m.pos[i * 2], y: m.pos[i * 2 + 1], vx: 0, vy: 0, degree: m.degrees[i],
    }));
    links = [];
    for (let i = 0; i < m.links.length; i += 2) links.push({ source: nodes[m.links[i]], target: nodes[m.links[i + 1]] });
    return;
  }
  if (m.gen !== gen) return; // a request for a graph that has since been replaced
  let alpha = m.alpha;
  const pinned = m.pin ? nodes[m.pin.i] : null;
  if (pinned) Object.assign(pinned, { x: m.pin.x, y: m.pin.y, vx: 0, vy: 0 });
  for (let i = 0; i < m.ticks && alpha > 0.004; i++) {
    layoutTick(nodes, links, alpha, pinned);
    alpha *= 0.985;
  }
  const pos = new Float64Array(nodes.length * 2);
  nodes.forEach((n, i) => { pos[i * 2] = n.x; pos[i * 2 + 1] = n.y; });
  self.postMessage({ gen, pos, alpha }, [pos.buffer]);
};
