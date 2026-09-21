// GPUFree / selkies 信令探针
// 用法: node probe_rdp.js "https://<实例地址>:8443/"
const url = process.argv[2];
if (!url) {
  console.error('用法: node probe_rdp.js "https://host:8443/"');
  process.exit(1);
}
const wsUrl = url.replace(/^https:/, 'wss:').replace(/^http:/, 'ws:') + (url.endsWith('/') ? '' : '/') + 'webrtc/signalling/';
console.log('HTTP page:', url);
console.log('WS path:  ', wsUrl);

fetch(url).then(async (r) => {
  const t = await r.text();
  console.log('HTTP', r.status, 'title=', (t.match(/<title>([^<]+)/) || [])[1]);
}).catch((e) => console.log('HTTP ERR', e.message));

const ws = new WebSocket(wsUrl);
const t0 = Date.now();
ws.onopen = () => {
  console.log(`+${Date.now() - t0}ms WS OPEN`);
  const meta = Buffer.from(JSON.stringify({ res: '1600x900', scale: 1 })).toString('base64');
  const msg = `HELLO 1 ${meta}`;
  console.log('SEND', msg);
  ws.send(msg);
};
ws.onmessage = (ev) => console.log('RECV', String(ev.data).slice(0, 500));
ws.onerror = () => console.log('WS ERROR');
ws.onclose = (ev) => {
  console.log(`+${Date.now() - t0}ms CLOSE code=${ev.code} reason=${ev.reason || '(empty)'} clean=${ev.wasClean}`);
  process.exit(0);
};
setTimeout(() => {
  console.log('timeout 8s');
  try { ws.close(); } catch {}
  process.exit(0);
}, 8000);
