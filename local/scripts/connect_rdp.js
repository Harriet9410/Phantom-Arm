const { chromium } = require('playwright-core');
const fs = require('fs');

const URL = 'https://wm0ih8p9-f5wasg41-3000.zj02restapi.gpufree.cn:8443/';
const LOG = 'D:/MYCODE/TCEI/local/logs';

(async () => {
  fs.mkdirSync(LOG, { recursive: true });
  console.log('Launching Chrome...');
  const browser = await chromium.launch({
    headless: false,
    channel: 'chrome',
    args: [
      '--autoplay-policy=no-user-gesture-required',
      '--ignore-certificate-errors',
    ],
  });
  const context = await browser.newContext({
    viewport: { width: 1600, height: 900 },
    ignoreHTTPSErrors: true,
  });
  const page = await context.newPage();

  page.on('websocket', (ws) => {
    if (/signalling/i.test(ws.url())) {
      console.log('WS:', ws.url());
      ws.on('framesent', ({ payload }) => console.log('SENT', String(payload).slice(0, 120)));
      ws.on('framereceived', ({ payload }) => console.log('RECV', String(payload).slice(0, 200)));
      ws.on('close', () => console.log('WS CLOSE'));
    }
  });

  console.log('Opening', URL);
  await page.goto(URL, { waitUntil: 'domcontentloaded', timeout: 60000 });
  console.log('Title:', await page.title());

  let connected = false;
  for (let i = 0; i < 24; i++) {
    await page.waitForTimeout(5000);
    const s = await page.evaluate(() => {
      const t = document.body.innerText;
      const v = document.querySelector('#stream') || document.querySelector('video');
      return {
        peer: (t.match(/Peer connection state:\s*(\S+)/) || [])[1],
        videoW: v ? v.videoWidth : 0,
        videoH: v ? v.videoHeight : 0,
        loading: Array.from(document.querySelectorAll('.loading-text')).map(e => e.textContent.trim()).filter(Boolean),
      };
    });
    console.log(`[${i * 5}s]`, JSON.stringify(s));
    await page.screenshot({ path: `${LOG}/rdp_${i}.png` });

    // click start button if present
    const startBtn = await page.$('.loading button');
    if (startBtn) {
      const txt = (await startBtn.textContent()) || '';
      if (/start/i.test(txt)) {
        console.log('Clicking start...');
        await startBtn.click();
      }
    }
    if (s.peer === 'connected' && s.videoW > 0) {
      connected = true;
      console.log('CONNECTED');
      break;
    }
  }

  await page.screenshot({ path: `${LOG}/rdp_connected.png` });
  console.log('connected=', connected, 'shot=', `${LOG}/rdp_connected.png`);

  if (connected) {
    // keep browser open for control session
    console.log('BROWSER_READY for control. Holding 10 minutes...');
    // expose a marker file so we know process is alive
    fs.writeFileSync(`${LOG}/cdp_session.txt`, String(process.pid));
    await page.waitForTimeout(10 * 60 * 1000);
  } else {
    console.log('NOT_CONNECTED, closing in 30s...');
    await page.waitForTimeout(30000);
  }
  await browser.close();
})().catch((e) => {
  console.error('ERROR', e);
  process.exit(1);
});
