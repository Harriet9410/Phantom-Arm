const { chromium } = require('playwright-core');
const fs = require('fs');

const URL = 'https://wm0ih8p9-f5wasg41-3000.zj02restapi.gpufree.cn:8443/';
const LOG = 'D:/MYCODE/TCEI/local/logs';

(async () => {
  fs.mkdirSync(LOG, { recursive: true });
  const browser = await chromium.launch({
    headless: false,
    channel: 'chrome',
    args: ['--autoplay-policy=no-user-gesture-required', '--ignore-certificate-errors'],
  });
  const context = await browser.newContext({
    viewport: { width: 1600, height: 900 },
    ignoreHTTPSErrors: true,
  });
  const page = await context.newPage();

  console.log('goto', URL);
  await page.goto(URL, { waitUntil: 'domcontentloaded', timeout: 60000 });

  // wait up to 90s, try recovery actions
  for (let i = 0; i < 18; i++) {
    await page.waitForTimeout(5000);

    const s = await page.evaluate(() => {
      const t = document.body.innerText;
      const v = document.querySelector('#stream') || document.querySelector('video');
      const audio = document.querySelector('#audio_stream') || document.querySelector('audio');
      return {
        peerLine: (t.match(/Peer connection state:\s*([^\n]+)/) || [])[1] || null,
        videoW: v ? v.videoWidth : 0,
        videoH: v ? v.videoHeight : 0,
        vReady: v ? v.readyState : -1,
        vPaused: v ? v.paused : null,
        vErr: v && v.error ? v.error.code : null,
        aReady: audio ? audio.readyState : -1,
        loading: Array.from(document.querySelectorAll('.loading-text')).map(e => e.textContent.trim()).filter(Boolean),
        hasStart: !!document.querySelector('.loading button'),
        startText: document.querySelector('.loading button') ? document.querySelector('.loading button').textContent.trim() : null,
      };
    });
    console.log(`[${i * 5}s]`, JSON.stringify(s));

    await page.screenshot({ path: `${LOG}/ctrl_${i}.png` });

    // actions
    try {
      // click start if visible
      if (s.startText && /start/i.test(s.startText)) {
        console.log('click start');
        await page.click('.loading button');
      }
      // click video center to unlock autoplay / play
      await page.evaluate(() => {
        const v = document.querySelector('#stream') || document.querySelector('video');
        if (v && v.paused) v.play().catch(() => {});
      });
      // try app playStream if exists
      await page.evaluate(() => {
        const btns = Array.from(document.querySelectorAll('button'));
        const start = btns.find(b => /start/i.test(b.textContent || ''));
        if (start) start.click();
      });
    } catch (e) { console.log('action err', e.message); }

    if (s.videoW > 0 && s.vReady >= 2 && !s.loading.some(x => /Waiting|Connecting|Registering/i.test(x))) {
      console.log('LOOKS_READY');
      break;
    }
  }

  // open side drawer stats once
  try {
    await page.click('.fab-container');
    await page.waitForTimeout(1000);
    const drawer = await page.evaluate(() => document.body.innerText.slice(0, 1500));
    console.log('DRAWER:', drawer.replace(/\n+/g, ' | ').slice(0, 800));
    await page.screenshot({ path: `${LOG}/ctrl_drawer.png` });
  } catch (e) { console.log('drawer err', e.message); }

  await page.screenshot({ path: `${LOG}/ctrl_final.png` });
  console.log('final shot saved');
  console.log('HOLD 8 minutes for manual check');
  await page.waitForTimeout(8 * 60 * 1000);
  await browser.close();
})().catch((e) => { console.error(e); process.exit(1); });
