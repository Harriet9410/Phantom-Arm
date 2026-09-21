// 通过 Chrome 远程调试端口接管 GPUFree 远程桌面
// 前提: Chrome 已用 --remote-debugging-port=9222 启动并打开远程桌面页
// 用法: node cdp_attach.js
const { chromium } = require('playwright-core');
const path = require('path');

const CDP = process.env.CDP_URL || 'http://127.0.0.1:9222';
const SHOT = process.env.SCREENSHOT || path.join(__dirname, '..', 'logs', 'takeover.png');
const URL_HINT = /gpufree|zj02restapi|8443/i;

(async () => {
  const browser = await chromium.connectOverCDP(CDP);
  let page = null;
  for (const ctx of browser.contexts()) {
    for (const p of ctx.pages()) {
      console.log('PAGE:', p.url());
      if (URL_HINT.test(p.url())) { page = p; break; }
    }
    if (page) break;
  }
  if (!page) {
    throw new Error('未找到 GPUFree 标签页。请用调试模式 Chrome 打开远程桌面 URL。');
  }

  const state = await page.evaluate(() => {
    const t = document.body.innerText;
    const video = document.querySelector('#stream') || document.querySelector('video');
    return {
      title: document.title,
      peer: (t.match(/Peer connection state:\s*(\S+)/) || [])[1] || null,
      videoW: video ? video.videoWidth : null,
      videoH: video ? video.videoHeight : null,
      loading: Array.from(document.querySelectorAll('.loading-text')).map((e) => e.textContent.trim()),
    };
  });
  console.log('STATE:', JSON.stringify(state, null, 2));

  await page.screenshot({ path: SHOT });
  console.log('截图:', SHOT);

  if (state.peer !== 'connected' || !state.videoW) {
    console.warn('警告: WebRTC 未连接或无视频流，请先排查信令/实例状态');
  } else {
    console.log('接管条件就绪: peer=connected, video=', state.videoW + 'x' + state.videoH);
  }

  // 仅断开 CDP，不关闭用户的 Chrome
  browser.close();
})().catch((e) => {
  console.error('ERR', e.message || e);
  process.exit(1);
});
