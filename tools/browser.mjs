/**
 * Find and launch a browser for the unattended runs in tools/: the GPU
 * self-test (browsertest.mjs) and the media capture rig (capture/record.mjs).
 *
 * Windowed (the default) runs the real browser on the real driver, which is
 * what those tools exist for. Under WSL the browser is a Windows executable
 * reached through /mnt/c, with its throwaway profile under %LOCALAPPDATA%.
 *
 * Headless runs Chrome or Chromium with WebGPU on SwiftShader. That is a
 * software device, so it says nothing about drivers, but it exercises Tint,
 * Dawn's validation and the whole compute pipeline with no GPU and no display,
 * which is what CI has. A SwiftShader device is lost as soon as it presents to
 * a WebGPU canvas, so headless covers compute and offscreen rendering only:
 * the self-test, the GPU-vs-CPU page and the capture rig. Headless Firefox is
 * no use here: it exposes navigator.gpu but returns no adapter.
 */

import { spawn, execFileSync } from 'node:child_process';
import { existsSync, mkdirSync, mkdtempSync, readdirSync, rmSync, writeFileSync } from 'node:fs';
import { homedir, tmpdir } from 'node:os';
import { join } from 'node:path';

const WINDOWS = [
  '/mnt/c/Program Files/Zen Browser/zen.exe',
  '/mnt/c/Program Files/Mozilla Firefox/firefox.exe',
  '/mnt/c/Program Files (x86)/Mozilla Firefox/firefox.exe',
  '/mnt/c/Program Files/Google/Chrome/Application/chrome.exe',
  '/mnt/c/Program Files/Microsoft/Edge/Application/msedge.exe',
  '/mnt/c/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',
];

const NATIVE_CHROMIUM = [
  '/usr/bin/google-chrome',
  '/usr/bin/google-chrome-stable',
  '/usr/bin/chromium',
  '/usr/bin/chromium-browser',
  '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  '/Applications/Chromium.app/Contents/MacOS/Chromium',
];

const NATIVE_GECKO = [
  '/usr/bin/firefox',
  '/Applications/Firefox.app/Contents/MacOS/firefox',
];

/** Chromium builds Playwright has downloaded, newest first. */
function playwrightChromium() {
  const roots = [
    process.env.PLAYWRIGHT_BROWSERS_PATH,
    join(homedir(), '.cache', 'ms-playwright'),
  ].filter(Boolean);
  const found = [];
  for (const root of roots) {
    if (!existsSync(root)) continue;
    for (const entry of readdirSync(root).sort().reverse()) {
      if (/^chromium-\d+$/.test(entry)) found.push(join(root, entry, 'chrome-linux', 'chrome'));
    }
  }
  return found;
}

export const isGecko = (browser) => /firefox|zen/i.test(browser);
const isWindowsExe = (browser) => browser.startsWith('/mnt/');

/**
 * @param {{ explicit?: string, headless?: boolean }} options
 * @returns {{ browser: string|null, looked: string[] }}
 */
export function findBrowser({ explicit, headless = false } = {}) {
  const looked = [
    explicit,
    process.env.BROWSER,
    ...(headless
      ? [...NATIVE_CHROMIUM, ...playwrightChromium()]
      : [...WINDOWS, ...NATIVE_GECKO, ...NATIVE_CHROMIUM, ...playwrightChromium()]),
  ].filter(Boolean);
  return { browser: looked.find((path) => existsSync(path)) ?? null, looked };
}

/** %LOCALAPPDATA% as both a Windows path and its WSL mount. */
function windowsLocalAppData() {
  const win = execFileSync('cmd.exe', ['/c', 'echo %LOCALAPPDATA%'], { encoding: 'utf8' })
    .trim()
    .replace(/\r/g, '');
  return { win, wsl: '/mnt/' + win[0].toLowerCase() + win.slice(2).replace(/\\/g, '/') };
}

const GECKO_PREFS = [
  // A fresh profile does not enable WebGPU. Without these the adapter is null.
  'user_pref("dom.webgpu.enabled", true);',
  'user_pref("dom.webgpu.workers.enabled", true);',
  'user_pref("gfx.webrender.all", true);',
  'user_pref("layers.acceleration.force-enabled", true);',
  // Keep the throwaway profile quiet: no first-run pages, no crash restore,
  // no telemetry prompts stealing focus from the test page.
  'user_pref("browser.shell.checkDefaultBrowser", false);',
  'user_pref("browser.aboutwelcome.enabled", false);',
  'user_pref("browser.startup.homepage_override.mstone", "ignore");',
  'user_pref("browser.sessionstore.resume_from_crash", false);',
  'user_pref("datareporting.policy.dataSubmissionEnabled", false);',
  'user_pref("toolkit.telemetry.reportingpolicy.firstRun", false);',
];

/**
 * Close Gecko windows on the Windows side whose title matches, through the
 * normal close path so Gecko records a clean shutdown. A forceful kill made
 * the throwaway profile open in Troubleshoot Mode on the next run.
 */
export function closeWindowsGeckoWindows(titlePatterns) {
  const match = titlePatterns.map((t) => `$_.MainWindowTitle -like '${t}'`).join(' -or ');
  try {
    execFileSync('powershell.exe', [
      '-NoProfile', '-Command',
      'Get-Process zen,firefox -ErrorAction SilentlyContinue | ' +
      `Where-Object { ${match} } | ` +
      'ForEach-Object { $_.CloseMainWindow() | Out-Null }; Start-Sleep -Milliseconds 900',
    ], { stdio: 'ignore' });
  } catch { /* nothing to close */ }
}

/**
 * Launch `browser` at `url` with a throwaway profile.
 *
 * @param {string} browser
 * @param {string} url
 * @param {object} options
 * @param {boolean} [options.headless]
 * @param {string} options.profileName directory name for the throwaway profile
 * @param {string[]} [options.windowTitles] Gecko-on-Windows titles to close
 * @param {string[]} [options.extraGeckoPrefs]
 * @param {[number, number]} [options.size] window size
 * @param {string[]} [options.extraArgs] more Chrome or Chromium switches
 * @returns {{ child: import('node:child_process').ChildProcess, close(): void }}
 */
export function launchBrowser(browser, url, {
  headless = false,
  profileName,
  windowTitles = [],
  extraGeckoPrefs = [],
  size = [900, 620],
  extraArgs = [],
} = {}) {
  if (headless && isGecko(browser)) {
    throw new Error('Headless runs need Chrome or Chromium; headless Firefox has no WebGPU adapter.');
  }

  let args;
  let scratchProfile = null;
  if (isGecko(browser)) {
    let profile;
    let profileDir;
    if (isWindowsExe(browser)) {
      const local = windowsLocalAppData();
      profile = `${local.win}\\${profileName}`;
      profileDir = `${local.wsl}/${profileName}`;
      mkdirSync(profileDir, { recursive: true });
    } else {
      profileDir = scratchProfile = mkdtempSync(join(tmpdir(), `${profileName}-`));
      profile = profileDir;
    }
    writeFileSync(join(profileDir, 'user.js'), [...GECKO_PREFS, ...extraGeckoPrefs].join('\n') + '\n');
    args = ['--no-remote', '--profile', profile, '--width', String(size[0]), '--height', String(size[1])];
  } else {
    const profile = isWindowsExe(browser)
      ? `${windowsLocalAppData().win}\\${profileName}`
      : (scratchProfile = mkdtempSync(join(tmpdir(), `${profileName}-`)));
    args = [
      '--enable-unsafe-webgpu',
      '--enable-features=Vulkan',
      '--no-first-run',
      '--no-default-browser-check',
      `--user-data-dir=${profile}`,
      `--window-size=${size[0]},${size[1]}`,
    ];
    if (headless) {
      args.push(
        '--headless=new',
        '--enable-unsafe-swiftshader',
        '--use-webgpu-adapter=swiftshader',
        '--use-angle=swiftshader',
      );
    }
    // Chrome refuses to start as root with its sandbox on (containers), and CI
    // runners may not grant the user namespaces the sandbox needs. Both are
    // disposable machines loading this repository's own pages.
    if (process.getuid?.() === 0 || process.env.CI) args.push('--no-sandbox');
    args.push(...extraArgs);
  }

  const child = spawn(browser, [...args, url], { stdio: 'ignore' });

  const close = () => {
    try {
      if (isGecko(browser) && isWindowsExe(browser)) closeWindowsGeckoWindows(windowTitles);
      else child.kill();
    } catch { /* already gone */ }
    if (scratchProfile) {
      const profile = scratchProfile;
      scratchProfile = null;
      const remove = () => rmSync(profile, { recursive: true, force: true });
      if (child.exitCode !== null || child.signalCode !== null) remove();
      else child.once('exit', remove);
    }
  };
  return { child, close };
}
