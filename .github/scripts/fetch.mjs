/**
 * URL 清单下载器 —— 支持 MEGA 分享 / 普通直链
 * 环境变量: URLS(必填) ONLY(可选glob) MAX_MB(可选) OUT(可选)
 */
import fs from "node:fs";
import fsp from "node:fs/promises";
import path from "node:path";
import { File as MegaFile } from "megajs";

const OUT = process.env.OUT || "out";
const ONLY = (process.env.ONLY || "").trim();
const MAX_MB = Number(process.env.MAX_MB || 0);
const MAX_BYTES = MAX_MB > 0 ? MAX_MB * 1024 * 1024 : Infinity;

const log = (...a) => console.log(...a);
const warn = (...a) => console.warn(...a);

function globMatch(name, pattern) {
  if (!pattern) return true;
  const re = new RegExp(
    "^" + pattern.split("").map(c => (c === "*" ? ".*" : c === "?" ? "." : c.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"))).join("") + "$", "i");
  return re.test(name);
}
function safeName(name) {
  return String(name || "unnamed").replace(/[\/\\:*?"<>|]/g, "_").replace(/\.{2,}/g, ".").slice(0, 180) || "unnamed";
}
async function uniquePath(dir, name) {
  const ext = path.extname(name), base = path.basename(name, ext);
  let c = path.join(dir, name), i = 1;
  while (fs.existsSync(c)) c = path.join(dir, `${base}-${i++}${ext}`);
  return c;
}

async function resolveTargets(src) {
  const tokens = String(src || "").split(/[\s,]+/).map(s => s.trim()).filter(Boolean);
  if (!tokens.length) throw new Error("URLS 为空");
  const targets = [];
  for (const token of tokens) {
    if (/mega\.nz\/|mega\.co\.nz\//i.test(token)) { targets.push(token); continue; }
    try {
      const r = await fetch(token, { signal: AbortSignal.timeout(30000) });
      const ct = (r.headers.get("content-type") || "").toLowerCase();
      const text = await r.text();
      const urls = text.split(/\s+/).map(s => s.trim()).filter(s => /^https?:\/\//i.test(s) && !s.startsWith("#"));
      if (r.ok && urls.length && (ct.includes("text") || ct.includes("json") || text.trim().startsWith("http"))) {
        if (/\.(zip|mp4|png|jpe?g|pdf|bin|safetensors|gguf)$/i.test(new URL(token).pathname)) targets.push(token);
        else { log(`📋 清单 ${token} → ${urls.length} 个链接`); targets.push(...urls); }
      } else if (r.ok) targets.push(token);
      else { warn(`⚠️ 清单返回 ${r.status}，按直链处理: ${token}`); targets.push(token); }
    } catch (e) { warn(`⚠️ 抓取失败（${e.message}），按直链处理`); targets.push(token); }
  }
  return [...new Set(targets)];
}

async function downloadHttp(url, results) {
  const r = await fetch(url, { redirect: "follow", signal: AbortSignal.timeout(1800000) });
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  let name = new URL(url).pathname.split("/").pop() || "download.bin";
  const m = (r.headers.get("content-disposition") || "").match(/filename\*?=(?:UTF-8''|")?([^";]+)/i);
  if (m) name = decodeURIComponent(m[1].replace(/"/g, ""));
  const filePath = await uniquePath(OUT, safeName(name));
  const total = Number(r.headers.get("content-length") || 0);
  log(`⬇️  ${name} ${total ? (total / 1048576).toFixed(2) + "MB" : ""}`);
  const stream = fs.createWriteStream(filePath);
  await new Promise((res, rej) => {
    r.body.pipeTo(new WritableStream({
      write(c) { stream.write(Buffer.from(c)); },
      close() { stream.end(res); },
      abort(e) { rej(e); },
    })).catch(rej);
  });
  const size = fs.statSync(filePath).size;
  results.push({ name, path: filePath, size, source: url });
  return size;
}

async function walkMega(node, dir, results, depth = 0) {
  if (depth > 8) return;
  if (typeof node.loadAttributes === "function" && !node.name) {
    try { await node.loadAttributes(); } catch (e) { warn(`⚠️ 读属性失败: ${e.message}`); return; }
  }
  if (node.directory) {
    const sub = path.join(dir, node.name ? safeName(node.name) : "");
    fs.mkdirSync(sub, { recursive: true });
    const children = node.children || [];
    log(`📁 ${node.name || "(根目录)"} → ${children.length} 项`);
    for (const child of children) await walkMega(child, sub, results, depth + 1);
    return;
  }
  const name = node.name || "unnamed";
  if (!globMatch(name, ONLY)) { log(`⏭️  跳过: ${name}`); return; }
  const filePath = await uniquePath(dir, safeName(name));
  const size = node.size || 0;
  log(`⬇️  ${name} ${(size / 1048576).toFixed(2)}MB`);
  for (let attempt = 1; attempt <= 3; attempt++) {
    try {
      await new Promise((res, rej) => {
        const ws = fs.createWriteStream(filePath), rs = node.download();
        rs.on("error", rej); ws.on("error", rej); ws.on("finish", res); rs.pipe(ws);
      });
      const got = fs.statSync(filePath).size;
      if (size && got !== size) throw new Error(`大小不符 ${got}/${size}`);
      results.push({ name, path: filePath, size: got, source: "mega" });
      return;
    } catch (e) {
      warn(`⚠️ 第 ${attempt} 次失败: ${name} → ${e.message}`);
      try { fs.unlinkSync(filePath); } catch {}
      if (attempt === 3) { warn(`❌ 放弃: ${name}`); return; }
      await new Promise(r => setTimeout(r, 2000 * attempt));
    }
  }
}

const main = async () => {
  fs.mkdirSync(OUT, { recursive: true });
  const targets = await resolveTargets(process.env.URLS);
  log(`\n🎯 共 ${targets.length} 个目标\n`);
  const results = [];
  for (const t of targets) {
    try {
      if (/mega\.nz\/(folder|file)\//i.test(t)) {
        const node = MegaFile.fromURL(t);
        await walkMega(node, OUT, results);
      } else await downloadHttp(t, results);
      if (results.reduce((s, r) => s + r.size, 0) > MAX_BYTES) { warn(`⛔ 达到 MAX_MB=${MAX_MB}`); break; }
    } catch (e) { warn(`❌ 目标失败 ${t}: ${e.message}`); }
  }
  const totalBytes = results.reduce((s, r) => s + r.size, 0);
  const manifest = {
    finishedAt: new Date().toISOString(), totalMB: +(totalBytes / 1048576).toFixed(2),
    count: results.length,
    files: results.map(r => ({ name: r.name, sizeMB: +(r.size / 1048576).toFixed(2), source: r.source })),
  };
  await fsp.writeFile(path.join(OUT, "_manifest.json"), JSON.stringify(manifest, null, 1));
  log("\n════════ 完成 ════════");
  log(`文件: ${manifest.count} 个 | 总计: ${manifest.totalMB} MB`);
  for (const f of manifest.files.sort((a, b) => b.sizeMB - a.sizeMB).slice(0, 40)) log(`  ${String(f.sizeMB).padStart(8)} MB  ${f.name}`);
  if (!results.length) { warn("没有下载到任何文件"); process.exitCode = 1; }
};
main().catch(e => { console.error("💥 " + e.message); process.exitCode = 1; });
