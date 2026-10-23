// Usage: node brand/render_capi.js <rough.js> web/template.html <out.json>  (rough.js 4.6.6 bundled build)
// Render Capi with the exact drawing code from web/template.html, using a tiny SVG DOM shim.
const fs = require("fs");
const [roughPath, templatePath, outPath] = process.argv.slice(2);
class Node_ {
  constructor(tag) { this.tagName = tag; this.attrs = {}; this.children = []; this.style = {}; this.textContent = ""; this.dataset = {}; this.classList = { add() {} }; }
  setAttribute(k, v) { this.attrs[k] = String(v); }
  getAttribute(k) { return this.attrs[k] ?? null; }
  appendChild(c) { this.children.push(c); return c; }
  querySelectorAll() { return []; }
  get ownerDocument() { return document; }
  toString() {
    const a = Object.entries(this.attrs).map(([k, v]) => ` ${k}="${v.replace(/&/g, "&amp;").replace(/"/g, "&quot;")}"`).join("");
    const t = this.textContent.replace(/&/g, "&amp;").replace(/</g, "&lt;");
    return `<${this.tagName}${a}>${t}${this.children.map(String).join("")}</${this.tagName}>`;
  }
}
global.document = { createElementNS: (_, tag) => new Node_(tag) };
global.window = global; global.matchMedia = () => ({ matches: true });
eval(fs.readFileSync(roughPath, "utf8") + "\nglobal.rough = rough;");
const t = fs.readFileSync(templatePath, "utf8");
const code = t.slice(t.indexOf("/* ---------- doodles: little hand-drawn characters ---------- */"), t.indexOf("function drawDoodles("));
const prelude = `const NS = "http://www.w3.org/2000/svg", INK = "#f2d37b", reduce = true;
const el = (tag, attrs = {}, parent) => { const e = document.createElementNS(NS, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); if (parent) parent.appendChild(e); return e; };`;
eval(prelude + code + `
const out = {};
for (const kind of Object.keys(DOODLES)) { const svg = new Node_("svg"); svg.setAttribute("viewBox", "0 0 260 170"); DOODLES[kind](rough.svg(svg), svg); out[kind] = svg.children.map(String).join(""); }
{ const svg = new Node_("svg"), r = rough.svg(svg);
  svg.appendChild(r.circle(19, 19, 35, { stroke: "#8b97ff", strokeWidth: 1.4, roughness: 1.2, seed: 11 }));
  const g = grp(svg, -28, 1, 0.38), o = { stroke: PEN, strokeWidth: 3, roughness: 0.8, seed: 12 };
  g.appendChild(r.path("M100 27 C100 18 112 17 113 25", { ...o, fill: FUR_HI, fillStyle: "solid" }));
  g.appendChild(r.path(CAPY_HEAD, { ...o, fill: FUR, fillStyle: "solid" }));
  g.appendChild(r.circle(124, 37, 7, { ...o, fill: "#111", fillStyle: "solid" }));
  g.appendChild(r.ellipse(151, 36, 7, 5, { ...o, fill: "#111", fillStyle: "solid" }));
  out.logo = svg.children.map(String).join(""); }
fs.writeFileSync(outPath, JSON.stringify(out));
console.log(Object.entries(out).map(([k, v]) => k + ":" + v.length).join(" "));`);
