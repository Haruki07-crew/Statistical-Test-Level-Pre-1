// dist.js を scipy の参照値（verify/dist_reference.json）と比較する
import fs from 'node:fs';
import vm from 'node:vm';
const root = new URL('..', import.meta.url);
vm.runInThisContext(fs.readFileSync(new URL('src/dist.js', root), 'utf8'));
const ref = JSON.parse(fs.readFileSync(new URL('verify/dist_reference.json', root), 'utf8'));
const NONCENTRAL = new Set(['ncx2', 'nct', 'ncf']);
let ok = 0, ng = 0;
const relErr = (a, b) => Math.abs(a - b) / Math.max(Math.abs(b), 1e-12);
for (const c of ref) {
  const tol = NONCENTRAL.has(c.dist) ? 1e-4 : 1e-6;
  const checks = c.quantile !== undefined
    ? [['quantile', Dist.quantile(c.dist, c.p, c.params), c.quantile]]
    : [['pdf', Dist.pdf(c.dist, c.x, c.params), c.pdf], ['cdf', Dist.cdf(c.dist, c.x, c.params), c.cdf]];
  for (const [kind, got, want] of checks) {
    const e = (Math.abs(want) < 1e-10) ? Math.abs(got - want) : relErr(got, want);
    if (e <= tol) ok++;
    else { ng++; console.log(`NG ${c.dist} ${JSON.stringify(c.params)} x=${c.x ?? c.p} ${kind}: got ${got} want ${want} (err ${e.toExponential(2)})`); }
  }
}
console.log(`dist.js テスト: OK ${ok} / NG ${ng}`);
process.exit(ng ? 1 : 0);
